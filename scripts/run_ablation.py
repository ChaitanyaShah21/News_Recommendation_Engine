"""the serving-time ablation.

Answers *"report metrics with and without features unavailable at serving time"*
by running five arms through the Q4 harness unchanged, so the only thing varying
is the feature:

  popularity (train)      SAFE   click counts from the training window
  popularity (FUTURE)     LEAK   click counts from the window being evaluated
  semantic                SAFE   the Q3 scorer, our best honest system
  semantic + seen-before  SAFE   plus "has this user already read this candidate"
  semantic + FUTURE pop   LEAK   plus val-window click counts

The two popularity arms are the cleanest comparison in the file: identical
algorithm, identical candidates, differing only in whether the clicks being
counted had happened yet when the recommendation was served.

    python scripts/run_ablation.py [--datasets mind ebnerd]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.eval import ablation, bootstrap as boot, metrics as met, rerank  # noqa: E402
from newsrec.retrieval import semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def run(dataset: str, split: str, n_recent: int, resamples: int) -> list[dict]:
    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(
        pl.col("dataset") == dataset
    )
    all_impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        pl.col("dataset") == dataset
    )
    impressions = all_impressions.filter(pl.col("split") == split)
    train_impressions = all_impressions.filter(pl.col("split") == "train")
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == split)
    )
    article_ids = articles.get_column("article_id").to_list()

    embed_ids, embeddings = semantic.load_article_embeddings(
        PROCESSED / "embeddings.parquet", dataset=dataset
    )
    users = semantic_search.build_user_vectors(history, embed_ids, embeddings, n_recent=n_recent)
    if embed_ids != article_ids:
        raise ValueError("embedding row order does not match the articles table")

    candidates = rerank.build_candidate_set(impressions, article_ids, history)
    train_counts = rerank.train_click_counts(train_impressions, article_ids)
    # The leaky feature. Quarantined in ablation.py; nothing else imports it.
    future_counts = ablation.future_click_counts(impressions, article_ids)

    row_of_article = {a: i for i, a in enumerate(article_ids)}
    hist_of_user = {
        u: np.asarray([row_of_article[a] for a in (ids or []) if a in row_of_article],
                      dtype=np.int32)
        for u, ids in zip(
            history.get_column("user_id").to_list(),
            history.get_column("history_article_ids").to_list(),
        )
    }
    history_rows = [
        hist_of_user.get(u, np.zeros(0, dtype=np.int32)) for u in candidates.user_ids
    ]

    semantic_scores = rerank.score_semantic(candidates, users, embeddings)
    seen_before = ablation.seen_before_feature(candidates.candidate_rows, history_rows)
    future_pop = [
        ablation.rank_normalise(future_counts[rows]) for rows in candidates.candidate_rows
    ]

    arms = {
        "popularity (train)": (rerank.score_popularity(candidates, train_counts), "safe"),
        "popularity (FUTURE)": (rerank.score_popularity(candidates, future_counts), "LEAK"),
        "semantic": (semantic_scores, "safe"),
        "semantic + seen-before": (ablation.blend(semantic_scores, seen_before), "safe"),
        "semantic + FUTURE pop": (ablation.blend(semantic_scores, future_pop), "LEAK"),
    }

    # `has_query` per impression, matching D17's headline population. A user
    # absent from the history table is query-less, same as an empty history.
    user_row = {u: i for i, u in enumerate(users.user_ids)}
    has_query = np.array(
        [u in user_row and bool(users.has_query[user_row[u]]) for u in candidates.user_ids]
    )

    n = int(has_query.sum())
    draw = boot.draw_indices(n, resamples, boot.DEFAULT_SEED)
    keep = np.flatnonzero(has_query)

    print(f"\n=== {dataset} ({split}, n = {n:,} impressions with a query) ===")
    print(f"  {'arm':<24} {'':<5} " + " ".join(f"{m:>24}" for m in METRICS))

    rows = []
    baseline = {}
    for name, (scores, status) in arms.items():
        m = met.evaluate_impressions(
            [scores[i] for i in keep], [candidates.labels[i] for i in keep]
        )
        values = m.as_dict()
        cells, record = [], {"dataset": dataset, "arm": name, "serving_time": status, "n": n}
        for metric, key in zip(METRICS, ("AUC", "MRR", "nDCG@5", "nDCG@10")):
            ci = boot.bootstrap_mean(values[key], indices=draw)
            cells.append(f"{ci.point:6.4f} [{ci.low:6.4f},{ci.high:6.4f}]")
            record[metric] = round(ci.point, 4)
            record[f"{metric}_lo"] = round(ci.low, 4)
            record[f"{metric}_hi"] = round(ci.high, 4)
        if name == "semantic":
            baseline = {k: record[k] for k in METRICS}
        print(f"  {name:<24} {status:<5} " + " ".join(f"{c:>24}" for c in cells))
        rows.append(record)

    # What each arm gains over the honest best system.
    print(f"\n  {'arm':<24} {'':<5} " + " ".join(f"{'d ' + m:>12}" for m in METRICS))
    for record in rows:
        deltas = " ".join(
            f"{record[m] - baseline[m]:+12.4f}" if baseline else f"{'-':>12}"
            for m in METRICS
        )
        print(f"  {record['arm']:<24} {record['serving_time']:<5} {deltas}")
        for m in METRICS:
            record[f"{m}_delta_vs_semantic"] = round(record[m] - baseline.get(m, float('nan')), 4)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-recent", type=int, default=10)
    parser.add_argument("--resamples", type=int, default=boot.DEFAULT_RESAMPLES)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        rows.extend(run(dataset, args.split, args.n_recent, args.resamples))

    REPORTS.mkdir(exist_ok=True)
    name = f"ablation_{'-'.join(args.datasets)}_{args.split}.csv"
    pl.DataFrame(rows).write_csv(REPORTS / name)
    print(f"\nwrote {REPORTS / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
