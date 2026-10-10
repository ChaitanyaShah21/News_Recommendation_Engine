"""re-rank each impression's own candidate list and score it.

Deliberately the same shape as `run_bm25_recall.py` and `run_semantic_recall.py`,
for the same reason: four scorers run through one harness means the numbers
compare the scorers, not four differently-built pipelines.

What differs from Q2/Q3 is the candidate set, not the scoring. Q2 and Q3
retrieved top-K from the whole corpus and reported recall@K. Here the platform's
own candidate list is re-ranked, because AUC, MRR and nDCG need a per-item
clicked/not-clicked label and only the shown list carries one.

Held identical to the retrieval runs:
  D12  N = 10 most recent clicked article titles form the BM25 query
  D17  `has_query` carried per impression so cold-start can be reported both ways
  D20-D22  the same embeddings and the same mean-pooled user vector

Deliberately NOT carried over:
  D15  history exclusion - the platform chose the candidates; see rerank.py
  D19  availability - the candidates were already in circulation by definition,
       so restricting them further would re-decide a decision the log records

Writes one row per (impression, method) with the per-impression metric values
kept intact, because Q4.4's bootstrap resamples impressions and a mean has
already thrown away what it needs.

    python scripts/run_rerank_eval.py [--split val] [--datasets mind ebnerd]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.eval import metrics as met  # noqa: E402
from newsrec.eval import rerank  # noqa: E402
from newsrec.retrieval import bm25, bm25_search, semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
METHODS = ("bm25", "semantic", "popularity", "random")


def evaluate_dataset(dataset: str, split: str, n_recent: int) -> pl.DataFrame:
    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(
        pl.col("dataset") == dataset
    )
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == split)
    )
    train_impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == "train")
    )
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == split)
    )

    article_ids = articles.get_column("article_id").to_list()
    print(f"  {len(impressions):,} impressions, {len(article_ids):,} articles")

    t0 = time.perf_counter()
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    queries = bm25_search.build_queries(history, index, title_term, n_recent=n_recent)

    embed_ids, embeddings = semantic.load_article_embeddings(
        PROCESSED / "embeddings.parquet", dataset=dataset
    )
    users = semantic_search.build_user_vectors(
        history, embed_ids, embeddings, n_recent=n_recent
    )

    # One CandidateSet is shared by both scorers, so both must agree on what row
    # i means. D22 chose a storage format that makes misalignment impossible by
    # construction; this asserts it anyway, because a silent mismatch would not
    # raise - it would just score every candidate against the wrong article.
    if index.article_ids != article_ids:
        raise ValueError("BM25 index row order does not match the articles table")
    if embed_ids != article_ids:
        raise ValueError("embedding row order does not match the articles table")

    candidates = rerank.build_candidate_set(impressions, article_ids, history)
    counts = rerank.train_click_counts(train_impressions, article_ids)
    print(f"  setup {time.perf_counter() - t0:.1f}s")

    scored = {}
    for method in METHODS:
        t = time.perf_counter()
        if method == "bm25":
            scored[method] = rerank.score_bm25(candidates, index, queries)
        elif method == "semantic":
            scored[method] = rerank.score_semantic(candidates, users, embeddings)
        elif method == "popularity":
            scored[method] = rerank.score_popularity(candidates, counts)
        else:
            scored[method] = rerank.score_random(candidates)
        print(f"  scored {method:<11} {time.perf_counter() - t:6.1f}s")

    # `has_query` is a property of the user, identical for BM25 and semantic
    # (both fall back to an all-zero score row), so it is recorded once.
    query_rows, present = rerank._rows_by_query(candidates.user_ids, queries.user_ids)
    has_query = present & queries.has_query[np.clip(query_rows, 0, None)]

    n_candidates = np.array([len(r) for r in candidates.candidate_rows], dtype=np.int32)
    n_clicks = np.array([int(y.sum()) for y in candidates.labels], dtype=np.int32)

    frames = []
    for method, scores in scored.items():
        pess = met.evaluate_impressions(scores, candidates.labels, tie=met.PESSIMISTIC)
        opt = met.evaluate_impressions(scores, candidates.labels, tie=met.OPTIMISTIC)
        frames.append(
            pl.DataFrame(
                {
                    "dataset": dataset,
                    "split": split,
                    "method": method,
                    "impression_id": candidates.impression_ids,
                    "user_id": candidates.user_ids,
                    "n_candidates": n_candidates,
                    "n_clicks": n_clicks,
                    "history_len": candidates.history_len,
                    "has_query": has_query,
                    # AUC carries its own tie rule, so it has one value, not two.
                    "auc": pess.auc,
                    "mrr": pess.mrr,
                    "ndcg@5": pess.ndcg_at_5,
                    "ndcg@10": pess.ndcg_at_10,
                    "mrr_optimistic": opt.mrr,
                    "ndcg@5_optimistic": opt.ndcg_at_5,
                    "ndcg@10_optimistic": opt.ndcg_at_10,
                }
            )
        )

    return pl.concat(frames)


def summarise(frame: pl.DataFrame) -> None:
    """Headline table: macro mean per method over impressions with a query (D17)."""
    for dataset in frame.get_column("dataset").unique(maintain_order=True):
        d = frame.filter(pl.col("dataset") == dataset)
        print(f"\n{dataset} - macro mean over impressions with a query (D17 headline)")
        print(f"  {'method':<11} {'AUC':>7} {'MRR':>7} {'nDCG@5':>8} {'nDCG@10':>8}  "
              f"{'tie gap':>8}  {'n':>7}")
        for method in METHODS:
            m = d.filter((pl.col("method") == method) & pl.col("has_query"))
            vals = {c: met.macro_mean(m.get_column(c).to_numpy()) for c in
                    ("auc", "mrr", "ndcg@5", "ndcg@10", "ndcg@10_optimistic")}
            gap = vals["ndcg@10_optimistic"][0] - vals["ndcg@10"][0]
            print(f"  {method:<11} {vals['auc'][0]:7.4f} {vals['mrr'][0]:7.4f} "
                  f"{vals['ndcg@5'][0]:8.4f} {vals['ndcg@10'][0]:8.4f}  "
                  f"{gap:+8.4f}  {vals['auc'][1]:7,}")

        cold = d.filter(~pl.col("has_query"))
        n_cold = cold.filter(pl.col("method") == "bm25").height
        print(f"  cold-start impressions excluded from the above: {n_cold:,} "
              f"({n_cold / (d.height / len(METHODS)) * 100:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    parser.add_argument("--n-recent", type=int, default=bm25_search.DEFAULT_N_RECENT)
    args = parser.parse_args()

    frames = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} ({args.split}) ===")
        frames.append(evaluate_dataset(dataset, args.split, args.n_recent))

    frame = pl.concat(frames)
    summarise(frame)

    REPORTS.mkdir(exist_ok=True)
    # Every varying input goes in the filename - during Q2,
    # `--datasets mind` silently overwrote `--datasets ebnerd`'s results.
    name = f"rerank_{'-'.join(args.datasets)}_{args.split}_n{args.n_recent}.parquet"
    frame.write_parquet(REPORTS / name)
    print(f"\nwrote {REPORTS / name}  ({frame.height:,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
