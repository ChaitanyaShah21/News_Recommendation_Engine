"""intra-list diversity, novelty and coverage for all four methods.

Measured on TWO different outputs on purpose, and the pair is the point:

  retrieval   top-K chosen by us out of the whole corpus (Q2/Q3's operation).
              This is the headline: it is our system's own output, so its
              diversity, novelty and coverage are ours to answer for.
  re-ranking  top-K of the platform's supplied candidate list (Q4.2's
              operation). Reported to *show* rather than assert that only 6.8%
              of MIND's corpus (19.2% of EB-NeRD's) ever appears in any val
              candidate list, which hard-caps coverage there at someone else's
              recommender's decision.

Held identical to Q2/Q3 so the numbers sit beside the recall figures:
  D12  N = 10 most recent clicked titles form the BM25 query
  D15  the user's own history is excluded from retrieval candidates
  D20-D22  the same embeddings and the same mean-pooled user vector

Read every number here against the accuracy table, never alone: the random arm
scores BEST on all three metrics, so these price what a method gave up rather
than ranking the methods.

Diversity and novelty carry bootstrap 95% intervals (Q4.4). Coverage does not:
the bootstrap is biased downward for a union statistic, so its percentile
interval is not a confidence interval (D27). The raw spread and the pivotal
correction are both kept in the CSV under explicit column names.

    python scripts/run_beyond_accuracy.py [--k 10] [--datasets mind ebnerd]
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

from newsrec.eval import beyond_accuracy as ba  # noqa: E402
from newsrec.eval import bootstrap as boot  # noqa: E402
from newsrec.eval import metrics as met  # noqa: E402
from newsrec.eval import rerank  # noqa: E402
from newsrec.retrieval import bm25, bm25_search, semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
METHODS = ("bm25", "semantic", "popularity", "random")
RANDOM_SEED = 20260825


def _exclude_rows(history: pl.DataFrame, row_of_article: dict[str, int]) -> list[np.ndarray]:
    """D15: each user's whole history, not just the N that formed the query."""
    return [
        np.asarray(
            [row_of_article[a] for a in (ids or []) if a in row_of_article],
            dtype=np.int32,
        )
        for ids in history.get_column("history_article_ids").to_list()
    ]


def _retrieval_lists(
    dataset: str, k: int, n_recent: int
) -> tuple[dict[str, list[np.ndarray]], np.ndarray, np.ndarray, np.ndarray, int]:
    """Top-k retrieved article rows per user, for each of the four methods."""
    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(
        pl.col("dataset") == dataset
    )
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == "val")
    )
    train_impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == "train")
    )
    article_ids = articles.get_column("article_id").to_list()
    categories = articles.get_column("category").to_numpy()

    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    queries = bm25_search.build_queries(history, index, title_term, n_recent=n_recent)
    embed_ids, embeddings = semantic.load_article_embeddings(
        PROCESSED / "embeddings.parquet", dataset=dataset
    )
    users = semantic_search.build_user_vectors(history, embed_ids, embeddings, n_recent=n_recent)

    if index.article_ids != article_ids or embed_ids != article_ids:
        raise ValueError("index / embedding row order does not match the articles table")

    counts = rerank.train_click_counts(train_impressions, article_ids)
    novelty = ba.item_novelty(counts)
    exclude = _exclude_rows(history, {a: i for i, a in enumerate(article_ids)})

    lists: dict[str, list[np.ndarray]] = {}
    t = time.perf_counter()
    lists["bm25"] = [
        np.asarray(r[:k], dtype=np.int32)
        for r in bm25_search.retrieve(index, queries, k=k, exclude_rows=exclude)
    ]
    print(f"    bm25 retrieval      {time.perf_counter() - t:6.1f}s")
    t = time.perf_counter()
    lists["semantic"] = [
        np.asarray(r[:k], dtype=np.int32)
        for r in semantic_search.retrieve(users, embeddings, k=k, exclude_rows=exclude)
    ]
    print(f"    semantic retrieval  {time.perf_counter() - t:6.1f}s")

    # Popularity retrieval: the k most-clicked training articles, minus this
    # user's own history so D15 applies uniformly across methods. Ranked once;
    # the per-user difference is only which of them get excluded.
    ranked = np.argsort(-counts, kind="stable")
    n_articles = len(article_ids)
    rng = np.random.default_rng(RANDOM_SEED)
    pop_lists, rand_lists = [], []
    for drop in exclude:
        mask = np.zeros(n_articles, dtype=bool)
        mask[drop] = True
        pop_lists.append(ranked[~mask[ranked]][:k].astype(np.int32))
        allowed = np.flatnonzero(~mask)
        rand_lists.append(
            rng.choice(allowed, size=min(k, len(allowed)), replace=False).astype(np.int32)
        )
    lists["popularity"] = pop_lists
    lists["random"] = rand_lists

    return lists, embeddings, categories, novelty, n_articles


def _rerank_lists(dataset: str, k: int, n_recent: int):
    """Top-k of each impression's own re-ranked candidate list."""
    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(
        pl.col("dataset") == dataset
    )
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == "val")
    )
    train_impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == "train")
    )
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == "val")
    )
    article_ids = articles.get_column("article_id").to_list()

    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    queries = bm25_search.build_queries(history, index, title_term, n_recent=n_recent)
    embed_ids, embeddings = semantic.load_article_embeddings(
        PROCESSED / "embeddings.parquet", dataset=dataset
    )
    users = semantic_search.build_user_vectors(history, embed_ids, embeddings, n_recent=n_recent)
    candidates = rerank.build_candidate_set(impressions, article_ids, history)
    counts = rerank.train_click_counts(train_impressions, article_ids)

    scored = {
        "bm25": rerank.score_bm25(candidates, index, queries),
        "semantic": rerank.score_semantic(candidates, users, embeddings),
        "popularity": rerank.score_popularity(candidates, counts),
        "random": rerank.score_random(candidates),
    }

    out: dict[str, list[np.ndarray]] = {}
    for method, scores in scored.items():
        lists = []
        for s, y, rows in zip(scores, candidates.labels, candidates.candidate_rows):
            order = met.rank_order(s, y, met.PESSIMISTIC)
            lists.append(rows[order][:k].astype(np.int32))
        out[method] = lists
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    parser.add_argument("--n-recent", type=int, default=bm25_search.DEFAULT_N_RECENT)
    parser.add_argument("--resamples", type=int, default=boot.DEFAULT_RESAMPLES)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} (val, k={args.k}) ===")
        lists, embeddings, categories, novelty, n_articles = _retrieval_lists(
            dataset, args.k, args.n_recent
        )
        rr = _rerank_lists(dataset, args.k, args.n_recent)

        print(f"\n  {'output':<11} {'method':<11} {'ILD-embed':>22} {'ILD-cat':>22} "
              f"{'novelty':>22} {'coverage':>22}")
        for label, group in (("retrieval", lists), ("re-rank", rr)):
            # One index draw shared by every metric in this (dataset, output,
            # method) cell, so the four intervals describe the same resampled
            # worlds rather than four independent ones (Q4.4).
            n_lists = len(group[METHODS[0]])
            draw = boot.draw_indices(n_lists, args.resamples, boot.DEFAULT_SEED)
            for method in METHODS:
                r = ba.evaluate_lists(
                    group[method], embeddings, categories, novelty, n_articles
                )
                cis = {
                    "ild_embedding": boot.bootstrap_mean(r["ild_embedding"], indices=draw),
                    "ild_category": boot.bootstrap_mean(r["ild_category"], indices=draw),
                    "novelty": boot.bootstrap_mean(r["novelty"], indices=draw),
                    # Coverage is a property of the union, not a mean, so it is
                    # recomputed inside each resample rather than averaged.
                    "coverage": boot.bootstrap_coverage(
                        group[method], n_articles, indices=draw
                    ),
                }
                cell = lambda c: f"{c.point:7.3f} [{c.low:6.3f},{c.high:6.3f}]"
                # D27: coverage is printed WITHOUT an interval. The bootstrap is
                # biased downward for a union statistic - a resample holds only
                # 63.2% of the distinct lists, and a duplicate adds no new
                # articles - so its percentile interval sits entirely below the
                # point estimate and is not a confidence interval. Both the raw
                # spread and the pivotal correction are kept in the CSV.
                print(f"  {label:<11} {method:<11} {cell(cis['ild_embedding']):>22} "
                      f"{cell(cis['ild_category']):>22} {cell(cis['novelty']):>22} "
                      f"{cis['coverage'].point * 100:21.3f}%")
                row = {
                    "dataset": dataset,
                    "output": label,
                    "method": method,
                    "k": args.k,
                    "n_lists": r["n_lists"],
                    "catalogue": n_articles,
                }
                for name in ("ild_embedding", "ild_category", "novelty"):
                    ci = cis[name]
                    row[name] = round(ci.point, 5)
                    row[f"{name}_lo"] = round(ci.low, 5)
                    row[f"{name}_hi"] = round(ci.high, 5)
                cov = cis["coverage"]
                piv_lo, piv_hi = cov.pivotal()
                row["coverage"] = round(cov.point, 5)
                row["coverage_resample_lo_BIASED"] = round(cov.low, 5)
                row["coverage_resample_hi_BIASED"] = round(cov.high, 5)
                row["coverage_pivotal_lo"] = round(piv_lo, 5)
                row["coverage_pivotal_hi"] = round(piv_hi, 5)
                rows.append(row)
        print(f"\n  novelty range for this corpus: "
              f"{novelty.min():.2f} (most clicked) to {novelty.max():.2f} (never clicked)")

    REPORTS.mkdir(exist_ok=True)
    name = f"beyond_accuracy_{'-'.join(args.datasets)}_val_k{args.k}.csv"
    pl.DataFrame(rows).write_csv(REPORTS / name)
    print(f"\nwrote {REPORTS / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
