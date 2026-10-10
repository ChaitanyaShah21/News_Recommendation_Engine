"""report recall@K for embedding-based retrieval on both datasets.

Thin entry point, deliberately the same shape as `run_bm25_recall.py` so the two
sets of numbers are produced by the same procedure and Q3.5 compares the
retrieval methods rather than two differently-built harnesses.

Held identical to the BM25 run on purpose:
  D12  N = 10 most recent clicked articles form the query
  D15  the user's own history is excluded from their candidates
  D17  headline over impressions whose user has a query; all-impressions second
  D18  macro headline, micro alongside
  D19  run under both the whole-corpus and in-circulation candidate pools

What differs is only the matching: cosine similarity over dense vectors (D20)
searched by exact brute force (D21), instead of BM25 over a sparse index.

    python scripts/run_semantic_recall.py [--availability] [--split val]
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

from newsrec.eval.recall import recall_at_k  # noqa: E402
from newsrec.retrieval import availability as avail_mod  # noqa: E402
from newsrec.retrieval import semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
K_VALUES = (50, 100, 200)


def run_dataset(
    dataset: str,
    split: str,
    n_recent: int,
    use_availability: bool = False,
    bucket: str = "1h",
) -> list[dict]:
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == split)
    )
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == split)
    )

    t0 = time.perf_counter()
    article_ids, embeddings = semantic.load_article_embeddings(
        PROCESSED / "embeddings.parquet", dataset=dataset
    )
    users = semantic_search.build_user_vectors(
        history, article_ids, embeddings, n_recent=n_recent
    )
    t_setup = time.perf_counter() - t0

    # D15, built from the FULL history rather than just the n_recent used for the
    # query: the point is "do not recommend what they have already read", which
    # is not limited to the last 10. This binds harder here than it did for BM25 -
    # the mean-pooled user vector is provably the point of maximum average
    # similarity to exactly these articles, and each scores 1.000 against itself.
    row_of_article = {aid: i for i, aid in enumerate(article_ids)}
    exclude = [
        np.asarray(
            [row_of_article[a] for a in (ids or []) if a in row_of_article],
            dtype=np.int32,
        )
        for ids in history.get_column("history_article_ids").to_list()
    ]

    query_row_of_user = {u: i for i, u in enumerate(users.user_ids)}
    has_query = dict(zip(users.user_ids, users.has_query))
    pool_note = f"{len(article_ids):,} (whole corpus)"

    t0 = time.perf_counter()
    if use_availability:
        all_impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
            pl.col("dataset") == dataset
        )
        # Boolean masks used directly - dense cosine scoring masks with -inf,
        # not by multiplying with 0/1, because 0 is mid-range for cosine.
        bucketed, bucket_id, masks = avail_mod.build_availability(
            all_impressions, impressions, article_ids, bucket
        )
        tasks = bucketed.select("user_id", "bucket_start").unique(maintain_order=True)
        task_key = {
            (u, b): i
            for i, (u, b) in enumerate(
                zip(
                    tasks.get_column("user_id").to_list(),
                    tasks.get_column("bucket_start").to_list(),
                )
            )
        }
        task_query_row = np.array(
            [query_row_of_user[u] for u in tasks.get_column("user_id").to_list()],
            dtype=np.int64,
        )
        task_bucket = np.array(
            [bucket_id[b] for b in tasks.get_column("bucket_start").to_list()],
            dtype=np.int64,
        )
        top_per_task = semantic_search.retrieve_bucketed(
            users,
            embeddings,
            task_query_row,
            task_bucket,
            masks,
            k=max(K_VALUES),
            exclude_rows=exclude,
        )
        impression_task = [
            task_key[(u, b)]
            for u, b in zip(
                bucketed.get_column("user_id").to_list(),
                bucketed.get_column("bucket_start").to_list(),
            )
        ]
        retrieved_per_impression = [
            [article_ids[r] for r in top_per_task[t]] for t in impression_task
        ]
        avail_sizes = [int(m.sum()) for m in masks]
        pool_note = (
            f"{min(avail_sizes):,}-{max(avail_sizes):,} available per "
            f"{bucket} bucket ({len(masks)} buckets, {len(tasks):,} tasks)"
        )
    else:
        top = semantic_search.retrieve(
            users, embeddings, k=max(K_VALUES), exclude_rows=exclude
        )
        by_user = {
            user: [article_ids[r] for r in rows]
            for user, rows in zip(users.user_ids, top)
        }
        retrieved_per_impression = [
            by_user.get(u, []) for u in impressions.get_column("user_id").to_list()
        ]
    t_retrieve = time.perf_counter() - t0

    impression_users = impressions.get_column("user_id").to_list()
    clicked = impressions.get_column("clicked_article_ids").to_list()

    warm = [i for i, u in enumerate(impression_users) if has_query.get(u, False)]
    n_cold = len(impression_users) - len(warm)

    pool = "available" if use_availability else "whole-corpus"
    print(f"\n=== {dataset} / {split} / {pool} / semantic ===")
    print(
        f"  pool: {pool_note}; {len(users.user_ids):,} users, "
        f"{len(impression_users):,} impressions"
    )
    print(
        f"  {n_cold:,} impressions ({n_cold / len(impression_users) * 100:.1f}%) "
        f"have a query-less user"
    )
    print(f"  load+pool {t_setup:.1f}s, retrieval {t_retrieve:.1f}s")

    lengths = [len(r) for r in retrieved_per_impression]
    print(
        f"  retrieved list length: mean {np.mean(lengths):.1f}, min {min(lengths)}, "
        f"{sum(1 for n in lengths if n < max(K_VALUES)):,} impressions matched "
        f"fewer than {max(K_VALUES)} articles"
    )

    rows = []
    for k in K_VALUES:
        r_warm = recall_at_k(
            [clicked[i] for i in warm], [retrieved_per_impression[i] for i in warm], k
        )
        r_all = recall_at_k(clicked, retrieved_per_impression, k)
        for result, label in ((r_warm, "has-query"), (r_all, "all-impressions")):
            rows.append(
                {
                    "dataset": dataset,
                    "pool": pool,
                    "query": "semantic",
                    **result.as_row(label),
                }
            )
        print(
            f"  recall@{k:<4} has-query: macro {r_warm.macro:.4f}  "
            f"micro {r_warm.micro:.4f}   |   "
            f"all: macro {r_all.macro:.4f}  micro {r_all.micro:.4f}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--n-recent", type=int, default=semantic_search.DEFAULT_N_RECENT
    )
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    parser.add_argument(
        "--availability",
        action="store_true",
        help="D19: restrict candidates to articles already in circulation",
    )
    parser.add_argument("--bucket", default="1h")
    parser.add_argument("--tag", default="", help="suffix for the output csv name")
    args = parser.parse_args()

    if not (PROCESSED / "embeddings.parquet").exists():
        print(
            f"Embeddings not found at {PROCESSED / 'embeddings.parquet'}. Run "
            "`python scripts/build_embeddings.py` first.",
            file=sys.stderr,
        )
        return 1

    all_rows = []
    for dataset in args.datasets:
        all_rows += run_dataset(
            dataset,
            args.split,
            args.n_recent,
            use_availability=args.availability,
            bucket=args.bucket,
        )

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    # Every varying input in the filename - during Q2 an
    # omitted dataset name silently overwrote a previous run's results.
    pool = "available" if args.availability else "wholecorpus"
    out_path = out_dir / (
        f"semantic_recall_{'-'.join(args.datasets)}_{args.split}"
        f"_n{args.n_recent}_{pool}{args.tag}.csv"
    )
    pl.DataFrame(all_rows).write_csv(out_path)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
