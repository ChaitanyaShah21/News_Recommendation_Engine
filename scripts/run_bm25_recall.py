"""report recall@K for BM25 retrieval on both datasets.

Thin entry point: all logic lives in newsrec.retrieval / newsrec.eval. Reads
the feature store, builds one index per dataset (D11/D14), builds one query per
user from the titles of their last N clicks (D12), excludes each user's own
history from their candidates (D15), and reports recall@{50,100,200} macro and
micro (D18) both over impressions that have a query and over all impressions
with query-less users counted as misses (D17).

Also runs D16's ablation: raw query term frequency (default) versus binary.

    python scripts/run_bm25_recall.py [--split val] [--n-recent 10]
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
from newsrec.retrieval import availability, bm25, bm25_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
K_VALUES = (50, 100, 200)


def build_availability(
    dataset: str, impressions: pl.DataFrame, index, bucket: str
):
    """D19 masks for BM25, in the float32 0/1 form its multiplicative mask needs.

    The logic itself now lives in `newsrec.retrieval.availability`, shared with
    semantic retrieval - two copies would drift, and a fix applied to one and not
    the other would silently corrupt Q3.5's lexical-vs-semantic comparison.

    The cast to float32 is here rather than there because the two retrievers mask
    differently: BM25 multiplies by 0/1 (correct, since 0 is its minimum possible
    score), while dense cosine scoring must mask with -inf (0 is mid-range there,
    not the floor).
    """
    all_impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        pl.col("dataset") == dataset
    )
    bucketed, bucket_id, masks = availability.build_availability(
        all_impressions, impressions, index.article_ids, bucket
    )
    return bucketed, bucket_id, [m.astype(np.float32) for m in masks]


def run_dataset(
    dataset: str,
    split: str,
    n_recent: int,
    binary: bool,
    availability: bool = False,
    bucket: str = "1h",
) -> list[dict]:
    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(
        pl.col("dataset") == dataset
    )
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == split)
    )
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == split)
    )

    t0 = time.perf_counter()
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    queries = bm25_search.build_queries(
        history, index, title_term, n_recent=n_recent, binary=binary
    )
    t_index = time.perf_counter() - t0

    # D15: a user's own history is removed from their candidates. Built from the
    # FULL history, not just the n_recent used to form the query - the point is
    # "don't recommend what they've read", which is not limited to the last 10.
    row_of_article = {aid: i for i, aid in enumerate(index.article_ids)}
    exclude = [
        np.asarray(
            [row_of_article[a] for a in (ids or []) if a in row_of_article],
            dtype=np.int32,
        )
        for ids in history.get_column("history_article_ids").to_list()
    ]

    query_row_of_user = {u: i for i, u in enumerate(queries.user_ids)}
    has_query = dict(zip(queries.user_ids, queries.has_query))
    pool_note = f"{index.n_docs:,} (whole corpus)"

    t0 = time.perf_counter()
    if availability:
        bucketed, bucket_id, masks = build_availability(
            dataset, impressions, index, bucket
        )
        # One task per (user, time-bucket): the query is fixed per user, but
        # what is *available* to retrieve is not, so the same user must be
        # re-ranked in each bucket they appear in.
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
        top_per_task = bm25_search.retrieve_bucketed(
            index,
            queries,
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
            [index.article_ids[r] for r in top_per_task[t]] for t in impression_task
        ]
        avail_sizes = [int(m.sum()) for m in masks]
        pool_note = (
            f"{min(avail_sizes):,}-{max(avail_sizes):,} available per "
            f"{bucket} bucket ({len(masks)} buckets, {len(tasks):,} tasks)"
        )
    else:
        top = bm25_search.retrieve(
            index, queries, k=max(K_VALUES), exclude_rows=exclude
        )
        by_user = {
            user: [index.article_ids[r] for r in rows]
            for user, rows in zip(queries.user_ids, top)
        }
        retrieved_per_impression = [
            by_user.get(u, []) for u in impressions.get_column("user_id").to_list()
        ]
    t_retrieve = time.perf_counter() - t0

    users = impressions.get_column("user_id").to_list()
    clicked = impressions.get_column("clicked_article_ids").to_list()

    warm = [i for i, u in enumerate(users) if has_query.get(u, False)]
    n_cold = len(users) - len(warm)

    variant = "binary" if binary else "raw_tf"
    pool = "available" if availability else "whole-corpus"
    print(f"\n=== {dataset} / {split} / {pool} / {variant} query terms ===")
    print(
        f"  pool: {pool_note}; {len(queries.user_ids):,} users, "
        f"{len(users):,} impressions"
    )
    print(
        f"  {n_cold:,} impressions ({n_cold / len(users) * 100:.1f}%) have a "
        f"query-less user"
    )
    print(f"  index+queries {t_index:.1f}s, retrieval {t_retrieve:.1f}s")

    lengths = [len(r) for r in retrieved_per_impression]
    print(
        f"  retrieved list length: mean {np.mean(lengths):.1f}, min {min(lengths)}, "
        f"{sum(1 for n in lengths if n < max(K_VALUES)):,} impressions matched "
        f"fewer than {max(K_VALUES)} articles"
    )

    rows = []
    for k in K_VALUES:
        # D17 headline: only impressions whose user actually has a query.
        r_warm = recall_at_k(
            [clicked[i] for i in warm], [retrieved_per_impression[i] for i in warm], k
        )
        # D17 second number: everyone, query-less users retrieving nothing.
        r_all = recall_at_k(clicked, retrieved_per_impression, k)
        for result, label in ((r_warm, "has-query"), (r_all, "all-impressions")):
            rows.append(
                {
                    "dataset": dataset,
                    "pool": pool,
                    "query": variant,
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
    parser.add_argument("--n-recent", type=int, default=bm25_search.DEFAULT_N_RECENT)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    parser.add_argument(
        "--availability",
        action="store_true",
        help="D19: restrict candidates to articles already in circulation at "
        "the impression's time bucket",
    )
    parser.add_argument("--bucket", default="1h")
    parser.add_argument(
        "--query-variants",
        nargs="+",
        default=["raw_tf", "binary"],
        choices=["raw_tf", "binary"],
        help="D16 ablation; raw_tf alone skips the binary run",
    )
    parser.add_argument("--tag", default="", help="suffix for the output csv name")
    args = parser.parse_args()

    if not (PROCESSED / "articles.parquet").exists():
        print(
            f"Feature store not found at {PROCESSED}. Run "
            "`python scripts/build_pipeline.py` first.",
            file=sys.stderr,
        )
        return 1

    all_rows = []
    for dataset in args.datasets:
        for variant in args.query_variants:
            all_rows += run_dataset(
                dataset,
                args.split,
                args.n_recent,
                binary=(variant == "binary"),
                availability=args.availability,
                bucket=args.bucket,
            )

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    # Every varying input goes in the filename. An earlier version omitted the
    # dataset, so running MIND after EB-NeRD silently overwrote EB-NeRD's
    # results - no error, and the numbers still looked right on screen.
    pool = "available" if args.availability else "wholecorpus"
    out_path = out_dir / (
        f"bm25_recall_{'-'.join(args.datasets)}_{args.split}"
        f"_n{args.n_recent}_{pool}{args.tag}.csv"
    )
    pl.DataFrame(all_rows).write_csv(out_path)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
