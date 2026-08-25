"""Consolidate every per-run recall CSV - BM25 and semantic - into one table.

This is where Q3.5's lexical-vs-semantic comparison actually gets made, which is
why it reads both methods' outputs rather than one. The `query` column names the
method: `raw_tf`/`binary` are BM25 (D16's ablation), `semantic` is the embedding
run (D20/D21).

An absolute recall@K means little on its own: restricting the candidate pool
(D19) raises recall and raises random-chance recall at the same time. Without
the baseline column, EB-NeRD's 2.45% -> 7.27% reads as a 3x improvement when
almost all of it is the pool shrinking from 11,777 articles to ~2,963.

The baseline is the expected recall of drawing K uniformly from whatever pool
that run retrieved from - the whole corpus, or per-impression the set already
in circulation at that impression's time bucket.

    python scripts/summarise_recall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.retrieval import availability  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
K_VALUES = (50, 100, 200)


def random_baselines(dataset: str, split: str, bucket: str = "1h") -> dict:
    """Expected recall@K of a uniform draw, per pool, for one dataset."""
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        pl.col("dataset") == dataset
    )
    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(
        pl.col("dataset") == dataset
    )
    article_ids = articles.get_column("article_id").to_list()

    first_seen = availability.first_seen_times(impressions)
    seen_at = dict(
        zip(
            first_seen.get_column("article_id").to_list(),
            first_seen.get_column("first_seen").to_list(),
        )
    )
    article_first_seen = np.array(
        [seen_at.get(a) or availability.NEVER_SEEN for a in article_ids],
        dtype="datetime64[us]",
    )

    evaluated = impressions.filter(pl.col("split") == split).with_columns(
        pl.col("timestamp").dt.truncate(bucket).alias("bucket_start")
    )
    pool_size = {
        start: int((article_first_seen < np.datetime64(start, "us")).sum())
        for start in evaluated.get_column("bucket_start").unique().to_list()
    }
    # One entry per ground-truth click, so the average is weighted the same way
    # micro-recall is - a click in a bucket with a big pool is genuinely harder.
    per_click_bucket = (
        evaluated.select("bucket_start", "clicked_article_ids")
        .explode("clicked_article_ids", empty_as_null=True)
        .drop_nulls("clicked_article_ids")
        .get_column("bucket_start")
        .to_list()
    )

    out = {}
    for k in K_VALUES:
        out[("whole-corpus", k)] = k / len(article_ids)
        out[("available", k)] = float(
            np.mean(
                [min(k, pool_size[b]) / pool_size[b] for b in per_click_bucket]
            )
        )
    return out


def main() -> int:
    paths = sorted(REPORTS.glob("bm25_recall_*_val_*.csv")) + sorted(
        REPORTS.glob("semantic_recall_*_val_*.csv")
    )
    # Guard against the summary silently under-reporting: a missing run shows up
    # as absent rows in the final table, which is exactly how four EB-NeRD rows
    # went missing in Q2 (see PROGRESS.md's error log).
    print(f"reading {len(paths)} per-run CSVs:")
    for p in paths:
        print(f"  {p.name}")
    frames = [pl.read_csv(p) for p in paths]
    if not frames:
        print("No per-run CSVs found; run the run_*_recall.py scripts first.")
        return 1

    table = pl.concat(frames, how="diagonal").filter(pl.col("slice") == "has-query")

    baselines: dict[str, dict] = {}
    for dataset in table.get_column("dataset").unique().to_list():
        baselines[dataset] = random_baselines(dataset, "val")

    table = table.with_columns(
        pl.struct(["dataset", "pool", "k"])
        .map_elements(
            lambda s: baselines[s["dataset"]][(s["pool"], s["k"])],
            return_dtype=pl.Float64,
        )
        .alias("random_baseline")
    ).with_columns(
        (pl.col("recall@k (macro)") / pl.col("random_baseline"))
        .round(2)
        .alias("lift_over_random")
    )

    header = f"{'dataset':<9}{'pool':<14}{'query':<9}{'K':>5}{'recall':>9}{'random':>9}{'lift':>7}"
    print(header)
    print("-" * len(header))
    for row in table.sort("dataset", "pool", "query", "k").iter_rows(named=True):
        print(
            f"{row['dataset']:<9}{row['pool']:<14}{row['query']:<9}{row['k']:>5}"
            f"{row['recall@k (macro)'] * 100:>8.2f}%"
            f"{row['random_baseline'] * 100:>8.2f}%"
            f"{row['lift_over_random']:>6.2f}x"
        )

    out_path = REPORTS / "recall_summary.csv"
    table.sort("dataset", "pool", "query", "k").write_csv(out_path)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
