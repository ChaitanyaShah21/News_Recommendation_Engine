"""Build the Q1 feature table for every (dataset, split) and cache it to disk.

Each build pays a fixed ~35 s (MIND) / ~75 s (EB-NeRD) for the BM25 index and
three decayed-vector passes, so the re-ranker (Q2), the ablations (Q3) and the
extended evaluation (Q5) all read these cached files instead of rebuilding.

    python scripts/build_feature_tables.py                    # all six
    python scripts/build_feature_tables.py --datasets mind --splits val

Output: data/processed/features/{dataset}_{split}.parquet (gitignored).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from newsrec.features import assemble  # noqa: E402
from newsrec.retrieval import semantic  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "features"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == ds)
        hist = pl.read_parquet(PROCESSED / "history.parquet").filter(pl.col("dataset") == ds)
        arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == ds)
        ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset=ds)
        for split in args.splits:
            t0 = time.perf_counter()
            target = imps.filter(pl.col("split") == split)
            ft = assemble.build_feature_table(ds, target, imps, hist, arts, ids, emb)
            path = OUT_DIR / f"{ds}_{split}.parquet"
            ft.write_parquet(path)
            print(f"{ds} {split}: {target.height:,} impressions -> {ft.height:,} rows, "
                  f"label rate {ft[assemble.LABEL].mean():.4f}, "
                  f"{time.perf_counter() - t0:.0f} s -> {path.relative_to(REPO_ROOT)}",
                  flush=True)


if __name__ == "__main__":
    main()
