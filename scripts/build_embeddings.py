"""embed every article in the feature store into embeddings.parquet.

Deliberately NOT part of `build_pipeline.py`. Embedding takes ~15 minutes on CPU
while the rest of the rebuild takes 2.81 s, and `build_pipeline.py` rewrites
articles.parquet on every run - so folding this in would either make Q1.5's
one-command rebuild 300x slower or silently destroy the vectors each time.
That separation is the whole reason D22 chose a standalone file.

    python scripts/build_embeddings.py                # both datasets
    python scripts/build_embeddings.py --datasets mind
"""

import argparse
import sys
import time
from pathlib import Path

# Anchor everything to the repo root, never to the caller's working directory.
# Phase 1 shipped this bug once already: the sys.path fix made the *import*
# location-independent while plain relative paths still resolved against $PWD,
# so the script worked from the repo root and crashed from /tmp.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from newsrec.retrieval.semantic import MODEL_NAME, build_article_embeddings  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
ARTICLES = PROCESSED / "articles.parquet"
OUTPUT = PROCESSED / "embeddings.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="embed only the first N articles (for timing runs)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not ARTICLES.exists():
        print(f"error: {ARTICLES} not found.\n"
              f"Run `python scripts/build_pipeline.py` first to build the feature store.",
              file=sys.stderr)
        return 1

    articles = pl.read_parquet(ARTICLES).filter(pl.col("dataset").is_in(args.datasets))
    if args.limit:
        articles = articles.head(args.limit)
    if articles.is_empty():
        print(f"error: no articles for datasets {args.datasets}", file=sys.stderr)
        return 1

    counts = dict(articles["dataset"].value_counts().iter_rows())
    print(f"model    : {MODEL_NAME}")
    print(f"articles : {articles.height:,}  {counts}")
    print("embedding (CPU, no GPU - expect roughly 87 articles/s)...")

    t0 = time.perf_counter()
    embeddings = build_article_embeddings(
        articles, batch_size=args.batch_size, show_progress=True
    )
    elapsed = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    embeddings.write_parquet(args.output)
    size_mb = args.output.stat().st_size / 1e6

    print(f"\ndone in {elapsed / 60:.1f} min ({articles.height / elapsed:.0f} articles/s)")
    print(f"wrote {embeddings.height:,} vectors -> {args.output}  ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
