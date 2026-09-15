#!/usr/bin/env python3
"""A2 / D33: rebuild embeddings.parquet for the ebnerd_large store without re-embedding.

The feature store's EB-NeRD articles now come from ebnerd_large, whose
articles.parquet is frame-equal to the testset's (verified 2026-09-15). Phase 5
already embedded every one of those 125,541 articles into
data/processed/submission/embeddings_ebnerd.parquet, so the store's embeddings
are assembled from two existing files instead of ~18 min of CPU:

    MIND rows    <- the current embeddings.parquet (MIND is unchanged by D33)
    EB-NeRD rows <- submission/embeddings_ebnerd.parquet

Refuses to write unless each dataset's embedded ids equal the store's article
ids exactly - a store rebuilt from a different bundle fails here, loudly, rather
than leaving articles with no vector. Idempotent: re-running reads its own MIND
rows back.

Usage:
    python scripts/assemble_embeddings.py
"""

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from newsrec.retrieval.semantic import load_article_embeddings  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
ARTICLES = PROCESSED / "articles.parquet"
OUTPUT = PROCESSED / "embeddings.parquet"
SOURCES = {
    "mind": OUTPUT,
    "ebnerd": PROCESSED / "submission" / "embeddings_ebnerd.parquet",
}


def main() -> int:
    store_ids = pl.read_parquet(ARTICLES, columns=["dataset", "article_id"])

    parts = []
    for dataset, source in SOURCES.items():
        rows = pl.read_parquet(source).filter(pl.col("dataset") == dataset)
        wanted = set(store_ids.filter(pl.col("dataset") == dataset)["article_id"])
        have = set(rows["article_id"])
        dupes = rows.height - len(have)
        if have != wanted or dupes:
            print(
                f"error: {dataset} embeddings from {source} do not match the store: "
                f"{len(wanted - have):,} store articles have no vector, "
                f"{len(have - wanted):,} vectors have no store article, "
                f"{dupes:,} duplicate ids",
                file=sys.stderr,
            )
            return 1
        print(f"{dataset:7s}: {rows.height:,} vectors from {source.relative_to(REPO_ROOT)}")
        parts.append(rows)

    # Write beside the target and rename: the MIND rows are *read from* OUTPUT,
    # so writing over it directly and crashing midway would destroy the source.
    tmp = OUTPUT.with_suffix(".parquet.tmp")
    pl.concat(parts).write_parquet(tmp)
    tmp.replace(OUTPUT)

    # Read back through the loader every retriever uses - it asserts the row
    # count x 384 shape and unit norms, so this is the check that matters.
    for dataset in SOURCES:
        ids, matrix = load_article_embeddings(OUTPUT, dataset=dataset)
        print(f"verified {dataset}: {len(ids):,} x {matrix.shape[1]}, unit-normalised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
