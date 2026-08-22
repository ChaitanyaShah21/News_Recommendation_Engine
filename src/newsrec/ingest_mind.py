"""Ingest MIND-small (train + dev) raw files into the unified schema.

MIND ships as headerless TSV (Tab-Separated Values) files - see GLOSSARY.md.
This module's job is narrow: read those files and return DataFrames shaped
exactly like the unified `articles` / `impressions` / `history` tables
decided in ARCHITECTURE.md (decision D3). No BM25, no evaluation, no
temporal split here - just "raw file in, unified shape out".
"""

from pathlib import Path

import polars as pl

# MIND's news.tsv has no header row, so we supply the column names ourselves.
# This list's order must match the file's actual column order exactly.
NEWS_COLS = [
    "article_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


def load_articles(news_tsv_path: Path) -> pl.DataFrame:
    """Read one MIND news.tsv file, return it in the unified `articles` schema."""
    news = pl.read_csv(
        news_tsv_path,
        separator="\t",
        has_header=False,
        quote_char=None,
        new_columns=NEWS_COLS,
        schema_overrides={"title_entities": pl.Utf8, "abstract_entities": pl.Utf8},
    )

    return news.select(
        pl.lit("mind").alias("dataset"),
        (pl.lit("mind:") + pl.col("article_id")).alias("article_id"),
        pl.col("title"),
        pl.col("abstract"),
        pl.lit(None, dtype=pl.Utf8).alias("body"),
        pl.col("category"),
        pl.col("subcategory"),
        pl.lit(None, dtype=pl.Datetime).alias("published_time"),
        pl.lit(None, dtype=pl.Float64).alias("sentiment_score"),
        pl.lit(None, dtype=pl.Utf8).alias("sentiment_label"),
        pl.lit(None, dtype=pl.Int64).alias("total_pageviews"),
        pl.lit(None, dtype=pl.Boolean).alias("premium"),
        pl.lit(None, dtype=pl.Utf8).alias("article_type"),
        pl.concat_list(
            [pl.col("title_entities"), pl.col("abstract_entities")]
        ).alias("entities_raw"),
    )
