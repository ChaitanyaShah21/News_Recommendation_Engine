"""Ingest EB-NeRD (demo/small/large bundles) raw files into the unified schema.

EB-NeRD ships as Parquet - self-describing schema, no header row to supply
ourselves (see GLOSSARY.md's "Parquet" entry). Same job as ingest_mind.py:
raw file in, unified shape out, matching the three tables from D3.

Note (not yet handled, deferred to Phase 5): the EB-NeRD *test* split's
behaviors.parquet has no article_ids_clicked column at all (it's the label
we're predicting) - load_behaviors below assumes a labeled split
(train/validation) and will raise a column-not-found error if pointed at
the test split unmodified.
"""

from pathlib import Path

import polars as pl


def load_articles(articles_parquet_path: Path) -> pl.DataFrame:
    """Read one EB-NeRD articles.parquet file, return it in the unified `articles` schema."""
    articles = pl.read_parquet(articles_parquet_path)

    return articles.select(
        pl.lit("ebnerd").alias("dataset"),
        (pl.lit("ebnerd:") + pl.col("article_id").cast(pl.Utf8)).alias("article_id"),
        pl.col("title"),
        pl.col("subtitle").alias("abstract"),
        pl.col("body"),
        pl.col("category_str").alias("category"),
        # Numeric codes, no name lookup provided in this file - stored as
        # stringified codes, not decoded. See GLOSSARY.md if this is read later.
        pl.col("subcategory")
        .list.eval(pl.element().cast(pl.Utf8))
        .alias("subcategory"),
        pl.col("published_time"),
        # EB-NeRD stores this as Float32; MIND's null placeholder is Float64
        # (Polars' default for pl.lit(None, dtype=...) with no width given).
        # Cast up so the two `articles` tables can concat - found by trying
        # the actual concat, not by reading the schema and assuming it'd match.
        pl.col("sentiment_score").cast(pl.Float64),
        pl.col("sentiment_label"),
        pl.col("total_pageviews").cast(pl.Int64),
        pl.col("premium"),
        pl.col("article_type"),
        pl.concat_list(
            [pl.col("ner_clusters"), pl.col("entity_groups")]
        ).alias("entities_raw"),
    )


def load_behaviors(behaviors_parquet_path: Path) -> pl.DataFrame:
    """Read one EB-NeRD behaviors.parquet file (train or validation split),
    return it in the unified `impressions` schema.

    Built on pl.scan_parquet (lazy), not pl.read_parquet (eager): this same
    file layout is used for EB-NeRD-large (12M+ rows) in a later phase, and
    scan_parquet lets Polars push column selection down before materializing
    anything. Still .collect() at the end so the *return type* matches
    ingest_mind.py's functions (a plain DataFrame, not a LazyFrame) - true
    batched/chunked processing for the large bundle is a Phase 5 concern,
    this alone doesn't solve that, it just avoids unnecessary work now.
    """
    behaviors = pl.scan_parquet(behaviors_parquet_path)

    def prefixed(list_col: str) -> pl.Expr:
        """EB-NeRD's inview/clicked lists are already lists - just int article
        IDs, not strings needing suffix-stripping like MIND. Cast each element
        to string, then prefix, same "mind:"-style tagging as ingest_mind.py."""
        return pl.col(list_col).list.eval(
            pl.lit("ebnerd:") + pl.element().cast(pl.Utf8)
        )

    return (
        behaviors.select(
            pl.lit("ebnerd").alias("dataset"),
            (pl.lit("ebnerd:") + pl.col("impression_id").cast(pl.Utf8)).alias(
                "impression_id"
            ),
            (pl.lit("ebnerd:") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
            pl.col("impression_time").alias("timestamp"),
            prefixed("article_ids_inview").alias("candidate_article_ids"),
            prefixed("article_ids_clicked").alias("clicked_article_ids"),
            # Same Float32-vs-Float64 mismatch as sentiment_score above.
            pl.col("read_time").cast(pl.Float64),
            pl.col("scroll_percentage").cast(pl.Float64),
            pl.col("device_type").cast(pl.Int64),
            pl.col("session_id").cast(pl.Utf8),
        )
        .collect()
    )


def load_history(history_parquet_path: Path) -> pl.DataFrame:
    """Read one EB-NeRD history.parquet file, return it in the unified `history`
    schema. Unlike MIND, EB-NeRD's history file is already one row per user -
    no collapsing/deduplication needed, verified against real demo data
    (1,590 history rows for 1,590 unique behaviors users, exactly 1:1)."""
    history = pl.scan_parquet(history_parquet_path)

    return (
        history.select(
            pl.lit("ebnerd").alias("dataset"),
            (pl.lit("ebnerd:") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
            pl.col("article_id_fixed")
            .list.eval(pl.lit("ebnerd:") + pl.element().cast(pl.Utf8))
            .alias("history_article_ids"),
            pl.col("impression_time_fixed").alias("history_timestamps"),
        )
        .collect()
    )
