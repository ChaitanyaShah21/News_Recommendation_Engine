"""Ingest EB-NeRD (demo/small/large bundles) raw files into the unified schema.

EB-NeRD ships as Parquet - self-describing schema, no header row to supply
ourselves (see GLOSSARY.md's "Parquet" entry). Same job as ingest_mind.py:
raw file in, unified shape out, matching the three tables from D3.

A2 (D33) trains on a user-level sample of ebnerd_large rather than the demo
bundle. The sample is applied here, at ingestion, so nothing downstream ever
sees an unsampled user - see `user_sample_filter`.

Note (not yet handled, deferred to Phase 5): the EB-NeRD *test* split's
behaviors.parquet has no article_ids_clicked column at all (it's the label
we're predicting) - load_behaviors below assumes a labeled split
(train/validation) and will raise a column-not-found error if pointed at
the test split unmodified.
"""

from pathlib import Path

import polars as pl


def user_sample_filter(user_sample_pct: int | None) -> pl.Expr:
    """Return the row filter that keeps D33's user sample: `user_id % 100 < pct`.

    Applied to the *raw numeric* user_id, before ingestion adds the "ebnerd:"
    prefix - modulo means nothing on a string. Keeps or drops whole users,
    never individual impressions, so a kept user's history row and impressions
    stay in sync. Deterministic (no random generator) and nested: the users
    kept at pct=5 are a subset of those kept at any larger pct. Residue
    uniformity on ebnerd_large was verified before choosing this (D33).

    `None` means no sampling. A null user_id evaluates to null and is dropped
    by the filter; ebnerd_large has none (checked 2026-09-15).
    """
    if user_sample_pct is None:
        return pl.lit(True)
    # bool is a subclass of int in Python, so `True` would otherwise pass as 1%.
    # A YAML value of `5.0` or `"5"` is a config mistake worth failing on too.
    if (
        isinstance(user_sample_pct, bool)
        or not isinstance(user_sample_pct, int)
        or not 1 <= user_sample_pct <= 100
    ):
        raise ValueError(
            f"user_sample_pct must be an int in 1..100 or None, got {user_sample_pct!r}"
        )
    return pl.col("user_id") % 100 < user_sample_pct


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


def load_behaviors(
    behaviors_parquet_path: Path, user_sample_pct: int | None = None
) -> pl.DataFrame:
    """Read one EB-NeRD behaviors.parquet file (train or validation split),
    return it in the unified `impressions` schema, keeping only D33's user
    sample when `user_sample_pct` is given.

    Built on pl.scan_parquet (lazy), not pl.read_parquet (eager): Polars
    pushes both the column selection and the sample filter down into the
    file read, so on ebnerd_large the 95% of unsampled rows are skipped while
    reading rather than loaded and then thrown away. Still .collect() at the
    end so the *return type* matches ingest_mind.py's functions.
    """
    behaviors = pl.scan_parquet(behaviors_parquet_path).filter(
        user_sample_filter(user_sample_pct)
    )

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


def load_history(
    history_parquet_path: Path, user_sample_pct: int | None = None
) -> pl.DataFrame:
    """Read one EB-NeRD history.parquet file, return it in the unified `history`
    schema, keeping only D33's user sample when `user_sample_pct` is given.
    Must be called with the same `user_sample_pct` as load_behaviors, or the
    two tables describe different users.

    Unlike MIND, EB-NeRD's history file is already one row per user - no
    collapsing/deduplication needed, verified against real demo data (1,590
    history rows for 1,590 unique behaviors users, exactly 1:1) and again on
    the ebnerd_large 5% sample (39,260 / 39,420 per split, D33)."""
    history = pl.scan_parquet(history_parquet_path).filter(
        user_sample_filter(user_sample_pct)
    )

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
