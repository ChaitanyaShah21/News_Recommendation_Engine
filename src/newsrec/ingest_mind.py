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


# MIND's behaviors.tsv has no header row either - same reasoning as NEWS_COLS above.
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]

# MIND's timestamps look like "11/11/2019 9:05:58 AM" - month/day/year, then a
# 12-hour clock with an AM/PM marker. Verified against real data that Polars'
# %I (12-hour) accepts the single-digit hours MIND actually uses (e.g. "9:05:58",
# not "09:05:58") before trusting this format string.
MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def load_behaviors(behaviors_tsv_path: Path) -> pl.DataFrame:
    """Read one MIND behaviors.tsv file, return it in the unified `impressions` schema."""
    behaviors = pl.read_csv(
        behaviors_tsv_path,
        separator="\t",
        has_header=False,
        quote_char=None,
        new_columns=BEHAVIOR_COLS,
        schema_overrides={
            "impression_id": pl.Int64,
            "history": pl.Utf8,
            "impressions": pl.Utf8,
        },
    )

    # Split "N55689-1 N35729-0" into a list of tokens first - every later step
    # works on this list, not the original string.
    tokens = pl.col("impressions").str.split(" ")

    # Every token, with the "-1"/"-0" suffix stripped and "mind:" prefixed.
    # Verified: every token in the real data matches "N<digits>-[01]" exactly,
    # so this regex can't accidentally strip something that isn't the suffix.
    candidate_ids = tokens.list.eval(
        pl.lit("mind:") + pl.element().str.replace(r"-[01]$", "")
    )

    # Keep only tokens ending "-1" (the clicked ones), THEN strip the suffix and
    # prefix - two separate list.eval passes, filter first, so each step is one
    # clear operation rather than one dense expression doing both at once.
    clicked_tokens = tokens.list.eval(
        pl.element().filter(pl.element().str.ends_with("-1"))
    )
    clicked_ids = clicked_tokens.list.eval(
        pl.lit("mind:") + pl.element().str.replace(r"-1$", "")
    )

    return behaviors.select(
        pl.lit("mind").alias("dataset"),
        (pl.lit("mind:") + pl.col("impression_id").cast(pl.Utf8)).alias("impression_id"),
        (pl.lit("mind:") + pl.col("user_id")).alias("user_id"),
        pl.col("time").str.strptime(pl.Datetime, MIND_TIME_FORMAT).alias("timestamp"),
        candidate_ids.alias("candidate_article_ids"),
        clicked_ids.alias("clicked_article_ids"),
        pl.lit(None, dtype=pl.Float64).alias("read_time"),
        pl.lit(None, dtype=pl.Float64).alias("scroll_percentage"),
        pl.lit(None, dtype=pl.Int64).alias("device_type"),
        pl.lit(None, dtype=pl.Utf8).alias("session_id"),
    )


def load_history(behaviors_tsv_path: Path) -> pl.DataFrame:
    """Read one MIND behaviors.tsv file, collapse repeated per-impression history
    strings down to one row per user, return in the unified `history` schema.

    Verified (R10) before writing this: every MIND user's history string is
    identical across all of their impression rows in the same file - 0 of
    33,617 multi-row users in MINDsmall_train had more than one distinct
    history string - so keeping just the first row per user loses nothing.
    """
    behaviors = pl.read_csv(
        behaviors_tsv_path,
        separator="\t",
        has_header=False,
        quote_char=None,
        new_columns=BEHAVIOR_COLS,
        schema_overrides={
            "impression_id": pl.Int64,
            "history": pl.Utf8,
            "impressions": pl.Utf8,
        },
    )

    one_row_per_user = behaviors.select("user_id", "history").unique(
        subset="user_id", keep="first"
    )

    # Cold-start users (~2.1% of MIND rows) have a null `history` field.
    # .str.split() on null stays null rather than becoming an empty list -
    # verified this above. fill_null([]) makes it a real empty list, so
    # every downstream caller can safely do things like .list.len() without
    # a separate null-check for cold-start users every single time.
    history_lists = pl.col("history").str.split(" ").fill_null([])

    return one_row_per_user.select(
        pl.lit("mind").alias("dataset"),
        (pl.lit("mind:") + pl.col("user_id")).alias("user_id"),
        history_lists.list.eval(
            pl.lit("mind:") + pl.element()
        ).alias("history_article_ids"),
        pl.lit(None, dtype=pl.List(pl.Datetime)).alias("history_timestamps"),
    )
