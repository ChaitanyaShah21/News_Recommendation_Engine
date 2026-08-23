"""Temporal train/val/test split (Q1.3).

Neither MIND-small nor EB-NeRD-demo provides three labeled partitions -
each only gives train + dev/validation (see PROGRESS.md, D7/D8 in
ARCHITECTURE.md). Train is used untouched; dev/validation gets carved into
val + test by a row-count-based timestamp cutoff: the earliest 70% of that
partition's rows (by time, not arbitrary order) become val, the latest 30%
become test - the same rule applied uniformly regardless of whether the
partition spans 1 day (MIND) or 7 (EB-NeRD).
"""

import polars as pl


def add_impressions_split(
    train: pl.DataFrame,
    dev_or_validation: pl.DataFrame,
    val_fraction: float = 0.7,
) -> pl.DataFrame:
    """Tag `train` as split="train" unchanged; split `dev_or_validation` by
    time at the row-count cutoff `val_fraction`, tagging the earlier share
    "val" and the later share "test". Returns one combined `impressions`
    DataFrame - same total row count as train + dev_or_validation combined,
    nothing dropped or duplicated.
    """
    n = dev_or_validation.height
    cutoff_index = int(n * val_fraction)

    # Sort by timestamp FIRST - this is what makes "row cutoff_index" a real
    # temporal boundary rather than an arbitrary slice of file order.
    sorted_dev = dev_or_validation.sort("timestamp")
    cutoff_timestamp = sorted_dev["timestamp"][cutoff_index]

    tagged_train = train.with_columns(pl.lit("train").alias("split"))
    tagged_val_test = dev_or_validation.with_columns(
        pl.when(pl.col("timestamp") < cutoff_timestamp)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )

    return pl.concat([tagged_train, tagged_val_test])


def add_history_split(
    train_history: pl.DataFrame,
    dev_or_validation_history: pl.DataFrame,
) -> pl.DataFrame:
    """Tag train_history as split="train". The dev/validation history table
    is duplicated once per new sub-partition (val, test) rather than split -
    there's only one history snapshot available at that file's granularity
    (verified: EB-NeRD's train/history.parquet and validation/history.parquet
    genuinely differ per user, so this snapshot can't be reconstructed any
    finer than the file boundary gives us). Reusing it for both val and test
    doesn't leak anything - it predates the entire dev/validation window,
    val and test both included - it's just conservatively stale for test's
    later rows rather than perfectly fresh.
    """
    tagged_train = train_history.with_columns(pl.lit("train").alias("split"))
    tagged_val = dev_or_validation_history.with_columns(pl.lit("val").alias("split"))
    tagged_test = dev_or_validation_history.with_columns(pl.lit("test").alias("split"))
    return pl.concat([tagged_train, tagged_val, tagged_test])
