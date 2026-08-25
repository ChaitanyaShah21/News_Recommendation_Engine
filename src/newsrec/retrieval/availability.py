"""D19 - which articles were already in circulation at a given moment.

Shared by BM25 (Q2) and semantic (Q3) retrieval. Extracted from
`scripts/run_bm25_recall.py` when Q3 needed the same logic: two copies of this
would drift, and a fix applied to one and not the other would silently corrupt
exactly the cross-method comparison Q3.5 asks for.

Leak-safety, which is what Q9 grades and the reason this lives in one place:
the masks come from `first_seen < bucket_start` - STRICTLY before. Two
conditions must both hold, and arguing only the first is the common mistake:

  1. Temporal - the predicate admits only facts true before the bucket began.
     `<=` would let an article first appearing in *this very impression* count
     as available, meaning our knowledge that it exists comes from the
     impression we are predicting. Circular.
  2. Label-free - `first_seen` is computed from `candidate_article_ids`, what
     was SHOWN, and never from `clicked_article_ids`. Availability derived from
     clicks would still satisfy condition 1 and still be leakage, because the
     candidate pool would be pre-selected by the answers.

`tests/test_no_leakage.py` asserts both.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Sentinel for "never appeared in any candidate list". Such an article is never
# available - and can never be clicked either, so excluding it removes only noise.
NEVER_SEEN = np.datetime64("2999-01-01")


def first_seen_times(impressions: pl.DataFrame) -> pl.DataFrame:
    """When each article first appeared in *any* impression's candidate list.

    Used instead of `published_time`, which is null for 100% of MIND.
    """
    return (
        impressions.select("candidate_article_ids", "timestamp")
        # empty_as_null is explicit because its default flips in Polars 2.0.
        # Either value is correct here - the drop_nulls below removes the null
        # row an empty candidate list would produce - but pinning it means the
        # upgrade cannot silently change what this returns.
        .explode("candidate_article_ids", empty_as_null=True)
        .drop_nulls("candidate_article_ids")
        .group_by("candidate_article_ids")
        .agg(pl.col("timestamp").min().alias("first_seen"))
        .rename({"candidate_article_ids": "article_id"})
    )


def build_availability(
    all_impressions: pl.DataFrame,
    split_impressions: pl.DataFrame,
    article_ids: list[str],
    bucket: str = "1h",
) -> tuple[pl.DataFrame, dict, list[np.ndarray]]:
    """Per time-bucket masks of which articles were already in circulation.

    Args:
        all_impressions: the whole impression log for this dataset, every split.
            Wider than the split being evaluated on purpose - an article's first
            appearance may predate the split. Safe because it is only ever read
            through the `first_seen < T` predicate described in the module
            docstring.
        split_impressions: the impressions actually being evaluated.
        article_ids: row order the masks must align to - the embedding store's
            order for Q3, the index's order for Q2.

    Returns:
        bucketed: `split_impressions` with a `bucket_start` column added.
        bucket_id: bucket_start -> position in `masks`.
        masks: one BOOLEAN array per bucket over article rows, True where the
            article had already appeared before that bucket began.

    Boolean, not the float32 0/1 this used to return. Callers that mask by
    multiplication must cast; callers that mask with -inf (which dense cosine
    scoring must, since 0 is mid-range there rather than the floor) use it
    directly.
    """
    first_seen = first_seen_times(all_impressions)

    seen_at = dict(
        zip(
            first_seen.get_column("article_id").to_list(),
            first_seen.get_column("first_seen").to_list(),
        )
    )
    article_first_seen = np.array(
        [seen_at.get(a) or NEVER_SEEN for a in article_ids],
        dtype="datetime64[us]",
    )

    bucketed = split_impressions.with_columns(
        pl.col("timestamp").dt.truncate(bucket).alias("bucket_start")
    )
    starts = sorted(bucketed.get_column("bucket_start").unique().to_list())
    bucket_id = {start: i for i, start in enumerate(starts)}
    # STRICTLY less than. Never <=. See the module docstring.
    masks = [article_first_seen < np.datetime64(start, "us") for start in starts]
    return bucketed, bucket_id, masks
