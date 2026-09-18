"""Q9's quarantine: features that are NOT available at serving time.

Never imported by the production feature set (asserted in
tests/test_feature_table.py). Each is used only as an ablation arm, to price
what a serving-unavailable feature would appear to buy.

  read_time, scroll_percentage  measured AFTER the impression is served, even
                                though EB-NeRD's testset still carries them -
                                present is not available (system design contract).
  total_pageviews               a whole-dataset aggregate, i.e. includes the
                                future of every impression.
  future_exposure_share_24h     the mirror of D34b's feature over (T, T+24h]:
                                what the site WILL push, which it cannot know.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from newsrec.features import article as af

UNAVAILABLE = ["read_time", "scroll_percentage", "total_pageviews", "future_exposure_share_24h"]


def unavailable_features(
    features: pl.DataFrame, target: pl.DataFrame, all_impressions: pl.DataFrame,
    articles: pl.DataFrame,
) -> pl.DataFrame:
    """Add the quarantined columns to a feature table. Ablation arm only."""
    out = features.join(
        target.select("impression_id", "read_time", "scroll_percentage"),
        on="impression_id", how="left",
    ).join(articles.select("article_id", "total_pageviews"), on="article_id", how="left")

    # Future exposure: shift every query to T + 24h and look back 24h, which
    # counts impressions in [T, T+24h) - the complement of the honest window.
    rows = features.select("impression_id", "article_id").join(
        target.select("impression_id", "timestamp"), on="impression_id", how="left")
    shifted = rows.with_columns(pl.col("timestamp") + pl.duration(hours=24))
    fut = af.exposure_shares(shifted, all_impressions, windows_hours=(24,))
    out = out.with_columns(fut["exposure_share_24h"].alias("future_exposure_share_24h"))
    return out
