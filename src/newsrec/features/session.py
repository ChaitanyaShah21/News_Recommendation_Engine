"""session features, EB-NeRD only and label-free (D34a).

MIND has no session or dwell fields at all, so these exist for one dataset;
D34's consequence is that the two datasets' re-rankers see different features.

A session is keyed by (user_id, session_id), never session_id alone: measured
2026-09-18, 1,032 EB-NeRD session ids are shared by more than one user, and
grouping on the id alone merges strangers into one "session" - the longest such
merge spans 14 days. Keyed properly, the longest real session is 29 minutes.
"""

from __future__ import annotations

import polars as pl

SESSION_KEY = ["user_id", "session_id"]


def session_features(impressions: pl.DataFrame) -> pl.DataFrame:
    """Per impression: position in its session, minutes since the session began,
    and device type.

    `impressions` must be ONE dataset across ALL its splits: 37 real sessions
    straddle the train/val boundary (browsing across 07:00), and a val
    impression's earlier session-mates in the train split are legitimately past.

    session_position = number of STRICTLY earlier impressions in the session.
    A same-second tie is not "earlier" (D34a), so tied impressions share a
    position - computed as rank(method="min") - 1, never "ordinal", which would
    break ties by file order and let row order stand in for time.

    minutes_since_session_start uses the session's earliest impression, which
    is at or before T for every member because T is itself in the session - so
    it only ever reports a past fact. No click label is read anywhere.
    """
    if impressions.select(pl.col("dataset").n_unique()).item() != 1:
        raise ValueError("pass exactly one dataset")
    if impressions["session_id"].null_count() == impressions.height:
        raise ValueError("no session_id at all - MIND has no sessions (D34)")
    if impressions["session_id"].null_count():
        raise ValueError("some impressions lack a session_id")

    return impressions.select(
        "impression_id",
        (pl.col("timestamp").rank(method="min").over(SESSION_KEY) - 1)
        .cast(pl.Int32)
        .alias("session_position"),
        ((pl.col("timestamp") - pl.col("timestamp").min().over(SESSION_KEY))
         .dt.total_seconds() / 60.0)
        .alias("minutes_since_session_start"),
        pl.col("device_type"),
    )
