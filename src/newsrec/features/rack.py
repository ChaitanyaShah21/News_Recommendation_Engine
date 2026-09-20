"""Within-impression ("rack") normalisation of the candidate-varying features.

WHY THIS EXISTS
D36 recorded a measured property of the gradient-boosted model, found with a
planted signal: a tree splits on an *absolute* feature value, so a signal that
only means something relative to its own impression is learned badly. Shifting
every candidate in a rack by the same random amount dropped top-1 accuracy from
1.00 to 0.32 (random = 0.125). Any feature whose level drifts between
impressions, but whose within-rack ORDER carries the click signal, is therefore
under-used by the shipped model.

That drift is not hypothetical. Share of each feature's total variance that sits
BETWEEN racks rather than within them, measured on the real validation tables
(`diag_rack.py`, 2026-09-20):

    feature              MIND    EB-NeRD
    bm25                 0.738     0.522      <- by far the worst
    exposure_share_1h    0.356     0.425
    exposure_share_24h   0.208     0.235
    cat_share_*          0.159-0.170  0.173-0.182
    cos_*                0.088-0.130  0.118-0.176
    freshness_hours      0.039     0.116

A threshold like "bm25 > 4.2" therefore means something different for almost
three quarters of MIND's racks, because the absolute BM25 level is a property of
the user's query (its length and its terms' rarity), not of the candidate.

WHAT IS ADDED, AND WHY BOTH
For each base feature f, two extra columns:

    f_pct  the candidate's percentile rank within its own rack, in [0, 1]
    f_z    (f - rack mean) / rack standard deviation

They answer different questions. `f_pct` is scale-free and outlier-proof but
throws away magnitude: "third of eight" regardless of whether the leader is
twice as fresh or a hundred times. `f_z` keeps the margin, which is what tells a
runaway favourite from a four-way tie, but one extreme candidate compresses
everything else. Neither dominates, so both are offered and the model chooses.

The base features are KEPT, not replaced. Absolute level is not noise - an
article shown to 4% of the last hour's impressions is genuinely popular whatever
its rack-mates did - so replacing would trade one blind spot for another. Adding
is strictly more expressive; it costs feature-table width and nothing else.

WHY THIS IS AVAILABLE AT SERVING TIME
The rack is the supplied candidate list. Both leaderboards hand it over in full
(`article_ids_inview`, MIND's `impressions`), and it is the input to the request
being served, not a fact about the future. No other impression is read, so
nothing here can see a click, a later impression, or any row outside the one
being scored. That is asserted, not argued: `test_no_leakage.py` deletes every
impression after a cutoff and requires these columns to be unchanged, and
shuffles the labels and requires the same.

WHAT IS DELIBERATELY NOT NORMALISED
EB-NeRD's `hours_since_last_click`, `session_position`,
`minutes_since_session_start` and `device_type` describe the impression, not the
candidate, so they are constant across every rack (measured: 100.0%). Their
percentile would be 0.5 everywhere and their z-score 0 everywhere - two dead
columns that still cost split candidates at every node. `RACK_BASE` is therefore
an explicit list, in the same spirit as D34's allowlist.
"""

from __future__ import annotations

import polars as pl

#: The candidate-varying features. Deliberately spelled out rather than derived
#: as "FEATURES minus the impression-level ones": a future impression-level
#: feature added to the allowlist must be considered here, not silently swept in.
RACK_BASE = [
    "cos_short", "cos_medium", "cos_inf",
    "cat_share_short", "cat_share_medium", "cat_share_inf",
    "bm25",
    "freshness_hours",
    "exposure_share_1h", "exposure_share_24h",
]

KINDS = ("pct", "z")

#: A rack whose candidates all carry the same value, or which has a single
#: non-null candidate, has no within-rack order to report. Both get the neutral
#: value of their statistic rather than a null: "this rack says nothing" is a
#: fact the model can use, whereas a null would be indistinguishable from the
#: cold-start users whose cos_* is genuinely unknown.
NEUTRAL_PCT = 0.5
NEUTRAL_Z = 0.0


def rack_feature_names(base: list[str] = RACK_BASE, kinds=KINDS) -> list[str]:
    """`["bm25", ...] -> ["bm25_pct", "bm25_z", ...]`, in a fixed order."""
    return [f"{f}_{k}" for f in base for k in kinds]


def _pct(f: str) -> pl.Expr:
    """Percentile rank of `f` within its impression, in [0, 1].

    `rank("average")` ranks the non-null values 1..n and leaves nulls null, and
    `count()` excludes nulls, so (rank - 1) / (n - 1) maps the rack's smallest
    non-null value to 0 and its largest to 1. Ties share their average rank, so
    an entirely constant rack lands every candidate on exactly 0.5 without a
    special case - ((n+1)/2 - 1) / (n - 1) = 1/2 for every n.

    The special case that IS needed is n = 1, where the denominator is zero.
    """
    rank = pl.col(f).rank("average").over("impression_id")
    n = pl.col(f).count().over("impression_id")
    return (
        pl.when(pl.col(f).is_null()).then(None)
        .when(n <= 1).then(pl.lit(NEUTRAL_PCT))
        .otherwise((rank - 1.0) / (n - 1.0))
        .cast(pl.Float32)
        .alias(f"{f}_pct")
    )


def _z(f: str) -> pl.Expr:
    """(value - rack mean) / rack standard deviation.

    Polars' `std` is the sample standard deviation, so it is null for a rack
    with one non-null value and exactly 0.0 for a constant rack. Both mean the
    same thing here - there is no spread to measure - and both become 0.0, the
    value an average candidate would score anyway.
    """
    mu = pl.col(f).mean().over("impression_id")
    sd = pl.col(f).std().over("impression_id")
    return (
        pl.when(pl.col(f).is_null()).then(None)
        .when(sd.is_null() | (sd == 0.0)).then(pl.lit(NEUTRAL_Z))
        .otherwise((pl.col(f) - mu) / sd)
        .cast(pl.Float32)
        .alias(f"{f}_z")
    )


_EXPR = {"pct": _pct, "z": _z}


def add_rack_features(
    rows: pl.DataFrame, base: list[str] = RACK_BASE, kinds=KINDS
) -> pl.DataFrame:
    """Append the rack-normalised columns, preserving row order.

    `rows` must carry `impression_id` and every name in `base`. Row order is
    untouched: `.over()` is a windowed expression, not a group-by, so the result
    stays aligned with the caller's rows - which matters because LightGBM is
    given the impression structure as run lengths and would silently mis-group
    a reordered table (Landmine 4).
    """
    missing = [f for f in base if f not in rows.columns]
    if missing:
        raise ValueError(f"cannot rack-normalise columns that are absent: {missing}")
    if "impression_id" not in rows.columns:
        raise ValueError("rack normalisation needs impression_id")
    return rows.with_columns([_EXPR[k](f) for f in base for k in kinds])
