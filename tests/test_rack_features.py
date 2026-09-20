"""D42: within-impression (rack) normalisation. R10 adversarial suite.

The failure modes actively hunted here, rather than assumed absent:
  - a null silently becoming a number (a cold user's cos_* would become the
    rack's average candidate, which is a fabricated fact about a user we know
    nothing about);
  - a degenerate rack - one candidate, or every candidate equal - producing a
    NaN or an infinity that LightGBM reads as "missing" and learns a direction
    for, making a rack with no information look like a rack with a null;
  - row order changing, which would mis-group every impression for LightGBM
    (Landmine 4) while the group sizes still summed correctly;
  - one rack's statistics bleeding into the next, which is the rack version of
    the key-range spill af.ExposureIndex has its own test for;
  - the normalisation accidentally depending on the label or on a later
    impression, which would make it a leak (also covered in test_no_leakage.py
    against real data).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from newsrec.features import rack
from newsrec.features.assemble import FEATURES


def _df(imp, x, col="bm25"):
    return pl.DataFrame({"impression_id": list(imp), col: list(x)},
                        schema={"impression_id": pl.String, col: pl.Float64})


def _add(df, col="bm25"):
    return rack.add_rack_features(df, base=[col])


# ------------------------------------------------------------ the happy path

def test_percentile_spans_zero_to_one_within_each_rack():
    out = _add(_df("aaa", [1.0, 2.0, 3.0]))
    assert out["bm25_pct"].to_list() == [0.0, 0.5, 1.0]


def test_z_score_is_centred_and_scaled_within_the_rack():
    out = _add(_df("aaaa", [1.0, 2.0, 3.0, 4.0]))
    z = np.array(out["bm25_z"].to_list())
    assert z.mean() == pytest.approx(0.0, abs=1e-6)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-6)


def test_two_racks_with_different_levels_get_the_same_normalised_values():
    """The whole point: a rack at 1-3 and a rack at 1001-1003 are the same shape.

    This is the planted signal from D36 that the absolute features fail."""
    out = _add(_df("aaabbb", [1.0, 2.0, 3.0, 1001.0, 1002.0, 1003.0]))
    assert out["bm25_pct"].to_list()[:3] == out["bm25_pct"].to_list()[3:]
    assert out["bm25_z"].to_list()[:3] == out["bm25_z"].to_list()[3:]


# ---------------------------------------------------------- degenerate racks

def test_a_constant_rack_is_neutral_everywhere_and_never_nan():
    """Every candidate equal: no within-rack order exists. 0.5 / 0.0, not 0/0."""
    out = _add(_df("aaa", [7.0, 7.0, 7.0]))
    assert out["bm25_pct"].to_list() == [rack.NEUTRAL_PCT] * 3
    assert out["bm25_z"].to_list() == [rack.NEUTRAL_Z] * 3
    assert not any(math.isnan(v) for v in out["bm25_pct"].to_list() + out["bm25_z"].to_list())


def test_a_single_candidate_rack_is_neutral_and_never_divides_by_zero():
    """n - 1 == 0 for the percentile, and the sample std is null for n == 1."""
    out = _add(_df("a", [7.0]))
    assert out["bm25_pct"].to_list() == [rack.NEUTRAL_PCT]
    assert out["bm25_z"].to_list() == [rack.NEUTRAL_Z]


def test_a_rack_with_one_non_null_candidate_is_neutral_not_extreme():
    """The non-null value is simultaneously the rack min and max. Ranking it 0
    or 1 would invent a comparison against candidates we have no value for."""
    out = _add(_df("aaa", [7.0, None, None]))
    assert out["bm25_pct"].to_list() == [rack.NEUTRAL_PCT, None, None]
    assert out["bm25_z"].to_list() == [rack.NEUTRAL_Z, None, None]


def test_zeros_are_a_real_value_not_a_missing_one():
    """An all-zero rack is constant, not absent - BM25 scores 2.4% of MIND's
    candidates exactly 0, and 0.10% of impressions score every candidate 0."""
    out = _add(_df("aaa", [0.0, 0.0, 0.0]))
    assert out["bm25_pct"].to_list() == [rack.NEUTRAL_PCT] * 3
    assert out["bm25_z"].null_count() == 0


# ------------------------------------------------------------------- nulls

def test_nulls_stay_null_and_are_excluded_from_the_rack_statistics():
    """A cold user's cos_* is unknown. Filling it with the rack average would
    hand the model a fact the log never recorded."""
    out = _add(_df("aaaa", [1.0, None, 3.0, 5.0]))
    assert out["bm25_pct"].to_list() == [0.0, None, 0.5, 1.0]
    assert out["bm25_z"][1] is None
    z = [v for v in out["bm25_z"].to_list() if v is not None]
    assert np.mean(z) == pytest.approx(0.0, abs=1e-6)


def test_an_all_null_rack_stays_all_null():
    out = _add(_df("aaa", [None, None, None]))
    assert out["bm25_pct"].null_count() == 3
    assert out["bm25_z"].null_count() == 3


# --------------------------------------------------------- isolation & order

def test_one_rack_never_reads_another_racks_values():
    """Interleave the racks so a group-by that ignored impression_id, or a
    window that leaked across the boundary, would show up as a changed value."""
    df = _df("ababab", [1.0, 100.0, 2.0, 200.0, 3.0, 300.0])
    out = _add(df)
    a = out.filter(pl.col("impression_id") == "a")["bm25_pct"].to_list()
    b = out.filter(pl.col("impression_id") == "b")["bm25_pct"].to_list()
    assert a == [0.0, 0.5, 1.0] and b == [0.0, 0.5, 1.0]


def test_row_order_is_preserved_exactly():
    """LightGBM is given run lengths, never ids (Landmine 4): a silently
    reordered table trains happily and ranks garbage."""
    df = _df("abaab", [1.0, 9.0, 2.0, 3.0, 8.0])
    out = _add(df)
    assert out["impression_id"].to_list() == df["impression_id"].to_list()
    assert out["bm25"].to_list() == df["bm25"].to_list()


def test_normalisation_is_invariant_to_the_order_rows_arrive_in():
    """Same rows, shuffled: each row's own normalised value must not change."""
    df = _df("aaabb", [1.0, 2.0, 3.0, 10.0, 20.0]).with_row_index("k")
    a = _add(df).sort("k")
    b = _add(df.sample(fraction=1.0, shuffle=True, seed=7)).sort("k")
    assert a["bm25_pct"].to_list() == b["bm25_pct"].to_list()
    assert a["bm25_z"].to_list() == b["bm25_z"].to_list()


# -------------------------------------------------- label and future blindness

def test_the_label_column_changes_nothing():
    df = _df("aaa", [1.0, 2.0, 3.0])
    a = _add(df.with_columns(pl.Series("clicked", [0, 0, 1])))
    b = _add(df.with_columns(pl.Series("clicked", [1, 0, 0])))
    assert a["bm25_pct"].to_list() == b["bm25_pct"].to_list()
    assert a["bm25_z"].to_list() == b["bm25_z"].to_list()


def test_deleting_later_impressions_leaves_earlier_racks_identical():
    """The future-deletion property, in miniature. A rack is one impression, so
    no later impression can reach it - asserted rather than reasoned about."""
    df = _df("aaabbb", [1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    full = _add(df).filter(pl.col("impression_id") == "a")
    trimmed = _add(df.filter(pl.col("impression_id") == "a"))
    assert full.equals(trimmed)


# --------------------------------------------------------------- the contract

def test_missing_base_column_raises_rather_than_skipping_it():
    with pytest.raises(ValueError, match="absent"):
        rack.add_rack_features(_df("aa", [1.0, 2.0]), base=["bm25", "nonesuch"])


def test_missing_impression_id_raises():
    with pytest.raises(ValueError, match="impression_id"):
        rack.add_rack_features(pl.DataFrame({"bm25": [1.0]}), base=["bm25"])


def test_names_are_stable_and_ordered():
    assert rack.rack_feature_names(["a", "b"]) == ["a_pct", "a_z", "b_pct", "b_z"]


def test_rack_base_excludes_every_impression_level_feature():
    """Measured: hours_since_last_click, session_position,
    minutes_since_session_start and device_type are constant across 100.0% of
    EB-NeRD racks. Normalising them would produce two dead columns each."""
    impression_level = {"hours_since_last_click", "session_position",
                        "minutes_since_session_start", "device_type"}
    assert impression_level.isdisjoint(rack.RACK_BASE)


def test_rack_base_is_a_subset_of_the_allowlist_for_both_datasets():
    """A rack feature derived from a column that is not itself allowlisted would
    smuggle that column into the model (D34)."""
    for ds in ("mind", "ebnerd"):
        assert set(rack.RACK_BASE) <= set(FEATURES[ds]), ds
