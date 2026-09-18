"""/ D34a: EB-NeRD session features. R10 adversarial suite."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from newsrec.features import session as sf

T0 = datetime(2023, 5, 20, 6, 50)
M = timedelta(minutes=1)
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
requires_store = pytest.mark.skipif(
    not (PROCESSED / "impressions.parquet").exists(), reason="feature store not built"
)


def _imps(rows, split="val"):
    """rows: (impression_id, user_id, session_id, timestamp)"""
    return pl.DataFrame({
        "dataset": ["ebnerd"] * len(rows),
        "split": [r[4] if len(r) > 4 else split for r in rows],
        "impression_id": [r[0] for r in rows],
        "user_id": [r[1] for r in rows],
        "session_id": [r[2] for r in rows],
        "timestamp": [r[3] for r in rows],
        "device_type": [2] * len(rows),
    })


def _by_id(out, col):
    return dict(zip(out["impression_id"], out[col]))


def test_position_counts_strictly_earlier_impressions_in_the_session():
    out = sf.session_features(_imps([
        ("c", "u", "s", T0 + 5 * M), ("a", "u", "s", T0), ("b", "u", "s", T0 + 2 * M),
    ]))
    assert _by_id(out, "session_position") == {"a": 0, "b": 1, "c": 2}
    assert _by_id(out, "minutes_since_session_start") == {"a": 0.0, "b": 2.0, "c": 5.0}


def test_same_second_impressions_share_a_position_whatever_the_file_order():
    rows = [("a", "u", "s", T0), ("b", "u", "s", T0 + M), ("c", "u", "s", T0 + M)]
    fwd = _by_id(sf.session_features(_imps(rows)), "session_position")
    rev = _by_id(sf.session_features(_imps(rows[::-1])), "session_position")
    assert fwd == rev == {"a": 0, "b": 1, "c": 1}


def test_two_users_sharing_a_session_id_are_not_merged():
    # 1,032 real EB-NeRD session ids are shared across users.
    out = sf.session_features(_imps([
        ("u1a", "u1", "s", T0), ("u1b", "u1", "s", T0 + 3 * M),
        ("u2a", "u2", "s", T0 + 10 * 24 * 60 * M),  # a stranger, ten days later
    ]))
    assert _by_id(out, "session_position")["u2a"] == 0
    assert _by_id(out, "minutes_since_session_start")["u2a"] == 0.0


def test_a_session_crossing_the_split_boundary_counts_its_earlier_half():
    out = sf.session_features(_imps([
        ("t", "u", "s", T0, "train"), ("v", "u", "s", T0 + 15 * M, "val"),
    ]))
    assert _by_id(out, "session_position")["v"] == 1
    assert _by_id(out, "minutes_since_session_start")["v"] == 15.0


def test_features_do_not_read_click_labels():
    base = _imps([("a", "u", "s", T0), ("b", "u", "s", T0 + M)])
    a = sf.session_features(base.with_columns(pl.Series("clicked_article_ids", [["x"], []])))
    b = sf.session_features(base.with_columns(pl.Series("clicked_article_ids", [[], ["y"]])))
    assert a.equals(b)


def test_mind_without_sessions_is_refused():
    imps = _imps([("a", "u", "s", T0)]).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("session_id"), pl.lit("mind").alias("dataset"))
    with pytest.raises(ValueError, match="MIND"):
        sf.session_features(imps)


def test_mixing_datasets_is_refused():
    imps = pl.concat([_imps([("a", "u", "s", T0)]),
                      _imps([("b", "u", "s", T0)]).with_columns(pl.lit("mind").alias("dataset"))])
    with pytest.raises(ValueError, match="one dataset"):
        sf.session_features(imps)


@requires_store
def test_real_ebnerd_sessions_are_short_and_complete():
    imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == "ebnerd")
    out = sf.session_features(imps)
    assert out.height == imps.height and out["impression_id"].n_unique() == imps.height
    assert out["minutes_since_session_start"].max() <= 30  # measured max 29
    assert out["session_position"].min() == 0
