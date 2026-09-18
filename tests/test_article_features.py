"""/ D34b, D34c: freshness and label-free exposure share. R10 adversarial suite."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from pathlib import Path

from newsrec.features import article as af

T0 = datetime(2023, 5, 1)
H = timedelta(hours=1)
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
requires_store = pytest.mark.skipif(
    not (PROCESSED / "impressions.parquet").exists(), reason="feature store not built"
)


def _imps(rows):
    """rows: (impression_id, timestamp, [candidates])"""
    return pl.DataFrame({
        "impression_id": [r[0] for r in rows],
        "timestamp": [r[1] for r in rows],
        "candidate_article_ids": [r[2] for r in rows],
    })


def _share(out, imp, art, w=1):
    return out.filter((pl.col("impression_id") == imp) & (pl.col("article_id") == art))[
        f"exposure_share_{w}h"
    ].item()


# ---------------------------------------------------------------- freshness

def _articles(pub):
    return pl.DataFrame({"article_id": list(pub), "published_time": list(pub.values())},
                        schema={"article_id": pl.Utf8, "published_time": pl.Datetime("us")})


def test_freshness_is_hours_since_publication():
    imps = _imps([("i1", T0 + 5 * H, ["A"])])
    out = af.freshness_hours(af.candidate_rows(imps), _articles({"A": T0}), imps)
    assert out["freshness_hours"].to_list() == [5.0]


def test_a_republished_article_is_dated_from_when_it_was_first_shown():
    # Shown at T0+1h and T0+5h, but published_time says T0+10h (a republication).
    imps = _imps([("i1", T0 + H, ["A"]), ("i2", T0 + 5 * H, ["A"])])
    out = af.freshness_hours(af.candidate_rows(imps), _articles({"A": T0 + 10 * H}), imps)
    assert dict(zip(out["impression_id"], out["freshness_hours"])) == {"i1": 0.0, "i2": 4.0}


def test_mind_with_no_published_time_falls_back_to_first_seen():
    imps = _imps([("i1", T0, ["A"]), ("i2", T0 + 3 * H, ["A"])])
    out = af.freshness_hours(af.candidate_rows(imps), _articles({"A": None}), imps)
    assert sorted(out["freshness_hours"].to_list()) == [0.0, 3.0]


def test_rows_from_impressions_missing_from_the_first_seen_source_are_refused():
    imps = _imps([("i1", T0, ["A"])])
    other = _imps([("i9", T0, ["B"])])
    with pytest.raises(ValueError, match="first_seen"):
        af.freshness_hours(af.candidate_rows(imps), _articles({"A": None}), other)


# ---------------------------------------------------------------- exposure share

def test_the_share_counts_impressions_strictly_before_t():
    imps = _imps([
        ("i1", T0, ["A", "B"]),
        ("i2", T0 + 10 * 60 * timedelta(seconds=1), ["B"]),
        ("i3", T0 + 30 * 60 * timedelta(seconds=1), ["A", "B"]),
    ])
    out = af.exposure_shares(af.candidate_rows(imps), imps)
    # Before i3 (in the last hour): i1 and i2. A was in 1 of 2, B in 2 of 2.
    assert _share(out, "i3", "A") == 0.5 and _share(out, "i3", "B") == 1.0


def test_an_impression_never_counts_itself_or_its_same_second_neighbours():
    imps = _imps([("i1", T0, ["A"]), ("i2", T0 + H / 2, ["A"]), ("i3", T0 + H / 2, ["A"])])
    out = af.exposure_shares(af.candidate_rows(imps), imps)
    # i2 and i3 share a second: each sees only i1.
    assert _share(out, "i2", "A") == 1.0 and _share(out, "i3", "A") == 1.0
    assert _share(out, "i1", "A") is None  # empty window -> null, not 0


def test_the_window_start_is_inclusive_and_older_impressions_drop_out():
    imps = _imps([("i0", T0, ["A"]), ("i1", T0 + H, ["B"]), ("i2", T0 + 2 * H, ["B"])])
    out = af.exposure_shares(af.candidate_rows(imps), imps)
    # 1 h window before i2 is [T0+1h, T0+2h): contains i1 only. i0 has dropped out.
    assert _share(out, "i2", "B") == 1.0
    assert _share(out, "i2", "B", w=24) == 0.5


def test_an_early_window_cannot_borrow_counts_from_the_previous_article():
    # Article "A" sorts before "B"; A is shown late, B is queried early. A window
    # for B reaching back before the data would land in A's key range and count
    # A's late impressions as B's. Two independent guards prevent it: the clamp
    # on `lo`, and `span` padded by the longest window. Mutation-tested
    # 2026-09-18: removing either alone is an equivalent mutant (the other
    # still holds); removing both makes this test fail with 30 borrowed
    # impressions instead of 0 - so the test does detect the real bug.
    late = [(f"a{i}", T0 + 20 * H + i * H / 60, ["A"]) for i in range(30)]
    imps = _imps([("b0", T0, ["X"]), ("b1", T0 + H / 2, ["B"])] + late)
    out = af.exposure_shares(af.candidate_rows(imps), imps)
    assert _share(out, "b1", "B", w=24) == 0.0


def test_the_share_is_scale_free_so_a_sample_matches_the_population():
    base = [("i1", T0, ["A", "B"]), ("i2", T0 + H / 4, ["B"]), ("i3", T0 + H / 2, ["A"])]
    doubled = base + [(f"{i}x", t, c) for i, t, c in base]  # twice the users
    a = af.exposure_shares(af.candidate_rows(_imps(base)), _imps(base))
    b = af.exposure_shares(af.candidate_rows(_imps(doubled)), _imps(doubled))
    assert _share(a, "i3", "A") == _share(b, "i3", "A") == 0.5


def test_a_candidate_listed_twice_in_one_impression_is_counted_once():
    imps = _imps([("i1", T0, ["A", "A", "B"]), ("i2", T0 + H / 2, ["A"])])
    rows = af.candidate_rows(imps)
    assert rows.filter(pl.col("impression_id") == "i1").height == 2
    assert _share(af.exposure_shares(rows, imps), "i2", "A") == 1.0


def test_the_share_ignores_clicks_entirely():
    imps = _imps([("i1", T0, ["A", "B"]), ("i2", T0 + H / 2, ["A"])])
    with_labels = imps.with_columns(pl.Series("clicked_article_ids", [["A"], ["A"]]))
    flipped = imps.with_columns(pl.Series("clicked_article_ids", [["B"], []]))
    a = af.exposure_shares(af.candidate_rows(with_labels), with_labels)
    b = af.exposure_shares(af.candidate_rows(flipped), flipped)
    assert a.equals(b)


# ---------------------------------------------------------------- real store

@requires_store
@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_real_val_features_are_in_range(dataset):
    imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == dataset)
    arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == dataset)
    val = imps.filter(pl.col("split") == "val").head(2000)
    rows = af.candidate_rows(val)
    fresh = af.freshness_hours(rows, arts, imps)  # raises on negative or missing
    assert fresh.height == rows.height and fresh["freshness_hours"].min() >= 0
    exp = af.exposure_shares(rows, imps)
    for w in af.EXPOSURE_WINDOWS_HOURS:
        s = exp[f"exposure_share_{w}h"].drop_nulls()
        assert s.min() >= 0 and s.max() <= 1
