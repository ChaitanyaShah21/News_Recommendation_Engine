"""/ D35: multi-scale recency-decay features. R10 adversarial suite.

Constructed data for the properties real files don't point at: an extreme
half-life that would underflow, a near-cancelling pair of vectors, an article
clicked twice, a history click tied with its impression. Two real-store tests
at the end pin the claims that matter most for the write-up: the infinite scale
IS A1's feature, and no real EB-NeRD history click postdates its impression.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from newsrec.features import history as hf
from newsrec.retrieval import semantic_search

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
requires_store = pytest.mark.skipif(
    not (PROCESSED / "impressions.parquet").exists(), reason="feature store not built"
)
T0 = datetime(2023, 5, 1)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- decay_weights

def test_the_newest_item_weighs_exactly_one_and_each_half_life_halves_it():
    w = hf.decay_weights(np.array([0.0, 6.0, 12.0]), half_life=6.0)
    assert np.allclose(w, [1.0, 0.5, 0.25])


def test_weights_depend_only_on_age_differences_so_impression_time_cancels():
    a = hf.decay_weights(np.array([5.0, 11.0, 17.0]), half_life=6.0)
    b = hf.decay_weights(np.array([500.0, 506.0, 512.0]), half_life=6.0)
    assert np.allclose(a, b) and a[0] == 1.0


def test_an_extreme_half_life_cannot_underflow_to_a_zero_sum():
    # Measured from the impression this is 0.5**(1e6) = 0.0 for every item -> 0/0.
    w = hf.decay_weights(np.array([1e6, 1e6 + 1, 2e6]), half_life=1e-3)
    assert w.sum() >= 1.0 and np.all(np.isfinite(w))


def test_infinite_half_life_is_the_plain_mean():
    assert np.array_equal(hf.decay_weights(np.array([0.0, 3.0, 99.0]), math.inf), np.ones(3))


@pytest.mark.parametrize("ages", [[-1.0, 0.0], [0.0, float("nan")], [0.0, float("inf")]])
def test_a_negative_or_non_finite_age_is_refused_not_clipped(ages):
    with pytest.raises(ValueError):
        hf.decay_weights(np.array(ages), half_life=6.0)


@pytest.mark.parametrize("h", [0.0, -1.0])
def test_a_non_positive_half_life_is_refused(h):
    with pytest.raises(ValueError):
        hf.decay_weights(np.array([0.0, 1.0]), half_life=h)


# ---------------------------------------------------- decayed user vectors

def _catalogue(rng, n=8, dim=16):
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return [f"a{i}" for i in range(n)], emb


def test_infinite_half_life_reproduces_a1_user_vectors_exactly():
    rng = np.random.default_rng(0)
    ids, emb = _catalogue(rng)
    emb[7] = -emb[6]  # a user reading exactly these two cancels to norm 0
    history = pl.DataFrame({
        "user_id": ["dup", "unknown", "empty", "null", "cancel", "long"],
        "history_article_ids": [
            ["a0", "a1", "a0"],            # duplicate
            ["a2", "zzz"],                 # an article absent from the store
            [],                            # cold start
            None,                          # null history
            ["a6", "a7"],                  # cancelling pair
            # Longer than N_RECENT, and its oldest clicks differ from its newest,
            # so taking the head instead of the tail changes the answer. (A cycle
            # over the same articles hid exactly that bug: mutation M8.)
            ["a0"] * 150 + [f"a{1 + i % 5}" for i in range(100)],
        ],
    })
    ours = hf.build_decayed_user_vectors(history, ids, emb, half_life=math.inf)
    a1 = semantic_search.build_user_vectors(history, ids, emb, n_recent=hf.N_RECENT)
    assert ours.user_ids == a1.user_ids
    assert np.array_equal(ours.has_query, a1.has_query)
    assert np.allclose(ours.matrix, a1.matrix, atol=1e-6)


def test_a_near_cancelling_history_is_treated_as_having_no_direction():
    # Two clicks almost exactly opposite: their mean is floating-point residue.
    # MIN_NORM must judge the MEAN, not the raw weighted sum - skipping the
    # division by sum(w) inflates the residue past the threshold (mutation M6),
    # and the user then gets a confident "profile" that is pure noise.
    rng = np.random.default_rng(1)
    ids, emb = _catalogue(rng, n=2)
    perp = rng.normal(size=emb.shape[1]).astype(np.float32)
    perp -= (perp @ emb[0]) * emb[0]
    perp /= np.linalg.norm(perp)
    emb[1] = -emb[0] + 1.5e-6 * perp  # mean norm ~7.5e-7 < MIN_NORM; sum ~1.5e-6 > it
    history = pl.DataFrame({"user_id": ["u"], "history_article_ids": [["a0", "a1"]]})
    ours = hf.build_decayed_user_vectors(history, ids, emb, half_life=math.inf)
    a1 = semantic_search.build_user_vectors(history, ids, emb, n_recent=hf.N_RECENT)
    assert not a1.has_query[0]
    assert not ours.has_query[0]


def test_decay_pulls_the_profile_towards_recent_reading():
    ids, emb = ["old", "new"], np.eye(2, dtype=np.float32)
    history = pl.DataFrame({
        "user_id": ["u"],
        "history_article_ids": [["old", "new"]],
        "history_timestamps": [[T0, T0 + timedelta(days=10)]],
    })
    short = hf.build_decayed_user_vectors(history, ids, emb, half_life=24.0).matrix[0]
    flat = hf.build_decayed_user_vectors(history, ids, emb, half_life=math.inf).matrix[0]
    assert short @ emb[1] > 0.99 and short @ emb[0] < 0.01
    assert np.isclose(flat @ emb[0], flat @ emb[1])


def test_an_article_read_again_counts_once_at_its_newest_click():
    ids, emb = ["A", "B"], np.eye(2, dtype=np.float32)
    history = pl.DataFrame({
        "user_id": ["u"],
        "history_article_ids": [["A", "B", "A"]],
        "history_timestamps": [[T0, T0 + timedelta(days=9), T0 + timedelta(days=10)]],
    })
    v = hf.build_decayed_user_vectors(history, ids, emb, half_life=24.0).matrix[0]
    # Keeping A's OLD click would make B (a day newer) dominate instead.
    assert v @ emb[0] > v @ emb[1]


def test_mind_ages_are_list_positions_when_there_are_no_timestamps():
    ids, emb = ["A", "B", "C"], np.eye(3, dtype=np.float32)
    history = pl.DataFrame({"user_id": ["u"], "history_article_ids": [["A", "B", "C"]]})
    v = hf.build_decayed_user_vectors(history, ids, emb, half_life=1.0).matrix[0]
    raw = np.array([0.25, 0.5, 1.0])
    assert np.allclose(v, raw / np.linalg.norm(raw), atol=1e-6)


def test_timestamps_that_do_not_align_with_articles_are_refused():
    history = pl.DataFrame({
        "user_id": ["u"],
        "history_article_ids": [["A", "B"]],
        "history_timestamps": [[T0]],
    })
    with pytest.raises(ValueError):
        hf.build_decayed_user_vectors(history, ["A", "B"], np.eye(2, dtype=np.float32), 24.0)


# ---------------------------------------------------- category shares

def test_category_shares_sum_to_one_and_follow_decay():
    cats = {"A": "sport", "B": "sport", "C": "news"}
    history = pl.DataFrame({
        "user_id": ["u", "cold"],
        "history_article_ids": [["A", "B", "C"], []],
    })
    flat = hf.user_category_shares(history, cats, half_life=math.inf)
    assert set(flat["user_id"]) == {"u"}  # cold user has no rows -> share 0 at join
    shares = dict(zip(flat["category"], flat["share"]))
    assert np.isclose(shares["sport"], 2 / 3) and np.isclose(sum(shares.values()), 1.0)
    decayed = dict(zip(*hf.user_category_shares(history, cats, half_life=1.0)
                       .select("category", "share").to_dict(as_series=False).values()))
    # positions: C=1, B=0.5, A=0.25 -> news 1/1.75, sport 0.75/1.75
    assert np.isclose(decayed["news"], 1 / 1.75)


# ---------------------------------------------------- the boundary guard

def _pair(click_offset: timedelta, split="val"):
    imps = pl.DataFrame({"impression_id": ["i1"], "user_id": ["u"], "split": [split],
                         "timestamp": [T0 + timedelta(days=1)]})
    hist = pl.DataFrame({"user_id": ["u"], "split": [split],
                         "history_timestamps": [[T0, T0 + timedelta(days=1) + click_offset]]})
    return imps, hist


def test_hours_since_last_click_is_measured_from_the_newest_click():
    imps, hist = _pair(timedelta(hours=-5))
    out = hf.hours_since_last_click(imps, hist)
    assert out["hours_since_last_click"].to_list() == [5.0]


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(seconds=1), timedelta(days=3)])
def test_a_history_click_at_or_after_its_impression_raises(offset):
    imps, hist = _pair(offset)
    with pytest.raises(ValueError, match="leakage"):
        hf.hours_since_last_click(imps, hist)


def test_mixing_splits_is_refused_because_the_join_would_pick_a_wrong_snapshot():
    imps, hist = _pair(timedelta(hours=-5))
    hist = pl.concat([hist, hist.with_columns(pl.lit("train").alias("split"))])
    with pytest.raises(ValueError, match="Landmine 5"):
        hf.hours_since_last_click(imps, hist)


# ---------------------------------------------------- real store

@requires_store
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_no_real_ebnerd_history_click_postdates_its_impression(split):
    imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == "ebnerd") & (pl.col("split") == split))
    hist = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == "ebnerd") & (pl.col("split") == split))
    out = hf.hours_since_last_click(imps, hist)  # raises on any violation
    assert out.height == imps.height
    assert out["hours_since_last_click"].drop_nulls().min() > 0


@requires_store
def test_infinite_scale_equals_a1_on_real_mind_val():
    from newsrec.retrieval import semantic
    ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset="mind")
    hist = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == "mind") & (pl.col("split") == "val")).head(3000)
    ours = hf.build_decayed_user_vectors(hist, ids, emb, half_life=math.inf)
    a1 = semantic_search.build_user_vectors(hist, ids, emb, n_recent=hf.N_RECENT)
    assert np.array_equal(ours.has_query, a1.has_query)
    assert np.allclose(ours.matrix, a1.matrix, atol=1e-6)
