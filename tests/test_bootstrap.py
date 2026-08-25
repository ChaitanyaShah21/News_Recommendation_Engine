"""Adversarial tests for Q4.4's bootstrap and Q4.3's slicing.

A bootstrap is a machine for producing confident-looking intervals from any
input, correct or not. So the tests here mostly ask whether the interval means
what it claims: does it cover a known truth at the right rate, does it shrink
with sample size, does it treat an undefined value as absent rather than zero,
and does the coverage version really recompute the union instead of averaging.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl
import pytest

from newsrec.eval.bootstrap import (
    bootstrap_coverage,
    bootstrap_mean,
    draw_indices,
)
from newsrec.eval.slices import (
    cold_start_mask,
    exposure_counts,
    head_set_from_counts,
    head_tail_masks,
    train_popularity_head_set,
)

# --------------------------------------------------------------------------
# bootstrap_mean
# --------------------------------------------------------------------------


def test_the_point_estimate_is_the_plain_mean():
    values = np.array([0.1, 0.5, 0.9, 0.3])
    assert bootstrap_mean(values, n_resamples=200).point == pytest.approx(values.mean())


def test_the_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.2, size=500)
    ci = bootstrap_mean(values, n_resamples=500)
    assert ci.low < ci.point < ci.high


def test_the_interval_narrows_as_the_sample_grows():
    """The property that makes a CI worth reporting at all."""
    rng = np.random.default_rng(1)
    small = bootstrap_mean(rng.normal(0, 1, size=50), n_resamples=500)
    large = bootstrap_mean(rng.normal(0, 1, size=5000), n_resamples=500)
    assert (large.high - large.low) < (small.high - small.low) / 3


def test_zero_variance_data_gives_a_zero_width_interval():
    ci = bootstrap_mean(np.full(100, 0.42), n_resamples=200)
    assert ci.low == pytest.approx(0.42)
    assert ci.high == pytest.approx(0.42)


def test_the_interval_covers_the_true_mean_at_roughly_the_nominal_rate():
    """The only test that checks the bootstrap is calibrated, not just plausible.

    A machine that returns [point-0.001, point+0.001] passes every other test
    here and would cover the truth almost never.
    """
    rng = np.random.default_rng(7)
    covered = 0
    trials = 200
    for _ in range(trials):
        sample = rng.normal(0.0, 1.0, size=300)
        ci = bootstrap_mean(sample, n_resamples=300, seed=int(rng.integers(1 << 30)))
        covered += ci.low <= 0.0 <= ci.high
    # nominal 95%; allow slack for 200 trials and the percentile method's bias
    assert 0.88 <= covered / trials <= 1.0


def test_nan_values_are_excluded_not_counted_as_zero():
    values = np.array([1.0, np.nan, 1.0, np.nan])
    ci = bootstrap_mean(values, n_resamples=200)
    assert ci.point == pytest.approx(1.0)
    assert ci.n == 2
    assert ci.low == pytest.approx(1.0) and ci.high == pytest.approx(1.0)


def test_an_all_nan_input_returns_nan_rather_than_a_fabricated_number():
    ci = bootstrap_mean(np.full(10, np.nan), n_resamples=50)
    assert np.isnan(ci.point) and np.isnan(ci.low) and np.isnan(ci.high)
    assert ci.n == 0


def test_an_empty_input_returns_nan():
    ci = bootstrap_mean(np.zeros(0), n_resamples=50)
    assert np.isnan(ci.point)
    assert ci.n == 0


def test_results_are_reproducible_from_the_seed():
    values = np.random.default_rng(3).normal(size=200)
    a = bootstrap_mean(values, n_resamples=300, seed=99)
    b = bootstrap_mean(values, n_resamples=300, seed=99)
    c = bootstrap_mean(values, n_resamples=300, seed=100)
    assert (a.low, a.high) == (b.low, b.high)
    assert (a.low, a.high) != (c.low, c.high)


def test_a_shared_index_draw_is_honoured():
    """Metrics must be able to share one resample, or difference intervals lie."""
    values = np.arange(20, dtype=float)
    idx = draw_indices(20, 100, seed=5)
    a = bootstrap_mean(values, indices=idx)
    b = bootstrap_mean(values, n_resamples=100, seed=5)
    assert (a.low, a.high) == (b.low, b.high)


def test_draw_indices_resamples_with_replacement_and_the_right_shape():
    idx = draw_indices(10, 50, seed=1)
    assert idx.shape == (50, 10)
    assert idx.min() >= 0 and idx.max() < 10
    # with replacement: some row must contain a repeat
    assert any(len(np.unique(row)) < 10 for row in idx)


def test_draw_indices_rejects_an_empty_population():
    with pytest.raises(ValueError, match="cannot resample"):
        draw_indices(0, 10, seed=1)


# --------------------------------------------------------------------------
# bootstrap_coverage - the metric that is not a mean
# --------------------------------------------------------------------------


def test_coverage_point_estimate_matches_the_union():
    lists = [np.array([0, 1]), np.array([1, 2]), np.array([7])]
    ci = bootstrap_coverage(lists, catalogue_size=10, n_resamples=100)
    assert ci.point == pytest.approx(0.4)  # {0,1,2,7}


def test_coverage_is_recomputed_per_resample_not_averaged():
    """The mean of per-list coverages would be 0.1 here, not 0.3.

    If this ever regressed to averaging, every coverage interval would sit an
    order of magnitude too low and still look like a plausible number.
    """
    lists = [np.array([0]), np.array([1]), np.array([2])]
    ci = bootstrap_coverage(lists, catalogue_size=10, n_resamples=200)
    assert ci.point == pytest.approx(0.3)
    # a resample of 3 draws with replacement can cover at most 3 articles and
    # at least 1, so the interval must sit strictly inside that band
    assert 0.1 <= ci.low <= ci.high <= 0.3


def test_the_coverage_bootstrap_is_biased_low_and_is_not_a_confidence_interval():
    """D27, pinned as a property rather than left as a surprise.

    Drawing n items with replacement yields ~63.2% distinct items. For a mean
    that is harmless - a duplicate still contributes its value. For a union a
    duplicate contributes nothing new, so every resample can only lose articles
    and the whole distribution shifts below the point estimate.

    This test exists so nobody later "fixes" the reporting by quietly treating
    `low`/`high` as a CI: the assertion states plainly that they cannot be one.
    """
    # catalogue deliberately much larger than the lists can saturate: with 50
    # articles the 60 lists cover all of them and the bias has nowhere to show
    rng = np.random.default_rng(4)
    lists = [rng.choice(500, size=5, replace=False) for _ in range(60)]
    ci = bootstrap_coverage(lists, catalogue_size=500, n_resamples=300)
    assert ci.point < 0.9, "fixture must not saturate the catalogue"
    assert ci.high < ci.point, "the interval must sit strictly below the point estimate"


def test_the_pivotal_correction_reflects_the_interval_through_the_point():
    rng = np.random.default_rng(4)
    lists = [rng.choice(500, size=5, replace=False) for _ in range(60)]
    ci = bootstrap_coverage(lists, catalogue_size=500, n_resamples=300)
    lo, hi = ci.pivotal()
    assert lo == pytest.approx(2 * ci.point - ci.high)
    assert hi == pytest.approx(2 * ci.point - ci.low)
    # and it lands entirely ABOVE the point, which is why it is not the headline
    assert lo > ci.point
    assert (hi - lo) == pytest.approx(ci.high - ci.low)


def test_a_resample_holds_about_63_percent_of_the_distinct_items():
    """The single fact behind the coverage bias, pinned directly."""
    for n in (100, 1000, 5000):
        idx = draw_indices(n, 50, seed=13)
        fraction = np.mean([len(np.unique(row)) / n for row in idx])
        assert 0.60 < fraction < 0.66


def test_coverage_handles_ragged_and_empty_lists():
    """MIND's 1,407 cold-start users retrieve nothing at all."""
    lists = [np.array([0, 1, 2]), np.zeros(0, dtype=np.int64), np.array([5])]
    ci = bootstrap_coverage(lists, catalogue_size=10, n_resamples=100)
    assert ci.point == pytest.approx(0.4)
    assert not np.isnan(ci.low)


def test_coverage_with_every_list_empty_is_zero_not_nan():
    lists = [np.zeros(0, dtype=np.int64) for _ in range(5)]
    ci = bootstrap_coverage(lists, catalogue_size=10, n_resamples=50)
    assert ci.point == pytest.approx(0.0)
    assert ci.low == pytest.approx(0.0) and ci.high == pytest.approx(0.0)


def test_coverage_vectorised_gather_matches_a_naive_union():
    """Pins the ragged-gather index arithmetic against an obvious implementation."""
    rng = np.random.default_rng(11)
    lists = [rng.choice(30, size=int(rng.integers(0, 6)), replace=False) for _ in range(25)]
    idx = draw_indices(25, 40, seed=2)
    ci = bootstrap_coverage(lists, catalogue_size=30, indices=idx)

    naive = []
    for draw in idx:
        seen = set()
        for i in draw:
            seen.update(lists[i].tolist())
        naive.append(len(seen) / 30)
    assert ci.low == pytest.approx(np.percentile(naive, 2.5))
    assert ci.high == pytest.approx(np.percentile(naive, 97.5))


def test_coverage_rejects_a_nonpositive_catalogue():
    with pytest.raises(ValueError, match="catalogue_size must be positive"):
        bootstrap_coverage([np.array([1])], catalogue_size=0)


# --------------------------------------------------------------------------
# Slices
# --------------------------------------------------------------------------


def test_cold_start_threshold_is_inclusive_and_includes_zero_history():
    lens = np.array([0, 1, 5, 6, 100])
    assert cold_start_mask(lens, threshold=5).tolist() == [True, True, True, False, False]


def test_cold_start_rejects_a_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        cold_start_mask(np.array([1]), threshold=-1)


def test_exposure_is_counted_from_candidates_not_clicks():
    """The rule that keeps the slice from being circular."""
    frame = pl.DataFrame(
        {
            "candidate_article_ids": [["a", "b"], ["a", "c"]],
            "clicked_article_ids": [["b"], ["c"]],
        }
    )
    counts = exposure_counts(frame)
    assert counts == Counter({"a": 2, "b": 1, "c": 1})


def test_head_set_takes_the_smallest_set_reaching_the_target_share():
    counts = Counter({"a": 50, "b": 30, "c": 15, "d": 5})
    # a alone is 50% of 100 -> the set that *reaches* the target stops there
    assert head_set_from_counts(counts, 0.5) == {"a"}
    assert head_set_from_counts(counts, 0.8) == {"a", "b"}
    assert head_set_from_counts(counts, 1.0) == {"a", "b", "c", "d"}


def test_head_set_is_deterministic_when_counts_tie():
    """Counter iteration order must not decide which article is 'head'."""
    counts = Counter({"z": 10, "a": 10, "m": 10})
    first = head_set_from_counts(counts, 0.5)
    for _ in range(5):
        assert head_set_from_counts(Counter(dict(reversed(list(counts.items())))), 0.5) == first


def test_head_set_rejects_an_out_of_range_fraction():
    with pytest.raises(ValueError, match="fraction must be in"):
        head_set_from_counts(Counter({"a": 1}), 0.0)
    with pytest.raises(ValueError, match="fraction must be in"):
        head_set_from_counts(Counter({"a": 1}), 1.5)


def test_head_set_of_an_empty_counter_is_empty():
    assert head_set_from_counts(Counter(), 0.5) == set()


def test_mixed_impressions_land_in_neither_slice_and_are_counted():
    """29.3% of MIND's val impressions carry more than one click."""
    clicked = [["a"], ["x"], ["a", "x"], ["a", "b"], []]
    head = {"a", "b"}
    head_mask, tail_mask, mixed = head_tail_masks(clicked, head)
    assert head_mask.tolist() == [True, False, False, True, False]
    assert tail_mask.tolist() == [False, True, False, False, False]
    assert mixed == 1
    # the click-less impression is in neither, and is not counted as mixed
    assert not head_mask[4] and not tail_mask[4]


def test_the_two_slices_never_overlap():
    rng = np.random.default_rng(6)
    clicked = [
        [str(x) for x in rng.choice(10, size=int(rng.integers(1, 4)), replace=False)]
        for _ in range(200)
    ]
    head = {str(x) for x in range(5)}
    head_mask, tail_mask, _ = head_tail_masks(clicked, head)
    assert not (head_mask & tail_mask).any()


def test_train_popularity_head_ignores_never_clicked_articles():
    """88.2% of MIND's corpus has zero train clicks; they cannot be 'head'."""
    counts = np.array([100.0, 40.0, 0.0, 0.0])
    ids = ["a", "b", "c", "d"]
    head = train_popularity_head_set(counts, ids, 0.5)
    assert head == {"a"}
    assert "c" not in head and "d" not in head
