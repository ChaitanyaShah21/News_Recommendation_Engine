"""Adversarial tests for Q9's ablation machinery.

This module deliberately contains a leaky function, so the first job of these
tests is to prove it is quarantined: that it leaks (otherwise the ablation
measures nothing), that the safe counterpart still refuses to, and that nothing
in the pipeline imports it.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from newsrec.eval import ablation

T0 = datetime(2026, 8, 25, 10, 0, 0)


def _impressions(rows, split="val"):
    return pl.DataFrame(
        {
            "dataset": ["t"] * len(rows),
            "split": [split] * len(rows),
            "impression_id": [r[0] for r in rows],
            "user_id": [r[1] for r in rows],
            "timestamp": [T0] * len(rows),
            "candidate_article_ids": [r[2] for r in rows],
            "clicked_article_ids": [r[3] for r in rows],
        }
    )


# --------------------------------------------------------------------------
# The leaky feature, and its quarantine
# --------------------------------------------------------------------------


def test_future_click_counts_really_does_see_the_evaluated_window():
    """If it did not leak, the ablation would measure nothing at all."""
    imps = _impressions(
        [("i1", "u1", ["a", "b"], ["a"]), ("i2", "u2", ["a", "b"], ["a"])], split="val"
    )
    counts = ablation.future_click_counts(imps, ["a", "b"])
    assert counts.tolist() == [2.0, 0.0]


def test_future_click_counts_does_not_guard_the_split_and_that_is_deliberate():
    """Its safe counterpart raises on a non-train split; this one must not.

    Pinning the asymmetry means a well-meaning "consistency" edit that adds a
    guard here would fail loudly rather than silently disarming the ablation.
    """
    for split in ("train", "val", "test"):
        imps = _impressions([("i1", "u1", ["a"], ["a"])], split=split)
        assert ablation.future_click_counts(imps, ["a"]).tolist() == [1.0]


def test_the_leaky_function_is_not_imported_anywhere_in_the_package():
    """Quarantine, asserted rather than trusted to code review."""
    from pathlib import Path

    package = Path(__file__).resolve().parent.parent / "src" / "newsrec"
    offenders = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if path.name != "ablation.py" and "future_click_counts" in path.read_text()
    ]
    assert offenders == [], f"the leaky feature is referenced by {offenders}"


def test_ignores_articles_outside_the_catalogue():
    imps = _impressions([("i1", "u1", ["a", "ghost"], ["ghost"])])
    assert ablation.future_click_counts(imps, ["a"]).tolist() == [0.0]


# --------------------------------------------------------------------------
# The legitimate serving-time feature
# --------------------------------------------------------------------------


def test_seen_before_marks_exactly_the_candidates_in_the_users_history():
    candidates = [np.array([3, 7, 1]), np.array([2, 3])]
    history = [np.array([1, 3]), np.zeros(0, dtype=np.int32)]
    out = ablation.seen_before_feature(candidates, history)
    assert out[0].tolist() == [1.0, 0.0, 1.0]
    assert out[1].tolist() == [0.0, 0.0]


def test_seen_before_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="candidate lists but"):
        ablation.seen_before_feature([np.array([1])], [])


# --------------------------------------------------------------------------
# Rank normalisation and blending
# --------------------------------------------------------------------------


def test_rank_normalise_maps_to_the_unit_interval_preserving_order():
    out = ablation.rank_normalise(np.array([10.0, -5.0, 3.0, 100.0]))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert np.argsort(out).tolist() == np.argsort([10.0, -5.0, 3.0, 100.0]).tolist()


def test_rank_normalise_is_invariant_to_the_score_scale():
    """The reason blending is rank-based: BM25 runs 0-40, cosine runs [-1, 1].

    Without this, adding two score vectors would let the wider scale decide the
    ranking regardless of which carried the signal.
    """
    base = np.array([1.0, 2.0, 3.0])
    assert ablation.rank_normalise(base) == pytest.approx(
        ablation.rank_normalise(base * 1000.0)
    )
    assert ablation.rank_normalise(base) == pytest.approx(
        ablation.rank_normalise(base - 500.0)
    )


def test_rank_normalise_averages_ties():
    out = ablation.rank_normalise(np.array([5.0, 5.0, 9.0]))
    assert out[0] == pytest.approx(out[1])
    assert out[2] > out[0]


def test_rank_normalise_of_a_single_candidate_is_neutral_not_a_division_by_zero():
    assert ablation.rank_normalise(np.array([4.2])).tolist() == [0.5]
    assert ablation.rank_normalise(np.zeros(0)).tolist() == []


def test_blend_moves_a_flagged_candidate_up_without_erasing_the_base_order():
    base = [np.array([0.9, 0.5, 0.1])]        # ranks -> 1.0, 0.5, 0.0
    feature = [np.array([0.0, 0.0, 1.0])]     # flag the worst candidate
    out = ablation.blend(base, feature, weight=0.5)[0]
    assert out.tolist() == pytest.approx([1.0, 0.5, 0.5])
    # weight 0.5 lifts the flagged candidate level with the middle one but not
    # above the best - the base ordering still counts for something
    assert out[0] > out[2]


def test_a_zero_weight_blend_is_the_base_ranking_unchanged():
    base = [np.array([0.9, 0.5, 0.1])]
    feature = [np.array([1.0, 1.0, 1.0])]
    out = ablation.blend(base, feature, weight=0.0)[0]
    assert np.argsort(out).tolist() == np.argsort(base[0]).tolist()


def test_a_constant_feature_cannot_change_any_ranking():
    """Sanity: a feature that says the same thing about every candidate is inert."""
    rng = np.random.default_rng(0)
    base = [rng.normal(size=6)]
    feature = [np.full(6, 1.0)]
    out = ablation.blend(base, feature, weight=5.0)[0]
    assert np.argsort(out).tolist() == np.argsort(base[0]).tolist()


def test_blend_rejects_misaligned_lists_and_lengths():
    with pytest.raises(ValueError, match="base lists but"):
        ablation.blend([np.array([1.0])], [])
    with pytest.raises(ValueError, match="disagree on candidate count"):
        ablation.blend([np.array([1.0, 2.0])], [np.array([1.0])])
