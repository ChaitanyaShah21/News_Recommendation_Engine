"""Adversarial tests for the Q4.1 ranking metrics.

The organising question (R10) is "what input makes this silently wrong rather
than loudly wrong?". For ranking metrics that is almost always a tie, a
degenerate impression, or a normalisation constant that is subtly unreachable.
"""

from __future__ import annotations

import numpy as np
import pytest

from newsrec.eval.metrics import (
    OPTIMISTIC,
    PESSIMISTIC,
    auc,
    evaluate_impressions,
    macro_mean,
    ndcg_at_k,
    rank_order,
    reciprocal_rank,
)


def _auc_by_pairs(scores, labels) -> float:
    """The O(P*N) definition, written out literally.

    The production code uses the rank identity instead. Implementing the
    definition separately here is the point: a mistake in the rank algebra
    (an off-by-one in P*(P+1)/2, ascending-vs-descending ranks, the wrong
    `rankdata` method) cannot be made in both at once.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (pos.size * neg.size)


# --------------------------------------------------------------------------
# Hand-computed values - the comprehension-check example, verified in code
# --------------------------------------------------------------------------


def test_clicked_at_rank_3_of_20_matches_the_hand_computation():
    # 20 candidates, exactly one clicked, our ranker puts it third.
    scores = np.arange(20, 0, -1, dtype=float)  # 20, 19, ... 1 -> already ordered
    labels = np.zeros(20, dtype=int)
    labels[2] = 1  # third position

    assert reciprocal_rank(scores, labels) == pytest.approx(1 / 3)
    # 19 positive-negative pairs; the 2 candidates above it are the only losses.
    assert auc(scores, labels) == pytest.approx(17 / 19)


def test_auc_depends_on_rack_size_where_mrr_does_not():
    """Same rank 3, different rack -> same MRR, very different AUC.

    This is the property that makes MIND's and EB-NeRD's AUCs incomparable to
    each other, so it is worth pinning rather than remembering.
    """
    big = (np.arange(20, 0, -1, dtype=float), np.eye(20, dtype=int)[2])
    small = (np.arange(4, 0, -1, dtype=float), np.eye(4, dtype=int)[2])

    assert reciprocal_rank(*big) == pytest.approx(reciprocal_rank(*small))
    assert auc(*big) == pytest.approx(17 / 19)
    assert auc(*small) == pytest.approx(1 / 3)


def test_ndcg_worked_example():
    # 6 candidates, 2 clicked, our ranking puts them at positions 2 and 5.
    scores = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([0, 1, 0, 0, 1, 0])
    expected = (1 / np.log2(3) + 1 / np.log2(6)) / (1 / np.log2(2) + 1 / np.log2(3))
    assert ndcg_at_k(scores, labels, 5) == pytest.approx(expected)


def test_perfect_and_worst_rankings():
    scores = np.array([3.0, 2.0, 1.0, 0.5])
    perfect = np.array([1, 1, 0, 0])
    worst = np.array([0, 0, 1, 1])

    assert auc(scores, perfect) == pytest.approx(1.0)
    assert ndcg_at_k(scores, perfect, 5) == pytest.approx(1.0)
    assert reciprocal_rank(scores, perfect) == pytest.approx(1.0)
    assert auc(scores, worst) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# AUC: the rank identity must equal the pair definition, ties included
# --------------------------------------------------------------------------


def test_auc_rank_identity_matches_the_pair_definition_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 30))
        # integer scores in a narrow range, so ties are frequent by construction
        scores = rng.integers(0, 4, size=n).astype(float)
        labels = rng.integers(0, 2, size=n)
        if labels.sum() in (0, n):
            continue
        assert auc(scores, labels) == pytest.approx(_auc_by_pairs(scores, labels))


def test_auc_is_exactly_one_half_when_every_score_ties():
    # Every pair is a tie, so every pair scores 0.5. If `rankdata` used
    # "ordinal" or "min" instead of "average", this comes out wrong.
    scores = np.zeros(10)
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    assert auc(scores, labels) == pytest.approx(0.5)


def test_auc_ignores_the_tie_policy():
    """AUC carries its own tie rule, so the two bounds must agree exactly.

    An AUC that moves between the pessimistic and optimistic runs would be a
    bug in the harness, not a finding - this pins that.
    """
    rng = np.random.default_rng(7)
    scores = rng.integers(0, 3, size=40).astype(float)
    labels = rng.integers(0, 2, size=40)
    metrics_p = evaluate_impressions([scores], [labels], tie=PESSIMISTIC)
    metrics_o = evaluate_impressions([scores], [labels], tie=OPTIMISTIC)
    assert metrics_p.auc[0] == metrics_o.auc[0]


# --------------------------------------------------------------------------
# Ties: the whole reason this module has a policy
# --------------------------------------------------------------------------


def test_tie_policy_overrides_argsort_stability():
    """The failure this module exists to prevent.

    `np.argsort` is stable, so leaving ties alone silently ranks them by
    position in the raw candidate list. With the clicked item sitting first in
    that list, a stable sort would report a perfect MRR of 1.0 for a ranker
    that produced no signal at all.
    """
    scores = np.zeros(4)  # the ranker distinguished nothing
    labels = np.array([1, 0, 0, 0])  # clicked item happens to be listed first

    assert np.argsort(-scores).tolist() == [0, 1, 2, 3]  # what stability would give
    assert rank_order(scores, labels, PESSIMISTIC).tolist() == [1, 2, 3, 0]
    assert reciprocal_rank(scores, labels, PESSIMISTIC) == pytest.approx(1 / 4)
    assert reciprocal_rank(scores, labels, OPTIMISTIC) == pytest.approx(1.0)


def test_pessimistic_never_exceeds_optimistic():
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(2, 25))
        scores = rng.integers(0, 3, size=n).astype(float)  # heavy ties
        labels = rng.integers(0, 2, size=n)
        if labels.sum() == 0:
            continue
        for k in (5, 10):
            assert ndcg_at_k(scores, labels, k, PESSIMISTIC) <= ndcg_at_k(
                scores, labels, k, OPTIMISTIC
            ) + 1e-12
        assert reciprocal_rank(scores, labels, PESSIMISTIC) <= reciprocal_rank(
            scores, labels, OPTIMISTIC
        ) + 1e-12


def test_ties_are_exact_float_equality_not_approximate():
    """Documents a real limit rather than pretending it away.

    Ties are detected by exact equality. The structurally important tie group -
    candidates sharing no term with the query, scoring exactly 0.0 from an
    empty sparse dot product - is caught exactly. Two candidates whose scores
    differ only by float rounding are *not* treated as tied, so the policy
    does not apply to them. Known and bounded, not silently assumed.
    """
    scores = np.array([1.0, 1.0 + 1e-15])
    labels = np.array([1, 0])
    # 1.0 + 1e-15 is genuinely a different float, so this is not a tie
    assert scores[0] != scores[1]
    assert reciprocal_rank(scores, labels, PESSIMISTIC) == pytest.approx(1 / 2)


def test_unknown_tie_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown tie policy"):
        rank_order(np.array([1.0, 2.0]), np.array([0, 1]), tie="random")


# --------------------------------------------------------------------------
# Degenerate impressions
# --------------------------------------------------------------------------


def test_no_clicks_gives_nan_not_zero():
    scores = np.array([3.0, 2.0, 1.0])
    labels = np.zeros(3, dtype=int)
    assert np.isnan(auc(scores, labels))
    assert np.isnan(reciprocal_rank(scores, labels))
    assert np.isnan(ndcg_at_k(scores, labels, 5))


def test_all_clicked_leaves_auc_undefined_but_the_rest_perfect():
    scores = np.array([3.0, 2.0, 1.0])
    labels = np.ones(3, dtype=int)
    assert np.isnan(auc(scores, labels))  # no negative to pair against
    assert reciprocal_rank(scores, labels) == pytest.approx(1.0)
    assert ndcg_at_k(scores, labels, 5) == pytest.approx(1.0)


def test_two_candidate_impression_is_a_coin_flip_not_an_error():
    """2,744 MIND val impressions have exactly 2 candidates - AUC is 0 or 1."""
    assert auc(np.array([2.0, 1.0]), np.array([1, 0])) == pytest.approx(1.0)
    assert auc(np.array([1.0, 2.0]), np.array([1, 0])) == pytest.approx(0.0)


def test_single_candidate_impression():
    assert np.isnan(auc(np.array([1.0]), np.array([1])))
    assert reciprocal_rank(np.array([1.0]), np.array([1])) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# nDCG normalisation - the constant that is easy to get subtly unreachable
# --------------------------------------------------------------------------


def test_ideal_dcg_is_capped_at_k_so_a_perfect_ranking_scores_one():
    """21 clicks cannot all fit inside nDCG@10.

    Normalising by the DCG of all 21 would cap this impression at ~0.51 for a
    reason that is about the cutoff, not about our ranking - and MIND really
    does have impressions with 21 clicks.
    """
    n = 60
    scores = np.arange(n, 0, -1, dtype=float)
    labels = np.zeros(n, dtype=int)
    labels[:21] = 1  # a flawless ranking: every click at the top
    assert ndcg_at_k(scores, labels, 10) == pytest.approx(1.0)
    assert ndcg_at_k(scores, labels, 5) == pytest.approx(1.0)


def test_ndcg_at_k_beyond_the_rack_becomes_ndcg_at_all():
    """EB-NeRD's median rack is 9, so nDCG@10 is routinely this case."""
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    assert ndcg_at_k(scores, labels, 10) == pytest.approx(ndcg_at_k(scores, labels, 4))
    # and it is emphatically not 1.0 just because the rack fits inside k
    assert ndcg_at_k(scores, labels, 10) < 1.0


def test_ndcg_never_decreases_when_a_clicked_item_moves_up():
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(4, 20))
        scores = rng.normal(size=n)
        labels = rng.integers(0, 2, size=n)
        if labels.sum() == 0:
            continue
        order = rank_order(scores, labels)
        # promote the last clicked item to the top
        clicked_positions = np.flatnonzero(labels[order])
        worst = order[clicked_positions[-1]]
        bumped = scores.copy()
        bumped[worst] = scores.max() + 1.0
        for k in (5, 10):
            assert ndcg_at_k(bumped, labels, k) >= ndcg_at_k(scores, labels, k) - 1e-12


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be positive"):
        ndcg_at_k(np.array([1.0, 2.0]), np.array([0, 1]), 0)


# --------------------------------------------------------------------------
# Input validation - the silent-corruption guards
# --------------------------------------------------------------------------


def test_nan_scores_are_rejected_rather_than_ranked_last():
    """A mean over zero history vectors produces NaN - Phase 3's cold-start trap.

    `np.argsort` sorts NaN to the end without complaint, so an unguarded
    harness would score a NaN-producing user as "ranked everything worst"
    instead of failing.
    """
    scores = np.array([1.0, np.nan, 3.0])
    labels = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="NaN or inf"):
        reciprocal_rank(scores, labels)
    with pytest.raises(ValueError, match="NaN or inf"):
        auc(scores, labels)


def test_infinite_scores_are_rejected():
    with pytest.raises(ValueError, match="NaN or inf"):
        auc(np.array([1.0, np.inf]), np.array([0, 1]))


def test_misaligned_scores_and_labels_are_rejected():
    with pytest.raises(ValueError, match="aligned position for position"):
        auc(np.array([1.0, 2.0, 3.0]), np.array([0, 1]))


def test_non_binary_labels_are_rejected():
    with pytest.raises(ValueError, match="binary clicks"):
        auc(np.array([1.0, 2.0]), np.array([0, 2]))


def test_misaligned_impression_lists_are_rejected():
    with pytest.raises(ValueError, match="impressions but labels has"):
        evaluate_impressions([np.array([1.0, 2.0])], [])


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_evaluate_impressions_handles_ragged_racks():
    """MIND racks run 2 to 295 candidates, so these can never be a matrix."""
    scores = [np.array([3.0, 1.0]), np.arange(50, 0, -1, dtype=float)]
    labels = [np.array([1, 0]), np.eye(50, dtype=int)[7]]
    m = evaluate_impressions(scores, labels)
    assert m.auc.shape == (2,)
    assert m.mrr[0] == pytest.approx(1.0)
    assert m.mrr[1] == pytest.approx(1 / 8)


def test_macro_mean_reports_undefined_rather_than_hiding_them():
    values = np.array([1.0, np.nan, 0.0, np.nan])
    mean, n_defined, n_undefined = macro_mean(values)
    assert mean == pytest.approx(0.5)
    assert (n_defined, n_undefined) == (2, 2)


def test_macro_mean_on_all_undefined_returns_nan_without_warning():
    mean, n_defined, n_undefined = macro_mean(np.full(5, np.nan))
    assert np.isnan(mean)
    assert (n_defined, n_undefined) == (0, 5)


def test_undefined_impressions_do_not_drag_the_mean_toward_zero():
    """The reason NaN is used instead of 0.0.

    Two impressions, one perfect and one with no clicks at all. The honest
    answer is 1.0 over one defined impression, not 0.5 over two.
    """
    scores = [np.array([2.0, 1.0]), np.array([2.0, 1.0])]
    labels = [np.array([1, 0]), np.array([0, 0])]
    m = evaluate_impressions(scores, labels)
    mean, n_defined, n_undefined = macro_mean(m.ndcg_at_5)
    assert mean == pytest.approx(1.0)
    assert (n_defined, n_undefined) == (1, 1)
