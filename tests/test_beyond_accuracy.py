"""Adversarial tests for the Q4.3 beyond-accuracy metrics.

Two of these metrics use closed forms in place of an explicit pairwise loop, so
the organising risk is an algebra error that returns a plausible number. Both
are checked against a brute-force implementation written separately here, the
same trick the AUC suite uses.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from newsrec.eval.beyond_accuracy import (
    coverage,
    evaluate_lists,
    intra_list_diversity_category,
    intra_list_diversity_embedding,
    item_novelty,
    list_novelty,
)


def _unit(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, dim))
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def _ild_embedding_by_pairs(rows, embeddings) -> float:
    """The explicit O(k^2) definition, for comparison against the closed form."""
    v = embeddings[rows].astype(np.float64)
    pairs = list(itertools.combinations(range(len(rows)), 2))
    return 1.0 - float(np.mean([v[i] @ v[j] for i, j in pairs]))


def _ild_category_by_pairs(rows, categories) -> float:
    c = categories[rows]
    pairs = list(itertools.combinations(range(len(rows)), 2))
    return float(np.mean([c[i] != c[j] for i, j in pairs]))


# --------------------------------------------------------------------------
# Intra-list diversity, embedding basis
# --------------------------------------------------------------------------


def test_embedding_ild_closed_form_matches_the_pairwise_definition():
    embeddings = _unit(40)
    rng = np.random.default_rng(1)
    for _ in range(200):
        k = int(rng.integers(2, 15))
        rows = rng.choice(40, size=k, replace=False)
        assert intra_list_diversity_embedding(rows, embeddings) == pytest.approx(
            _ild_embedding_by_pairs(rows, embeddings), abs=1e-6
        )


def test_a_list_of_identical_articles_has_zero_embedding_diversity():
    """The Popeyes case: five copies of the same story score exactly 0."""
    embeddings = _unit(5)
    embeddings[1:] = embeddings[0]
    assert intra_list_diversity_embedding(np.arange(5), embeddings) == pytest.approx(
        0.0, abs=1e-6
    )


def test_repeating_the_same_article_row_also_scores_zero():
    embeddings = _unit(5)
    rows = np.array([2, 2, 2, 2])
    assert intra_list_diversity_embedding(rows, embeddings) == pytest.approx(0.0, abs=1e-6)


def test_orthogonal_articles_score_exactly_one():
    embeddings = np.eye(4, dtype=np.float32)
    assert intra_list_diversity_embedding(np.arange(4), embeddings) == pytest.approx(1.0)


def test_opposed_articles_exceed_one():
    """Cosine distance runs [0, 2], so ILD is not capped at 1 on this basis.

    Worth pinning: a reader who assumes [0, 1] would misread the number, and
    the category basis really is capped at 1 - the two are not interchangeable.
    """
    embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    assert intra_list_diversity_embedding(np.arange(2), embeddings) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Intra-list diversity, category basis
# --------------------------------------------------------------------------


def test_category_ild_closed_form_matches_the_pairwise_definition():
    rng = np.random.default_rng(2)
    categories = rng.integers(0, 5, size=40)
    for _ in range(200):
        k = int(rng.integers(2, 15))
        rows = rng.choice(40, size=k, replace=False)
        assert intra_list_diversity_category(rows, categories) == pytest.approx(
            _ild_category_by_pairs(rows, categories)
        )


def test_all_one_category_scores_zero_and_all_distinct_scores_one():
    categories = np.array([7, 7, 7, 7, 1, 2, 3, 4])
    assert intra_list_diversity_category(np.arange(4), categories) == pytest.approx(0.0)
    assert intra_list_diversity_category(np.arange(4, 8), categories) == pytest.approx(1.0)


def test_category_ild_matches_the_worked_number_from_the_comprehension_check():
    """85% of pairs sharing a category must give 0.15, not 0.85."""
    # 10 items: 8 in category 0 (28 same-pairs), 2 in category 1 (1 same-pair)
    categories = np.array([0] * 8 + [1] * 2)
    rows = np.arange(10)
    # 45 pairs total, 29 same-category -> 16/45 different
    assert intra_list_diversity_category(rows, categories) == pytest.approx(16 / 45)


def test_category_ild_handles_string_categories():
    """The real column is strings, not integer codes."""
    categories = np.array(["news", "news", "sport"])
    assert intra_list_diversity_category(np.arange(3), categories) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# Short and empty lists
# --------------------------------------------------------------------------


def test_a_single_article_list_is_undefined_not_zero():
    """0.0 would read as 'perfectly uniform' rather than 'no pairs exist'."""
    embeddings = _unit(5)
    categories = np.zeros(5, dtype=int)
    assert np.isnan(intra_list_diversity_embedding(np.array([3]), embeddings))
    assert np.isnan(intra_list_diversity_category(np.array([3]), categories))


def test_an_empty_list_is_undefined_everywhere():
    embeddings = _unit(5)
    categories = np.zeros(5, dtype=int)
    empty = np.zeros(0, dtype=np.int32)
    assert np.isnan(intra_list_diversity_embedding(empty, embeddings))
    assert np.isnan(intra_list_diversity_category(empty, categories))
    assert np.isnan(list_novelty(empty, np.ones(5)))


# --------------------------------------------------------------------------
# Novelty
# --------------------------------------------------------------------------


def test_never_clicked_articles_get_finite_novelty_not_infinity():
    """88.2% of MIND's corpus has zero train clicks; -log2(0) is +inf.

    Without smoothing every mean containing one of them would be inf, and
    nothing would raise - the metric would just silently stop working.
    """
    counts = np.array([100.0, 0.0, 0.0, 0.0])
    nov = item_novelty(counts)
    assert np.isfinite(nov).all()
    # and the unclicked ones are the most novel
    assert nov[1] == nov[2] == nov[3] > nov[0]


def test_novelty_is_monotonically_decreasing_in_popularity():
    counts = np.array([0.0, 1.0, 10.0, 1000.0])
    nov = item_novelty(counts)
    assert np.all(np.diff(nov) < 0)


def test_novelty_matches_the_hand_computed_self_information():
    # counts +1 -> [2, 2], total 4, p = 0.5 each -> -log2(0.5) = 1.0
    assert item_novelty(np.array([1.0, 1.0])) == pytest.approx([1.0, 1.0])


def test_list_novelty_is_the_mean_over_the_list():
    nov = np.array([1.0, 2.0, 3.0, 10.0])
    assert list_novelty(np.array([0, 1, 2]), nov) == pytest.approx(2.0)


def test_novelty_scale_is_a_property_of_the_corpus_not_the_list():
    """Two corpora, identical list, different novelty. The reason MIND's and
    EB-NeRD's novelty figures cannot be compared to each other."""
    small = item_novelty(np.array([1.0, 1.0]))
    large = item_novelty(np.concatenate([[1.0, 1.0], np.zeros(1000)]))
    assert large[0] > small[0]


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_coverage_counts_the_union_not_the_sum():
    """Ten lists of the same article cover one article, not ten."""
    lists = [np.array([3, 4])] * 10
    assert coverage(lists, 100) == pytest.approx(0.02)


def test_coverage_of_everything_is_one():
    assert coverage([np.arange(10)], 10) == pytest.approx(1.0)


def test_coverage_of_nothing_is_zero():
    assert coverage([], 10) == pytest.approx(0.0)
    assert coverage([np.zeros(0, dtype=np.int32)], 10) == pytest.approx(0.0)


def test_coverage_rejects_a_nonpositive_catalogue():
    with pytest.raises(ValueError, match="catalogue_size must be positive"):
        coverage([np.array([1])], 0)


def test_coverage_is_not_the_mean_of_per_list_coverages():
    """Pins the reason Q4.4's bootstrap must recompute it inside each resample."""
    lists = [np.array([0]), np.array([1]), np.array([2])]
    assert coverage(lists, 10) == pytest.approx(0.3)
    per_list_mean = np.mean([coverage([lst], 10) for lst in lists])
    assert per_list_mean == pytest.approx(0.1)


# --------------------------------------------------------------------------
# The combined entry point
# --------------------------------------------------------------------------


def test_evaluate_lists_returns_per_list_values_for_the_bootstrap():
    embeddings = _unit(20)
    categories = np.arange(20) % 4
    nov = item_novelty(np.arange(20, dtype=float))
    lists = [np.array([0, 1, 2]), np.array([5, 6]), np.array([9])]
    out = evaluate_lists(lists, embeddings, categories, nov, catalogue_size=20)

    assert out["ild_embedding"].shape == (3,)
    assert np.isnan(out["ild_embedding"][2])  # single-article list
    assert out["n_lists"] == 3
    assert out["coverage"] == pytest.approx(6 / 20)


def test_a_random_list_beats_a_clustered_one_on_both_diversity_bases():
    """The property that makes these metrics readable only against accuracy:
    the *worse* recommender scores better here."""
    embeddings = _unit(60)
    # force a tight cluster: rows 0-9 all near row 0
    embeddings[1:10] = embeddings[0] + 0.01 * embeddings[1:10]
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    categories = np.array([0] * 10 + list(range(1, 51)))

    clustered = np.arange(10)
    scattered = np.arange(10, 20)
    assert intra_list_diversity_embedding(
        scattered, embeddings
    ) > intra_list_diversity_embedding(clustered, embeddings)
    assert intra_list_diversity_category(
        scattered, categories
    ) > intra_list_diversity_category(clustered, categories)
