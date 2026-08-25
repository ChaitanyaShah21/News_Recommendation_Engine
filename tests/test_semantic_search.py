"""R10 adversarial tests for semantic_search.py (Q3.2, Q3.3).

Constructed geometry throughout. The tests that matter here are built so that
the *wrong* implementation visibly fails: masking with 0.0 instead of -inf, or
dividing by a zero norm, both produce plausible-looking output on real data and
would never be caught by eyeballing a recall number.
"""

import numpy as np
import polars as pl

from newsrec.retrieval.semantic_search import (
    build_user_vectors,
    retrieve,
    retrieve_bucketed,
)

DIM = 384


def unit(*components) -> np.ndarray:
    """A unit vector whose first few coordinates are given, rest zero."""
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(components)] = components
    return v / np.linalg.norm(v)


# Four articles at known angles, so every cosine below is exact and hand-checkable:
#   a0 at 0 deg, a1 at 90 deg (cos 0 vs a0), a2 at 180 deg (cos -1), a3 at 45 deg (cos .707)
ARTICLE_IDS = ["a0", "a1", "a2", "a3"]
EMB = np.vstack([unit(1, 0), unit(0, 1), unit(-1, 0), unit(1, 1)]).astype(np.float32)


def history_frame(rows):
    return pl.DataFrame(
        rows,
        schema={"user_id": pl.String, "history_article_ids": pl.List(pl.String)},
        orient="row",
    )


# ------------------------------------------------------- mean pooling (Q3.3)

def test_mean_pool_of_one_article_is_that_article():
    uv = build_user_vectors(history_frame([("u1", ["a0"])]), ARTICLE_IDS, EMB)
    np.testing.assert_allclose(uv.matrix[0], EMB[0], atol=1e-6)
    assert uv.has_query[0]


def test_mean_pool_is_the_normalised_average():
    """a0 at 0 deg and a1 at 90 deg average to 45 deg - which is a3's direction."""
    uv = build_user_vectors(history_frame([("u1", ["a0", "a1"])]), ARTICLE_IDS, EMB)
    np.testing.assert_allclose(uv.matrix[0], EMB[3], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(uv.matrix[0]), 1.0, atol=1e-6)


def test_repeated_article_is_not_double_weighted():
    once = build_user_vectors(history_frame([("u", ["a0", "a1"])]), ARTICLE_IDS, EMB)
    twice = build_user_vectors(
        history_frame([("u", ["a0", "a0", "a1"])]), ARTICLE_IDS, EMB
    )
    np.testing.assert_allclose(once.matrix[0], twice.matrix[0], atol=1e-6)


def test_only_the_last_n_recent_are_used():
    uv = build_user_vectors(
        history_frame([("u", ["a2", "a0", "a1"])]), ARTICLE_IDS, EMB, n_recent=2
    )
    np.testing.assert_allclose(uv.matrix[0], EMB[3], atol=1e-6)  # mean(a0, a1) == a3


def test_unknown_article_ids_are_skipped_not_fatal():
    uv = build_user_vectors(
        history_frame([("u", ["not-in-store", "a0"])]), ARTICLE_IDS, EMB
    )
    np.testing.assert_allclose(uv.matrix[0], EMB[0], atol=1e-6)
    assert uv.has_query[0]


# ------------------------------------------------- the NaN hazard (D17 redux)

def test_cold_start_user_produces_zeros_not_nan():
    """BM25 gave an empty query an all-zero score vector - harmless and obvious.

    Mean pooling over zero vectors is a division by zero. NaN does not score 0;
    it propagates through argsort and yields an arbitrary ranking that looks
    entirely legitimate.
    """
    uv = build_user_vectors(history_frame([("cold", [])]), ARTICLE_IDS, EMB)
    assert not uv.has_query[0]
    assert np.isfinite(uv.matrix).all()
    assert np.all(uv.matrix[0] == 0.0)


def test_null_history_produces_zeros_not_nan():
    uv = build_user_vectors(history_frame([("cold", None)]), ARTICLE_IDS, EMB)
    assert not uv.has_query[0]
    assert np.isfinite(uv.matrix).all()


def test_history_entirely_absent_from_store_is_query_less():
    uv = build_user_vectors(history_frame([("u", ["ghost1", "ghost2"])]), ARTICLE_IDS, EMB)
    assert not uv.has_query[0]
    assert np.isfinite(uv.matrix).all()


def test_exactly_cancelling_history_does_not_produce_nan():
    """a0 at 0 deg and a2 at 180 deg sum to the zero vector: norm 0, no direction.

    Vanishingly unlikely on real data, which is exactly why it would never be
    found by testing against the real corpus.
    """
    uv = build_user_vectors(history_frame([("u", ["a0", "a2"])]), ARTICLE_IDS, EMB)
    assert not uv.has_query[0]
    assert np.isfinite(uv.matrix).all()


def test_nearly_cancelling_history_is_query_less_not_float_noise():
    """The hole a `norms > 0` check misses, found by mutation-testing the module.

    Two nearly-opposite unit vectors average to norm ~5e-08. That is greater
    than zero, so a zero-check treats the user as having a real query - and
    normalising inflates pure floating-point residue back to unit length. The
    user then retrieves 200 articles for a direction that means nothing.
    """
    ids = ["a0", "opposite"]
    almost_opposite = unit(-1.0, 1e-7)
    emb = np.vstack([unit(1, 0), almost_opposite]).astype(np.float32)

    uv = build_user_vectors(history_frame([("u", ["a0", "opposite"])]), ids, emb)
    assert not uv.has_query[0], "near-cancelling history treated as a real query"
    assert np.isfinite(uv.matrix).all()
    assert np.all(uv.matrix[0] == 0.0)


def test_genuinely_disagreeing_history_still_counts_as_a_query():
    """The threshold must not swallow real users: perpendicular is not cancelling.

    Two perpendicular unit vectors average to norm 1/sqrt(2) = 0.707 - five
    million times MIN_NORM. A reader of both football and politics has a weak
    direction, not an absent one, and must still get results.
    """
    uv = build_user_vectors(history_frame([("u", ["a0", "a1"])]), ARTICLE_IDS, EMB)
    assert uv.has_query[0]


def test_query_less_user_retrieves_nothing_rather_than_an_arbitrary_list():
    uv = build_user_vectors(
        history_frame([("cold", []), ("warm", ["a0"])]), ARTICLE_IDS, EMB
    )
    out = retrieve(uv, EMB, k=3)
    assert len(out[0]) == 0
    assert len(out[1]) == 3


# ------------------------------------- masking with -inf, not 0.0 (the big one)

def test_excluded_article_cannot_outrank_a_negative_scoring_one():
    """D15 exclusion must use -inf. With BM25's 0.0 convention this test fails.

    User vector is a0. Scores: a0=1.0, a1=0.0, a2=-1.0, a3=0.707.
    Excluding a0 by setting it to 0.0 would leave it tied with a1 at zero and
    ranked ABOVE a2 - so a0, an article we explicitly removed, comes back in the
    results. With -inf it is gone.
    """
    uv = build_user_vectors(history_frame([("u", ["a0"])]), ARTICLE_IDS, EMB)
    out = retrieve(uv, EMB, k=3, exclude_rows=[np.array([0], dtype=np.int32)])

    assert 0 not in out[0], "excluded article resurfaced - masked with 0.0 not -inf"
    assert list(out[0]) == [3, 1, 2]  # 0.707, 0.0, -1.0


def test_unavailable_article_cannot_outrank_an_available_negative_one():
    """D19's availability mask, same failure mode and a worse blast radius.

    EB-NeRD's available pool is ~2,963 of 11,777 articles, so a multiplicative
    0/1 mask would float ~8,814 masked articles to 0.0, above every available
    article scoring negative.
    """
    uv = build_user_vectors(history_frame([("u", ["a0"])]), ARTICLE_IDS, EMB)
    allowed = np.array([True, True, True, False])  # a3, the best match, is unavailable

    out = retrieve_bucketed(
        uv, EMB,
        task_query_row=np.array([0]),
        task_bucket=np.array([0]),
        bucket_allowed=[allowed],
        k=3,
    )
    assert 3 not in out[0], "unavailable article retrieved - mask floated it to 0.0"
    assert 2 in out[0], "available negative-scoring article was displaced"
    assert list(out[0]) == [0, 1, 2]


def test_negative_cosines_are_kept_not_filtered_out():
    """BM25's _top_k drops non-positive scores; here that would discard real rankings."""
    uv = build_user_vectors(history_frame([("u", ["a0"])]), ARTICLE_IDS, EMB)
    out = retrieve(uv, EMB, k=4)
    assert list(out[0]) == [0, 3, 1, 2]  # includes a1 at 0.0 and a2 at -1.0


def test_bucket_with_nothing_available_returns_empty_not_a_crash():
    uv = build_user_vectors(history_frame([("u", ["a0"])]), ARTICLE_IDS, EMB)
    out = retrieve_bucketed(
        uv, EMB,
        task_query_row=np.array([0]),
        task_bucket=np.array([0]),
        bucket_allowed=[np.zeros(4, dtype=bool)],
        k=3,
    )
    assert len(out[0]) == 0


# ------------------------------------------------------------------ mechanics

def test_k_larger_than_corpus_returns_everything_once():
    uv = build_user_vectors(history_frame([("u", ["a0"])]), ARTICLE_IDS, EMB)
    out = retrieve(uv, EMB, k=99)
    assert sorted(out[0]) == [0, 1, 2, 3]


def test_batch_size_does_not_change_results():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((60, DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = [f"a{i}" for i in range(60)]
    hist = history_frame([(f"u{i}", [f"a{i}", f"a{(i * 7) % 60}"]) for i in range(40)])
    uv = build_user_vectors(hist, ids, emb)

    small = retrieve(uv, emb, k=10, batch_size=3)
    large = retrieve(uv, emb, k=10, batch_size=256)
    for a, b in zip(small, large):
        np.testing.assert_array_equal(a, b)


def test_bucketed_matches_unbucketed_when_everything_is_available():
    """A mask of all-True must reduce to the plain path exactly."""
    uv = build_user_vectors(
        history_frame([("u1", ["a0"]), ("u2", ["a1"])]), ARTICLE_IDS, EMB
    )
    plain = retrieve(uv, EMB, k=4)
    bucketed = retrieve_bucketed(
        uv, EMB,
        task_query_row=np.array([0, 1]),
        task_bucket=np.array([0, 0]),
        bucket_allowed=[np.ones(4, dtype=bool)],
        k=4,
    )
    for a, b in zip(plain, bucketed):
        np.testing.assert_array_equal(a, b)


def test_ties_break_deterministically():
    """Two identical article vectors must resolve the same way on every run."""
    emb = np.vstack([unit(1, 0), unit(0, 1), unit(0, 1)]).astype(np.float32)
    uv = build_user_vectors(history_frame([("u", ["a0"])]), ["a0", "a1", "a2"], emb)
    first = retrieve(uv, emb, k=3)[0]
    for _ in range(5):
        np.testing.assert_array_equal(retrieve(uv, emb, k=3)[0], first)


def test_user_vectors_are_unit_length_where_a_query_exists():
    """`scores = U @ M.T` is only cosine similarity if both sides are normalised."""
    uv = build_user_vectors(
        history_frame([("u1", ["a0", "a1"]), ("u2", ["a3"]), ("cold", [])]),
        ARTICLE_IDS, EMB,
    )
    norms = np.linalg.norm(uv.matrix, axis=1)
    np.testing.assert_allclose(norms[uv.has_query], 1.0, atol=1e-6)
    assert np.all(norms[~uv.has_query] == 0.0)
