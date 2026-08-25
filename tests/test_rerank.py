"""Adversarial tests for the Q4.2 re-ranking runner.

The failure modes this module is actually exposed to are alignment ones: a
candidate id mapped to the wrong article row, a user mapped to the wrong query
row, or a batched computation that quietly depends on batch boundaries. None of
those raise - they return plausible numbers computed against the wrong thing.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from newsrec.eval import rerank
from newsrec.retrieval import bm25, bm25_search, semantic_search

ARTICLES = pl.DataFrame(
    {
        "dataset": ["t"] * 6,
        "article_id": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "title": [
            "tariff talks resume",
            "tariff deal signed",
            "storm warning issued",
            "storm damage repairs",
            "election poll results",
            "rådden kørsel på blå plader",
        ],
        "abstract": ["", "", "", "", "", ""],
    }
)

HISTORY = pl.DataFrame(
    {
        "dataset": ["t"] * 3,
        "user_id": ["u_tariff", "u_storm", "u_cold"],
        "history_article_ids": [["a1"], ["a3"], []],
    }
)


def _impressions(rows: list[tuple[str, str, list[str], list[str]]], split: str = "val"):
    return pl.DataFrame(
        {
            "dataset": ["t"] * len(rows),
            "split": [split] * len(rows),
            "impression_id": [r[0] for r in rows],
            "user_id": [r[1] for r in rows],
            "candidate_article_ids": [r[2] for r in rows],
            "clicked_article_ids": [r[3] for r in rows],
        }
    )


def _unit_embeddings(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _setup(impressions):
    article_ids = ARTICLES.get_column("article_id").to_list()
    index = bm25.build_index(ARTICLES)
    title_term = bm25_search.build_title_term_matrix(ARTICLES, index.vocab)
    queries = bm25_search.build_queries(HISTORY, index, title_term)
    candidates = rerank.build_candidate_set(impressions, article_ids, HISTORY)
    return article_ids, index, queries, candidates


# --------------------------------------------------------------------------
# Candidate set construction
# --------------------------------------------------------------------------


def test_candidate_rows_and_labels_align_with_the_platform_order():
    imps = _impressions([("i1", "u_tariff", ["a3", "a1", "a5"], ["a1"])])
    article_ids, _, _, cands = _setup(imps)

    # rows must be the platform's order, not sorted - the tie policy is defined
    # against that order, so sorting here would silently redefine the metrics
    assert cands.candidate_rows[0].tolist() == [2, 0, 4]
    assert cands.labels[0].tolist() == [0, 1, 0]
    assert cands.history_len.tolist() == [1]


def test_a_candidate_missing_from_the_article_matrix_raises():
    imps = _impressions([("i1", "u_tariff", ["a1", "ghost"], ["a1"])])
    with pytest.raises(KeyError, match="absent from the article matrix"):
        _setup(imps)


def test_history_length_is_zero_for_a_user_absent_from_the_history_table():
    imps = _impressions([("i1", "u_unknown", ["a1", "a2"], ["a1"])])
    _, _, _, cands = _setup(imps)
    assert cands.history_len.tolist() == [0]


# --------------------------------------------------------------------------
# BM25 scoring: alignment, batching, cold start
# --------------------------------------------------------------------------


def test_bm25_scores_match_a_direct_per_impression_computation():
    imps = _impressions(
        [
            ("i1", "u_tariff", ["a3", "a2", "a5"], ["a2"]),
            ("i2", "u_storm", ["a1", "a4", "a6"], ["a4"]),
        ]
    )
    _, index, queries, cands = _setup(imps)
    scores = rerank.score_bm25(cands, index, queries)

    row_of_user = {u: i for i, u in enumerate(queries.user_ids)}
    for i, user in enumerate(cands.user_ids):
        q = queries.matrix[row_of_user[user]]
        direct = np.asarray(
            (index.doc_term[cands.candidate_rows[i]] @ q.T).todense()
        ).ravel()
        assert scores[i] == pytest.approx(direct, abs=1e-5)


def test_bm25_ranks_the_topically_matching_candidate_first():
    """A sanity check on the concept, not just the plumbing."""
    imps = _impressions([("i1", "u_tariff", ["a3", "a5", "a2"], ["a2"])])
    _, index, queries, cands = _setup(imps)
    scores = rerank.score_bm25(cands, index, queries)[0]
    # a2 ("tariff deal signed") shares 'tariff' with the query built from a1
    assert scores.argmax() == 2
    assert scores[0] == 0.0 and scores[1] == 0.0  # no shared term at all


def test_batch_size_does_not_change_the_scores():
    """The batching is a memory device and must be invisible in the output.

    A grouping bug - splitting on the wrong index, or slicing `groups` out of
    step with `unique_rows` - produces perfectly plausible scores taken from a
    neighbouring user, and only shows up when the batch boundaries move.
    """
    imps = _impressions(
        [(f"i{i}", u, ["a1", "a2", "a3", "a4"], ["a1"])
         for i, u in enumerate(["u_tariff", "u_storm", "u_tariff", "u_cold", "u_storm"])]
    )
    _, index, queries, cands = _setup(imps)
    one = rerank.score_bm25(cands, index, queries, batch_size=1)
    many = rerank.score_bm25(cands, index, queries, batch_size=64)
    for a, b in zip(one, many):
        assert a == pytest.approx(b)


def test_the_same_user_gets_the_same_scores_across_their_impressions():
    """Scoring once per unique user is only valid if the query is user-fixed."""
    imps = _impressions(
        [
            ("i1", "u_tariff", ["a2", "a3"], ["a2"]),
            ("i2", "u_storm", ["a2", "a3"], ["a3"]),
            ("i3", "u_tariff", ["a2", "a3"], ["a3"]),
        ]
    )
    _, index, queries, cands = _setup(imps)
    scores = rerank.score_bm25(cands, index, queries)
    assert scores[0] == pytest.approx(scores[2])
    assert scores[0] != pytest.approx(scores[1])


def test_cold_start_user_scores_flat_zero_rather_than_nan():
    """NaN would be rejected by the metrics as unrankable; zero is a total tie.

    Under D23's pessimistic policy that ranks the clicked items last, which is
    the honest reading of "the scorer distinguished nothing".
    """
    imps = _impressions([("i1", "u_cold", ["a1", "a2", "a3"], ["a2"])])
    _, index, queries, cands = _setup(imps)
    scores = rerank.score_bm25(cands, index, queries)[0]
    assert np.isfinite(scores).all()
    assert (scores == 0.0).all()


def test_user_absent_from_the_history_table_is_treated_as_cold_start():
    imps = _impressions([("i1", "u_never_seen", ["a1", "a2"], ["a1"])])
    _, index, queries, cands = _setup(imps)
    scores = rerank.score_bm25(cands, index, queries)[0]
    assert (scores == 0.0).all()


def test_scores_do_not_depend_on_impression_order():
    """Grouping by user reorders internally; the output must not notice."""
    rows = [
        ("i1", "u_tariff", ["a1", "a2", "a3"], ["a2"]),
        ("i2", "u_storm", ["a3", "a4", "a5"], ["a4"]),
        ("i3", "u_cold", ["a1", "a5"], ["a5"]),
    ]
    _, index, queries, cands = _setup(_impressions(rows))
    forward = dict(zip(cands.impression_ids, rerank.score_bm25(cands, index, queries)))

    _, index2, queries2, cands2 = _setup(_impressions(rows[::-1]))
    backward = dict(zip(cands2.impression_ids, rerank.score_bm25(cands2, index2, queries2)))

    for key in forward:
        assert forward[key] == pytest.approx(backward[key])


# --------------------------------------------------------------------------
# Semantic scoring
# --------------------------------------------------------------------------


def test_semantic_scores_are_cosines_of_the_pooled_user_vector():
    imps = _impressions([("i1", "u_tariff", ["a3", "a2", "a5"], ["a2"])])
    article_ids, _, _, cands = _setup(imps)
    embeddings = _unit_embeddings(len(article_ids))
    users = semantic_search.build_user_vectors(HISTORY, article_ids, embeddings)

    scores = rerank.score_semantic(cands, users, embeddings)[0]
    row = users.user_ids.index("u_tariff")
    direct = embeddings[cands.candidate_rows[0]] @ users.matrix[row]
    assert scores == pytest.approx(direct, abs=1e-6)
    # a1 is u_tariff's only history article, so the pooled vector IS a1's vector
    assert embeddings[0] @ users.matrix[row] == pytest.approx(1.0, abs=1e-5)


def test_semantic_cold_start_scores_flat_zero_rather_than_nan():
    """A mean over zero vectors is the Phase 3 NaN trap; it must not resurface."""
    imps = _impressions([("i1", "u_cold", ["a1", "a2", "a3"], ["a2"])])
    article_ids, _, _, cands = _setup(imps)
    embeddings = _unit_embeddings(len(article_ids))
    users = semantic_search.build_user_vectors(HISTORY, article_ids, embeddings)
    scores = rerank.score_semantic(cands, users, embeddings)[0]
    assert np.isfinite(scores).all()
    assert (scores == 0.0).all()


def test_semantic_batch_size_does_not_change_the_scores():
    imps = _impressions(
        [(f"i{i}", u, ["a1", "a2", "a3", "a4"], ["a1"])
         for i, u in enumerate(["u_tariff", "u_storm", "u_tariff", "u_cold"])]
    )
    article_ids, _, _, cands = _setup(imps)
    embeddings = _unit_embeddings(len(article_ids))
    users = semantic_search.build_user_vectors(HISTORY, article_ids, embeddings)
    one = rerank.score_semantic(cands, users, embeddings, batch_size=1)
    many = rerank.score_semantic(cands, users, embeddings, batch_size=32)
    for a, b in zip(one, many):
        assert a == pytest.approx(b)


# --------------------------------------------------------------------------
# Popularity baseline - the one with a live leakage risk
# --------------------------------------------------------------------------


def test_popularity_refuses_to_count_anything_but_train():
    """Counting val clicks would mean the baseline had seen its own answers."""
    val = _impressions([("i1", "u_tariff", ["a1", "a2"], ["a1"])], split="val")
    with pytest.raises(ValueError, match="train only"):
        rerank.train_click_counts(val, ARTICLES.get_column("article_id").to_list())


def test_popularity_refuses_a_mixed_split_frame():
    mixed = pl.concat(
        [
            _impressions([("i1", "u_tariff", ["a1"], ["a1"])], split="train"),
            _impressions([("i2", "u_storm", ["a2"], ["a2"])], split="val"),
        ]
    )
    with pytest.raises(ValueError, match="train only"):
        rerank.train_click_counts(mixed, ARTICLES.get_column("article_id").to_list())


def test_popularity_counts_every_click_including_multi_click_impressions():
    train = _impressions(
        [
            ("t1", "u_tariff", ["a1", "a2"], ["a1", "a2"]),
            ("t2", "u_storm", ["a1", "a3"], ["a1"]),
        ],
        split="train",
    )
    counts = rerank.train_click_counts(train, ARTICLES.get_column("article_id").to_list())
    assert counts.tolist() == [2.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_popularity_scores_are_gathered_in_candidate_order():
    imps = _impressions([("i1", "u_tariff", ["a3", "a1", "a2"], ["a1"])])
    article_ids, _, _, cands = _setup(imps)
    counts = np.array([10.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert rerank.score_popularity(cands, counts)[0].tolist() == [0.0, 10.0, 5.0]


def test_popularity_scores_are_a_copy_not_a_view():
    """A view would let a later in-place edit rewrite history silently."""
    imps = _impressions([("i1", "u_tariff", ["a1", "a2"], ["a1"])])
    article_ids, _, _, cands = _setup(imps)
    counts = np.array([10.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scores = rerank.score_popularity(cands, counts)[0]
    scores[0] = -1.0
    assert counts[0] == 10.0


# --------------------------------------------------------------------------
# Random baseline
# --------------------------------------------------------------------------


def test_random_scores_are_reproducible_and_order_independent():
    """Seeded per impression id, not drawn from one stream in row order.

    A single shared generator would make impression i3's scores depend on how
    many impressions preceded it - so evaluating a slice would silently produce
    different numbers from evaluating the whole set.
    """
    rows = [
        ("i1", "u_tariff", ["a1", "a2", "a3"], ["a2"]),
        ("i2", "u_storm", ["a3", "a4"], ["a4"]),
        ("i3", "u_cold", ["a1", "a5", "a6"], ["a5"]),
    ]
    _, _, _, full = _setup(_impressions(rows))
    _, _, _, subset = _setup(_impressions(rows[2:]))

    full_scores = dict(zip(full.impression_ids, rerank.score_random(full)))
    subset_scores = dict(zip(subset.impression_ids, rerank.score_random(subset)))
    assert full_scores["i3"] == pytest.approx(subset_scores["i3"])

    # and stable across calls
    assert full_scores["i1"] == pytest.approx(
        dict(zip(full.impression_ids, rerank.score_random(full)))["i1"]
    )


def test_random_scores_differ_between_impressions():
    rows = [
        ("i1", "u_tariff", ["a1", "a2", "a3"], ["a2"]),
        ("i2", "u_tariff", ["a1", "a2", "a3"], ["a2"]),
    ]
    _, _, _, cands = _setup(_impressions(rows))
    scores = rerank.score_random(cands)
    assert not np.allclose(scores[0], scores[1])


def test_random_scores_have_the_right_length_per_impression():
    rows = [
        ("i1", "u_tariff", ["a1", "a2", "a3"], ["a2"]),
        ("i2", "u_storm", ["a3", "a4"], ["a4"]),
    ]
    _, _, _, cands = _setup(_impressions(rows))
    assert [len(s) for s in rerank.score_random(cands)] == [3, 2]
