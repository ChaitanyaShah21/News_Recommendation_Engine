"""Adversarial tests for the BM25 query side and recall@K (R10).

The batching tests matter most. Batching is invisible in the output - a bug
there produces plausible-looking numbers, not a crash - so several tests assert
that results are *identical* regardless of batch size, and that the bucketed
path agrees with the plain one when nothing is actually masked.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from newsrec.eval.recall import recall_at_k
from newsrec.retrieval import bm25, bm25_search


def _articles(rows, dataset="mind"):
    """rows: (article_id, title, abstract)."""
    return pl.DataFrame(
        {
            "dataset": [dataset] * len(rows),
            "article_id": [r[0] for r in rows],
            "title": [r[1] for r in rows],
            "abstract": [r[2] for r in rows],
        },
        schema={
            "dataset": pl.Utf8,
            "article_id": pl.Utf8,
            "title": pl.Utf8,
            "abstract": pl.Utf8,
        },
    )


def _history(rows, dataset="mind"):
    """rows: (user_id, [article_id, ...])."""
    return pl.DataFrame(
        {
            "dataset": [dataset] * len(rows),
            "user_id": [r[0] for r in rows],
            "history_article_ids": [r[1] for r in rows],
        },
        schema={
            "dataset": pl.Utf8,
            "user_id": pl.Utf8,
            "history_article_ids": pl.List(pl.Utf8),
        },
    )


def _setup(article_rows, history_rows, **kwargs):
    articles = _articles(article_rows)
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    queries = bm25_search.build_queries(
        _history(history_rows), index, title_term, **kwargs
    )
    return index, title_term, queries


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------


def test_query_uses_titles_only_not_abstracts():
    """D12 says the query is built from titles; Q2.1 says the index covers
    titles AND abstracts. Mixing them up would silently make queries ~4x longer
    and reintroduce the topic drift D12 exists to prevent.
    """
    articles = _articles([("mind:A", "budget", "abstractword election")])
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)

    assert "abstractword" in index.vocab  # the index does cover abstracts
    column = index.vocab["abstractword"]
    assert title_term[0, column] == 0.0  # but the query side does not


def test_query_uses_the_LAST_n_articles_not_the_first():
    """History is chronological oldest-first (verified: 0 of 4,714 EB-NeRD users
    have out-of-order timestamps). Slicing from the wrong end would build every
    query from the user's *oldest* interests - working code, inverted meaning,
    and no error.
    """
    articles = _articles([(f"mind:{i}", f"topic{i}", None) for i in range(15)])
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    history = _history([("u1", [f"mind:{i}" for i in range(15)])])

    queries = bm25_search.build_queries(history, index, title_term, n_recent=3)
    present = {
        term for term, col in index.vocab.items() if queries.matrix[0, col] > 0
    }
    assert present == {"topic12", "topic13", "topic14"}
    assert "topic0" not in present


def test_cold_start_user_is_flagged_not_silently_scored():
    """1,407 MIND val users have empty history. They must be detectable (D17),
    not quietly produce an all-zero query that ranks arbitrary articles.
    """
    _, _, queries = _setup(
        [("mind:A", "budget", None)],
        [("cold", []), ("warm", ["mind:A"])],
    )
    assert list(queries.has_query) == [False, True]
    assert queries.matrix[0].nnz == 0


def test_user_whose_titles_tokenise_to_nothing_is_also_flagged():
    """Emptiness must be judged from the query matrix, not from history length -
    a user with history whose titles are pure punctuation is just as query-less
    as a cold-start user, and a `len(history) > 0` test would miss them.
    """
    _, _, queries = _setup(
        [("mind:A", "!!! ???", None), ("mind:B", "budget", None)],
        [("punct", ["mind:A"]), ("real", ["mind:B"])],
    )
    assert list(queries.has_query) == [False, True]


def test_repeated_article_in_history_counts_its_title_once():
    """Real EB-NeRD histories repeat the same article (observed: one user's last
    5 clicks contained the same headline 3 times). Counting it three times would
    triple that headline's weight in the query for no evidential reason.
    """
    _, _, once = _setup(
        [("mind:A", "budget vote", None), ("mind:B", "other", None)],
        [("u", ["mind:A"])],
    )
    _, _, thrice = _setup(
        [("mind:A", "budget vote", None), ("mind:B", "other", None)],
        [("u", ["mind:A", "mind:A", "mind:A"])],
    )
    assert np.allclose(once.matrix.toarray(), thrice.matrix.toarray())


def test_unknown_article_in_history_is_skipped_not_fatal():
    """Verified 100% of history ids resolve today, but the large bundles are a
    different crawl - a missing id must drop out, not raise KeyError mid-run.
    """
    _, _, queries = _setup(
        [("mind:A", "budget", None)],
        [("u", ["mind:A", "mind:GHOST"])],
    )
    assert queries.has_query[0]


def test_binary_variant_keeps_support_but_flattens_counts():
    """D16's ablation. Same terms present, all weights 1."""
    articles = _articles(
        [("mind:A", "budget budget vote", None), ("mind:B", "budget other", None)]
    )
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    history = _history([("u", ["mind:A", "mind:B"])])

    raw = bm25_search.build_queries(history, index, title_term, binary=False)
    binary = bm25_search.build_queries(history, index, title_term, binary=True)

    assert raw.matrix[0, index.vocab["budget"]] == 3.0
    assert binary.matrix[0, index.vocab["budget"]] == 1.0
    assert set(raw.matrix.indices) == set(binary.matrix.indices)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def test_retrieval_excludes_the_users_own_history():
    """D15. The query is built FROM these articles' titles, so without the
    exclusion they self-match and occupy the top of every list.
    """
    index, title_term, queries = _setup(
        [
            ("mind:A", "budget vote", None),
            ("mind:B", "budget vote", None),
            ("mind:C", "unrelated", None),
        ],
        [("u", ["mind:A"])],
    )
    without = bm25_search.retrieve(index, queries, k=3)
    with_exclusion = bm25_search.retrieve(
        index, queries, k=3, exclude_rows=[np.array([0], dtype=np.int32)]
    )
    assert 0 in without[0]
    assert 0 not in with_exclusion[0]


def test_zero_scoring_articles_are_never_returned():
    """Padding a short list with zero-scored articles would let a top-200 list
    collect ground-truth hits by chance - a real risk at K=200 over a corpus
    where only 853 articles are ever clicked.
    """
    index, _, queries = _setup(
        [("mind:A", "budget", None)] + [(f"mind:{i}", "zzz", None) for i in range(9)],
        [("u", ["mind:A"])],
    )
    got = bm25_search.retrieve(index, queries, k=10)
    assert len(got[0]) == 1  # only the one article sharing a term
    assert got[0][0] == 0


def test_results_are_identical_regardless_of_batch_size():
    """The batching bug that would never announce itself. Batching exists only
    because the dense score matrix is 9.9 GB for MIND val (SCALE_NOTES.md); it
    must not change a single result.
    """
    articles = _articles(
        [(f"mind:{i}", f"topic{i % 7} shared", None) for i in range(40)]
    )
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    history = _history([(f"u{i}", [f"mind:{i}"]) for i in range(20)])
    queries = bm25_search.build_queries(history, index, title_term)

    one = bm25_search.retrieve(index, queries, k=5, batch_size=1)
    seven = bm25_search.retrieve(index, queries, k=5, batch_size=7)
    huge = bm25_search.retrieve(index, queries, k=5, batch_size=1000)
    for a, b, c in zip(one, seven, huge):
        assert np.array_equal(a, b)
        assert np.array_equal(a, c)


def test_retrieval_is_deterministic_across_runs():
    """Ties are common when many articles share one rare term. Unstable
    tie-breaking would make a reported recall@K unreproducible.
    """
    index, title_term, queries = _setup(
        [(f"mind:{i}", "identical text", None) for i in range(20)]
        + [("mind:q", "identical", None)],
        [("u", ["mind:q"])],
    )
    first = bm25_search.retrieve(index, queries, k=5)
    second = bm25_search.retrieve(index, queries, k=5)
    assert np.array_equal(first[0], second[0])


# --------------------------------------------------------------------------
# Availability / time bucketing (D19)
# --------------------------------------------------------------------------


def test_bucketed_retrieval_matches_plain_when_everything_is_available():
    """The two code paths must agree where they should agree, or any difference
    later cannot be attributed to the availability mask itself.
    """
    articles = _articles([(f"mind:{i}", f"topic{i % 5} shared", None) for i in range(20)])
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    history = _history([(f"u{i}", [f"mind:{i}"]) for i in range(10)])
    queries = bm25_search.build_queries(history, index, title_term)

    plain = bm25_search.retrieve(index, queries, k=5)
    allow_all = [np.ones(index.n_docs, dtype=np.float32)]
    bucketed = bm25_search.retrieve_bucketed(
        index,
        queries,
        task_query_row=np.arange(10),
        task_bucket=np.zeros(10, dtype=np.int64),
        bucket_allowed=allow_all,
        k=5,
    )
    for a, b in zip(plain, bucketed):
        assert np.array_equal(a, b)


def test_availability_mask_blocks_unavailable_articles():
    index, title_term, queries = _setup(
        [
            ("mind:A", "budget vote", None),
            ("mind:B", "budget vote", None),
            ("mind:C", "budget", None),
        ],
        [("u", ["mind:A"])],
    )
    blocked = np.ones(index.n_docs, dtype=np.float32)
    blocked[1] = 0.0  # article B not yet in circulation
    got = bm25_search.retrieve_bucketed(
        index,
        queries,
        task_query_row=np.array([0]),
        task_bucket=np.array([0]),
        bucket_allowed=[blocked],
        k=3,
    )
    assert 1 not in got[0]
    assert 2 in got[0]


def test_same_user_in_two_buckets_gets_two_different_results():
    """The whole reason bucketing exists: the query is fixed per user but the
    candidate pool is not, so one retrieval per user would be wrong.
    """
    index, title_term, queries = _setup(
        [("mind:A", "budget", None), ("mind:B", "budget", None)],
        [("u", ["mind:A"])],
    )
    early = np.array([1.0, 0.0], dtype=np.float32)  # B not yet published
    late = np.array([1.0, 1.0], dtype=np.float32)
    got = bm25_search.retrieve_bucketed(
        index,
        queries,
        task_query_row=np.array([0, 0]),
        task_bucket=np.array([0, 1]),
        bucket_allowed=[early, late],
        k=2,
    )
    assert 1 not in got[0]
    assert 1 in got[1]


def test_first_seen_times_takes_the_minimum():
    impressions = pl.DataFrame(
        {
            "candidate_article_ids": [["a", "b"], ["a"], ["b"]],
            "timestamp": [
                datetime(2023, 5, 2, 10),
                datetime(2023, 5, 1, 10),
                datetime(2023, 5, 3, 10),
            ],
        }
    )
    out = bm25_search.first_seen_times(impressions)
    seen = dict(
        zip(out.get_column("article_id").to_list(), out.get_column("first_seen").to_list())
    )
    assert seen["a"].day == 1  # earliest appearance, not the first row seen
    assert seen["b"].day == 2


# --------------------------------------------------------------------------
# recall@K
# --------------------------------------------------------------------------


def test_macro_and_micro_differ_when_click_counts_differ():
    """D18's whole point. MIND has 29% multi-click impressions, so the two
    averages genuinely disagree there and must not be conflated.

    Impression 1: 1 click, retrieved  -> recall 1.0
    Impression 2: 4 clicks, 1 retrieved -> recall 0.25
    macro = (1.0 + 0.25)/2 = 0.625 ; micro = (1+1)/(1+4) = 0.4
    """
    clicked = [["x"], ["a", "b", "c", "d"]]
    retrieved = [["x"], ["a", "z"]]
    result = recall_at_k(clicked, retrieved, k=10)
    assert result.macro == pytest.approx(0.625)
    assert result.micro == pytest.approx(0.4)


def test_k_truncates_the_retrieved_list():
    """One retrieval at k=200 is sliced for k=50/100. If slicing were ignored,
    every K would report the same number.
    """
    clicked = [["c"]]
    retrieved = [["a", "b", "c"]]
    assert recall_at_k(clicked, retrieved, k=2).macro == 0.0
    assert recall_at_k(clicked, retrieved, k=3).macro == 1.0


def test_impressions_with_no_clicks_are_skipped_not_counted_as_zero():
    """Their recall is 0/0. Counting them as zero would silently deflate every
    number on any split that contains them.
    """
    result = recall_at_k([["a"], []], [["a"], ["a"]], k=5)
    assert result.macro == pytest.approx(1.0)
    assert result.n_impressions == 1


def test_misaligned_inputs_raise():
    """Retrieved lists are joined back to impressions by position. A silent
    off-by-one would score every impression against another user's results.
    """
    with pytest.raises(ValueError, match="aligned"):
        recall_at_k([["a"], ["b"]], [["a"]], k=5)


def test_duplicate_retrieved_ids_cannot_inflate_recall():
    """Hits are counted per ground-truth article, not per retrieved slot."""
    result = recall_at_k([["a"]], [["a", "a", "a"]], k=5)
    assert result.macro == pytest.approx(1.0)
    assert result.micro == pytest.approx(1.0)
