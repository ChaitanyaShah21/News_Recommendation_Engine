"""BM25 retrieval - the query side (Q2.2, Q2.3).

`bm25.py` turned articles into an index. This module turns *users* into queries
and runs them against that index.

The shape of the computation, and why it is all sparse matrices:

    S  (users x articles)   1 where article is among the user's last N clicks
    T  (articles x terms)   raw token counts of each article's TITLE
    Q = S @ T               (users x terms) query term frequencies
    scores = Q @ W.T        (users x articles) BM25 scores, W = index.doc_term

The first product is the whole of D12 ("concatenate the titles of the last N
clicked articles"): selecting rows of T and summing them *is* concatenating
those titles, because a bag of words has no order to lose. The second product
is BM25 itself - W already holds the document-side weights with IDF folded in,
so multiplying by query term frequency and summing over terms is exactly the
formula, evaluated for every user against every article at once.

Note T is built from titles only, while the index W was built from title +
abstract. That asymmetry is deliberate: Q2.1 says index titles and abstracts,
D12 says query from titles. They are different sides of the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

from newsrec.retrieval.bm25 import BM25Index, tokenize

DEFAULT_N_RECENT = 10
DEFAULT_BATCH_SIZE = 256


@dataclass
class QuerySet:
    """User queries, aligned row-for-row with `user_ids`.

    Attributes:
        user_ids: row i of `matrix` belongs to `user_ids[i]`.
        matrix: (n_users x n_terms) query term frequencies.
        has_query: has_query[i] is False when the user produced no tokens at
            all - a cold-start user with empty history, or one whose recent
            titles tokenised to nothing. D17 excludes these from the headline
            recall and reports them separately, so they are flagged rather
            than silently dropped or silently scored.
    """

    user_ids: list[str]
    matrix: sparse.csr_matrix
    has_query: np.ndarray


def build_title_term_matrix(
    articles: pl.DataFrame, vocab: dict[str, int]
) -> sparse.csr_matrix:
    """Raw token counts of each article's TITLE, in the index's vocabulary.

    Rows are in the same order as `articles`, which must be the same frame the
    index was built from - row i here and row i of the index must be the same
    article, or every query is built from the wrong headlines.

    Terms in a title that never appear in any title+abstract are impossible
    (the index is built from a superset of this text), but terms dropped by
    `max_df` are skipped here too, so the two stay consistent.
    """
    titles = articles.get_column("title").to_list()

    indptr = np.zeros(len(titles) + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[float] = []
    for i, title in enumerate(titles):
        counts: dict[int, int] = {}
        for token in tokenize(title):
            column = vocab.get(token)
            if column is not None:
                counts[column] = counts.get(column, 0) + 1
        for column, count in counts.items():
            indices.append(column)
            data.append(count)
        indptr[i + 1] = len(indices)

    matrix = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            indptr,
        ),
        shape=(len(titles), len(vocab)),
        dtype=np.float32,
    )
    matrix.sort_indices()
    return matrix


def build_selection_matrix(
    recent_rows: list[np.ndarray], n_articles: int
) -> sparse.csr_matrix:
    """(n_users x n_articles) matrix with 1 at each user's recent articles.

    Deliberately holds 1 per (user, article) pair rather than a count: a user
    who clicked the same article twice in their last N would otherwise have its
    title counted twice in their query. `np.unique` upstream guarantees this,
    but constructing with explicit ones means a duplicate slipping through
    still cannot double-weight a headline.
    """
    indptr = np.zeros(len(recent_rows) + 1, dtype=np.int64)
    for i, rows in enumerate(recent_rows):
        indptr[i + 1] = indptr[i] + len(rows)
    indices = (
        np.concatenate(recent_rows).astype(np.int32)
        if recent_rows
        else np.zeros(0, dtype=np.int32)
    )
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix(
        (data, indices, indptr), shape=(len(recent_rows), n_articles), dtype=np.float32
    )


def build_queries(
    history: pl.DataFrame,
    index: BM25Index,
    title_term: sparse.csr_matrix,
    n_recent: int = DEFAULT_N_RECENT,
    binary: bool = False,
) -> QuerySet:
    """Build one query per user from the titles of their last `n_recent` clicks.

    Args:
        history: rows of the unified `history` table for one dataset and split.
        index: the article index these queries will be run against.
        title_term: output of `build_title_term_matrix` for the same articles.
        n_recent: D12's N. Taken from the *end* of the history list, which is
            chronological oldest-first (verified for EB-NeRD: 0 of 4,714 users
            have out-of-order `history_timestamps`; MIND has no timestamps at
            all, so its ordering rests on the dataset documentation).
        binary: D16's ablation. False (default) counts a term once per
            occurrence across the recent titles; True counts each distinct term
            once regardless of how many of those titles contained it.
    """
    row_of_article = {aid: i for i, aid in enumerate(index.article_ids)}

    user_ids = history.get_column("user_id").to_list()
    histories = history.get_column("history_article_ids").to_list()

    recent_rows: list[np.ndarray] = []
    for article_ids in histories:
        # `or []` guards the null case as well as the empty-list case: a
        # cold-start user is legitimately empty, and `.fill_null([])` upstream
        # is one refactor away from being removed (it already caught us once -
        # see PROGRESS.md's null-propagation entry).
        tail = (article_ids or [])[-n_recent:]
        rows = [row_of_article[a] for a in tail if a in row_of_article]
        # unique: the same article clicked twice must not weigh its title twice
        recent_rows.append(np.unique(np.asarray(rows, dtype=np.int32)))

    selection = build_selection_matrix(recent_rows, title_term.shape[0])
    matrix = (selection @ title_term).tocsr()

    if binary:
        matrix = matrix.copy()
        matrix.data[:] = 1.0

    # A user with history whose titles tokenised to nothing is as query-less as
    # a cold-start user, so detect emptiness from the matrix, not from history
    # length - the two are not the same test.
    has_query = np.diff(matrix.indptr) > 0

    return QuerySet(user_ids=user_ids, matrix=matrix, has_query=has_query)


def _top_k(row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest strictly-positive entries of `row`, best first.

    Returns fewer than k when fewer than k entries are positive. A zero-scored
    article shares no term with the query, so returning one would not be a BM25
    retrieval at all - padding a top-200 list with them would let it collect
    hits by chance.
    """
    nonzero = np.flatnonzero(row)
    if nonzero.size == 0:
        return np.zeros(0, dtype=np.int32)
    if nonzero.size > k:
        # argpartition finds the k largest without sorting the other 65,000 -
        # O(n) instead of O(n log n). It does not order those k among
        # themselves, hence the argsort on the small slice.
        top = nonzero[np.argpartition(-row[nonzero], k)[:k]]
    else:
        top = nonzero
    # kind="stable" so equal scores break ties by article row order, the same
    # way on every run. numpy's default quicksort does not promise that, and
    # unstable tie-breaking would make a reported recall@K irreproducible from
    # identical inputs.
    return top[np.argsort(-row[top], kind="stable")].astype(np.int32)


def retrieve_bucketed(
    index: BM25Index,
    queries: QuerySet,
    task_query_row: np.ndarray,
    task_bucket: np.ndarray,
    bucket_allowed: list[np.ndarray],
    k: int,
    exclude_rows: list[np.ndarray] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[np.ndarray]:
    """Top-k per *task*, where a task is one (user, time-bucket) pair (D19).

    Plain `retrieve` scores each user once, because a user's query is fixed
    within a split. Availability is not: an article in circulation at 6pm was
    not in circulation at 6am, so the candidate pool depends on *when* the
    impression happened, and the same user must be re-ranked per time bucket.

    Args:
        task_query_row: for each task, which row of `queries.matrix` to use.
        task_bucket: for each task, which bucket it falls in.
        bucket_allowed: per bucket, a float32 0/1 mask over article rows -
            1 where the article had already appeared in the impression log
            before that bucket began. Multiplying is faster than index
            assignment here and cannot accidentally leave a stale value behind.
        exclude_rows: indexed by *query row*, not by task (D15 exclusion is a
            property of the user, not of the moment).
    """
    weights_t = index.doc_term.T.tocsc()
    results: list[np.ndarray] = [np.zeros(0, dtype=np.int32)] * len(task_query_row)

    # Group tasks by bucket so each mask is applied to a whole batch at once.
    for bucket in np.unique(task_bucket):
        allowed = bucket_allowed[bucket]
        tasks = np.flatnonzero(task_bucket == bucket)
        for start in range(0, len(tasks), batch_size):
            chunk = tasks[start : start + batch_size]
            rows = task_query_row[chunk]
            scores = (queries.matrix[rows] @ weights_t).toarray()
            scores *= allowed

            if exclude_rows is not None:
                for offset, query_row in enumerate(rows):
                    drop = exclude_rows[query_row]
                    if len(drop):
                        scores[offset, drop] = 0.0

            for offset, task in enumerate(chunk):
                results[task] = _top_k(scores[offset], k)

    return results


def retrieve(
    index: BM25Index,
    queries: QuerySet,
    k: int,
    exclude_rows: list[np.ndarray] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[np.ndarray]:
    """Top-k article row indices per user, best first.

    Batched because the dense score matrix is the memory wall: all of MIND's
    val users at once would be 37,777 x 65,238 x 4 bytes = 9.9 GB against 7 GB
    of RAM (see SCALE_NOTES.md). Batching makes peak memory a function of
    `batch_size`, not of user count - the thing that actually grows at 10x.

    Only articles with a score strictly greater than zero are returned, so a
    user matching fewer than k articles gets a shorter list rather than k-minus-
    that-many arbitrary zero-scored ones. Those would not have been retrieved by
    BM25 in any meaningful sense, and padding with them would let a top-200 list
    collect hits by chance.

    Args:
        exclude_rows: per user, article rows to remove before ranking (D15:
            the user's own history). Must be aligned with `queries.user_ids`.
    """
    weights_t = index.doc_term.T.tocsc()
    results: list[np.ndarray] = []

    for start in range(0, len(queries.user_ids), batch_size):
        end = min(start + batch_size, len(queries.user_ids))
        scores = (queries.matrix[start:end] @ weights_t).toarray()

        if exclude_rows is not None:
            for offset in range(end - start):
                drop = exclude_rows[start + offset]
                if len(drop):
                    scores[offset, drop] = 0.0

        for offset in range(end - start):
            results.append(_top_k(scores[offset], k))

    return results


# Moved to newsrec.retrieval.availability when Q3 needed the same logic.
# Re-exported so existing imports and tests keep working unchanged.
from newsrec.retrieval.availability import first_seen_times  # noqa: E402,F401
