"""Semantic retrieval - the query side (Q3.2, Q3.3).

`semantic.py` turned articles into a dense matrix. This module turns *users*
into vectors and ranks the corpus against them.

The shape of the computation, deliberately parallel to `bm25_search.py`:

    S  (users x articles)   1/n at each of the user's last N clicked articles
    M  (articles x 384)     L2-normalised article embeddings
    U = S @ M               (users x 384) mean-pooled user vectors, re-normalised
    scores = U @ M.T        (users x articles) cosine similarity

The first product is Q3.3's mean pooling: a row of S holding 1/n at n articles,
multiplied into M, *is* the component-wise average of those n vectors. Same
selection-matrix trick BM25 used to concatenate titles, except the right-hand
side is dense.

The second is cosine similarity - and it is only cosine because both sides are
unit length. `semantic.py` normalises the articles and asserts it at load;
`build_user_vectors` normalises the pooled user vector here.

WHERE THIS DIVERGES FROM BM25, AND IT MATTERS
BM25 scores are >= 0, and 0 means "shares no term with the query" - so
`bm25_search` excludes candidates by setting their score to 0.0 and masks
availability by multiplying with a 0/1 array. Cosine similarity is in [-1, 1]
and **0 is mid-range, not the floor** (measured on our own model: an unrelated
pair scored -0.051). Reusing BM25's convention here would float every excluded
and every unavailable article above every genuinely negative-scoring one. This
module masks with -inf instead. The difference is invisible on the whole-corpus
run and decisive on D19's availability run, where EB-NeRD's pool drops to ~2,963
of 11,777 articles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

DEFAULT_N_RECENT = 10
DEFAULT_BATCH_SIZE = 256

# Below this, a pooled user vector's direction is float noise rather than signal.
# Well above float32's epsilon (~1.2e-07) so near-cancellation is caught, and far
# below any genuine mean-of-unit-vectors: even 10 mutually perpendicular unit
# vectors average to norm 1/sqrt(10) = 0.32.
MIN_NORM = 1e-6


@dataclass
class UserVectors:
    """Mean-pooled user representations, aligned row-for-row with `user_ids`.

    Attributes:
        user_ids: row i of `matrix` belongs to `user_ids[i]`.
        matrix: (n_users x 384) float32, each row L2-normalised - except
            query-less rows, which are all zeros.
        has_query: False where the user produced no usable vector - a cold-start
            user with empty history, or one whose history articles are all
            absent from the embedding store. D17 excludes these from the
            headline recall and reports them separately.
    """

    user_ids: list[str]
    matrix: np.ndarray
    has_query: np.ndarray


def build_user_vectors(
    history: pl.DataFrame,
    article_ids: list[str],
    embeddings: np.ndarray,
    n_recent: int = DEFAULT_N_RECENT,
) -> UserVectors:
    """Mean-pool each user's last `n_recent` clicked article embeddings (Q3.3).

    Args:
        history: rows of the unified `history` table for one dataset and split.
        article_ids: row order of `embeddings`, from `load_article_embeddings`.
        embeddings: (n_articles x 384) L2-normalised float32.
        n_recent: D12's N, reused unchanged so Q3.5 varies the algorithm and not
            the query length. Taken from the END of the history list, which is
            chronological oldest-first.
    """
    row_of_article = {aid: i for i, aid in enumerate(article_ids)}
    n_articles, dim = embeddings.shape

    user_ids = history.get_column("user_id").to_list()
    histories = history.get_column("history_article_ids").to_list()

    # Build the selection matrix with 1/n rather than 1, so the matrix product
    # yields the mean directly instead of the sum. Rows with no articles get no
    # entries at all, which produces an all-zero mean - and crucially NOT a
    # division by zero, because there is no division to perform.
    indptr = np.zeros(len(user_ids) + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[float] = []
    for i, ids in enumerate(histories):
        # `or []` guards null as well as empty: a cold-start user is legitimately
        # empty, and the upstream `.fill_null([])` has caught us out once before.
        tail = (ids or [])[-n_recent:]
        # unique for the same reason BM25 does it: an article clicked twice in
        # the last N must not be weighted twice in the mean.
        rows = np.unique(
            np.asarray([row_of_article[a] for a in tail if a in row_of_article],
                       dtype=np.int32)
        )
        if len(rows):
            indices.extend(rows.tolist())
            data.extend([1.0 / len(rows)] * len(rows))
        indptr[i + 1] = len(indices)

    selection = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            indptr,
        ),
        shape=(len(user_ids), n_articles),
        dtype=np.float32,
    )

    pooled = np.asarray(selection @ embeddings, dtype=np.float32)

    # Re-normalise: the mean of unit vectors is NOT itself unit length (it is
    # shorter the more they disagree), and `scores = U @ M.T` is only cosine
    # similarity if both sides are unit length.
    norms = np.linalg.norm(pooled, axis=1)

    # Three ways to end up with no usable direction:
    #   - the user contributed no articles at all (cold start),
    #   - their vectors cancelled out exactly (norm 0), or
    #   - their vectors very nearly cancelled (norm tiny but non-zero).
    #
    # The third is the one a `norms > 0` test misses, and it is the nastiest.
    # Two nearly-opposite unit vectors average to norm ~5e-08, which is
    # comfortably greater than zero - so the user is treated as having a real
    # query, and normalising blows that residue back up to unit length. The
    # resulting "direction" is decided entirely by floating-point noise, and we
    # then confidently retrieve 200 articles for it.
    #
    # MIN_NORM says: if N unit vectors average to something this short, they
    # disagree so completely that their mean carries no direction worth using.
    # Found by mutation-testing this function rather than by a traceback.
    has_query = norms > MIN_NORM
    # np.where evaluates both branches, so the divisor is patched away from zero
    # before the division rather than after it. The row is zeroed below as well;
    # that second guard is what actually guarantees no NaN survives.
    safe = np.where(has_query, norms, 1.0).astype(np.float32)
    matrix = (pooled / safe[:, None]).astype(np.float32)
    matrix[~has_query] = 0.0

    return UserVectors(user_ids=user_ids, matrix=matrix, has_query=has_query)


def _top_k(row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest finite entries of `row`, best first.

    Unlike BM25's version this does NOT filter to strictly-positive scores. A
    cosine of -0.2 is a real, ranked similarity - "less similar than
    perpendicular", not "no match" - so filtering it out would discard genuine
    rankings. Only -inf entries are dropped, and those are exactly the ones this
    module masked out deliberately (D15 exclusions, D19 unavailable articles).
    """
    finite = np.flatnonzero(np.isfinite(row))
    if finite.size == 0:
        return np.zeros(0, dtype=np.int32)
    if finite.size > k:
        top = finite[np.argpartition(-row[finite], k)[:k]]
    else:
        top = finite
    # kind="stable" so equal scores break ties by article row order identically
    # on every run - an unstable sort would make recall@K irreproducible from
    # identical inputs.
    return top[np.argsort(-row[top], kind="stable")].astype(np.int32)


def _score_batch(
    users: np.ndarray,
    embeddings: np.ndarray,
    allowed: np.ndarray | None,
    exclude: list[np.ndarray] | None,
    query_rows: np.ndarray,
) -> np.ndarray:
    """(batch x n_articles) cosine scores, with masked entries set to -inf."""
    scores = users @ embeddings.T

    if allowed is not None:
        # -inf, not multiplication by a 0/1 mask. See the module docstring.
        scores = np.where(allowed, scores, -np.inf)

    if exclude is not None:
        for offset, query_row in enumerate(query_rows):
            drop = exclude[query_row]
            if len(drop):
                scores[offset, drop] = -np.inf

    return scores


def retrieve(
    users: UserVectors,
    embeddings: np.ndarray,
    k: int,
    exclude_rows: list[np.ndarray] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[np.ndarray]:
    """Top-k article rows per user, best first.

    Batched for the same reason BM25 is: all of MIND's val users at once would be
    37,777 x 65,238 x 4 bytes = 9.9 GB against 7 GB of RAM. Peak memory is a
    function of `batch_size`, not of user count.

    A query-less user (D17) has an all-zero vector, which scores 0 against every
    article and would otherwise return an arbitrary tie-broken top-k. They get an
    empty list instead, so they are counted as misses rather than credited with
    an accidental hit.
    """
    results: list[np.ndarray] = []

    for start in range(0, len(users.user_ids), batch_size):
        end = min(start + batch_size, len(users.user_ids))
        rows = np.arange(start, end)
        scores = _score_batch(
            users.matrix[start:end], embeddings, None, exclude_rows, rows
        )
        for offset in range(end - start):
            if not users.has_query[start + offset]:
                results.append(np.zeros(0, dtype=np.int32))
            else:
                results.append(_top_k(scores[offset], k))

    return results


def retrieve_bucketed(
    users: UserVectors,
    embeddings: np.ndarray,
    task_query_row: np.ndarray,
    task_bucket: np.ndarray,
    bucket_allowed: list[np.ndarray],
    k: int,
    exclude_rows: list[np.ndarray] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[np.ndarray]:
    """Top-k per *task*, where a task is one (user, time-bucket) pair (D19).

    A user's vector is fixed within a split; what is *available* to retrieve is
    not, so the same user must be re-ranked in each bucket they appear in.

    Args:
        bucket_allowed: per bucket, a boolean mask over article rows - True where
            the article had already appeared in the impression log before that
            bucket began. Boolean, not the float32 0/1 BM25 uses, because these
            are consumed by np.where rather than by multiplication.
        exclude_rows: indexed by *query row*, not by task - D15's exclusion is a
            property of the user, not of the moment.
    """
    results: list[np.ndarray] = [np.zeros(0, dtype=np.int32)] * len(task_query_row)

    for bucket in np.unique(task_bucket):
        allowed = bucket_allowed[bucket]
        tasks = np.flatnonzero(task_bucket == bucket)
        for start in range(0, len(tasks), batch_size):
            chunk = tasks[start : start + batch_size]
            rows = task_query_row[chunk]
            scores = _score_batch(
                users.matrix[rows], embeddings, allowed, exclude_rows, rows
            )
            for offset, task in enumerate(chunk):
                if not users.has_query[rows[offset]]:
                    continue  # stays the empty array it was initialised to
                results[task] = _top_k(scores[offset], k)

    return results
