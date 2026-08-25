"""Re-ranking the supplied candidate list (Q4.2), for four scorers.

Q2 and Q3 retrieved top-K from the whole corpus. This module does the other
operation: it takes the candidate list the platform actually showed -
`candidate_article_ids` (MIND's `impressions` field, EB-NeRD's
`article_ids_inview`) - and orders it. Same scoring functions, different
candidate set. That is what Q4's metrics can grade, and what both Codabench
leaderboards consume.

Four scorers, all producing one float per candidate:

    bm25        the D11-D16 query run against the D14 index
    semantic    cosine between the D22 mean-pooled user vector and the article
    popularity  train-window click count - the non-personalised baseline
    random      seeded shuffle - the floor

The two baselines exist because MRR and nDCG have no natural zero point the way
AUC's 0.5 does. Phase 2 already paid for that lesson once: EB-NeRD's recall@200
looked like a 3x win over the whole-corpus run until the random-baseline column
showed the pool had simply shrunk.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
D15 excluded the user's own history articles from retrieval. Re-ranking cannot:
the platform chose the list, and every item on it needs a score. Nor does the
re-ranker adjust for them. Measured, they behave in *opposite* directions on the
two datasets - a candidate already in the user's history is clicked 3.5x more
often than average on MIND (14.3% vs 4.1%) and 0.49x as often on EB-NeRD (4.1%
vs 8.4%) - so "seen before" is a real signal, but folding it into the score
would mean the number reported as "BM25 nDCG@5" is two systems blended. That is
the conflation D17 rejected a popularity fallback for. It belongs in Q9's
serving-time-feature ablation, where it is graded rather than smuggled in.

COLD START (D17)
A user with no usable history gets an all-zero query, so every candidate ties.
Under the pessimistic tie policy (D23) that ranks the clicked items last, which
is the honest lower bound for "the scorer said nothing". `has_query` is carried
through per impression so the headline can be reported over impressions with a
query and the all-impressions number alongside, exactly as D17 chose for recall.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

from newsrec.retrieval.bm25 import BM25Index
from newsrec.retrieval.bm25_search import QuerySet
from newsrec.retrieval.semantic_search import UserVectors

DEFAULT_BATCH_SIZE = 256
RANDOM_SEED = 20260825


@dataclass
class CandidateSet:
    """The impressions to be re-ranked, flattened into row indices.

    Attributes:
        impression_ids: one per impression, in input order.
        user_ids: one per impression.
        candidate_rows: per impression, the article-matrix row of each candidate,
            in the platform's own order. That order is preserved rather than
            sorted, because the metrics' tie policy is defined against it.
        labels: per impression, 0/1 aligned position-for-position with
            `candidate_rows`. **None** for the Phase 5 submission splits, which
            have no labels at all (D30) - so any metric code reaching for a
            label on that path fails immediately rather than averaging over
            fabricated zeros. Every evaluation path always supplies real labels.
        history_len: per impression, the user's history length - carried for
            Q4.3's cold-start-versus-warm slice so it need not be re-derived.
    """

    impression_ids: list[str]
    user_ids: list[str]
    candidate_rows: list[np.ndarray]
    labels: list[np.ndarray]
    history_len: np.ndarray

    def __len__(self) -> int:
        return len(self.impression_ids)


def build_candidate_set(
    impressions: pl.DataFrame,
    article_ids: list[str],
    history: pl.DataFrame,
) -> CandidateSet:
    """Map each impression's candidate ids onto article-matrix rows.

    Raises:
        KeyError: if a candidate is absent from the article matrix. Measured as
            0 of 1,895,867 (MIND) and 0 of 212,474 (EB-NeRD) candidates on val,
            so this is a loud guard on an invariant rather than a live case -
            but a missing candidate would otherwise need an invented score, and
            inventing one silently is how a fake number reaches a design note.
    """
    row_of_article = {aid: i for i, aid in enumerate(article_ids)}
    hist_len = {
        u: len(h or [])
        for u, h in zip(
            history.get_column("user_id").to_list(),
            history.get_column("history_article_ids").to_list(),
        )
    }

    impression_ids = impressions.get_column("impression_id").to_list()
    user_ids = impressions.get_column("user_id").to_list()
    cand_lists = impressions.get_column("candidate_article_ids").to_list()
    click_lists = impressions.get_column("clicked_article_ids").to_list()

    candidate_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for imp_id, cands, clicks in zip(impression_ids, cand_lists, click_lists):
        missing = [c for c in cands if c not in row_of_article]
        if missing:
            raise KeyError(
                f"impression {imp_id} has {len(missing)} candidate(s) absent from "
                f"the article matrix, e.g. {missing[0]!r}"
            )
        candidate_rows.append(
            np.fromiter((row_of_article[c] for c in cands), dtype=np.int32, count=len(cands))
        )
        clicked = set(clicks)
        labels.append(
            np.fromiter((c in clicked for c in cands), dtype=np.int8, count=len(cands))
        )

    return CandidateSet(
        impression_ids=impression_ids,
        user_ids=user_ids,
        candidate_rows=candidate_rows,
        labels=labels,
        history_len=np.fromiter(
            (hist_len.get(u, 0) for u in user_ids), dtype=np.int32, count=len(user_ids)
        ),
    )


def _rows_by_query(user_ids: list[str], query_user_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per impression, which query row it uses; -1 for a user with no query row.

    A user absent from the history table is as query-less as one with an empty
    history - a distinction worth erasing here rather than letting a KeyError
    decide it three functions deeper.
    """
    row_of_user = {u: i for i, u in enumerate(query_user_ids)}
    rows = np.fromiter(
        (row_of_user.get(u, -1) for u in user_ids), dtype=np.int64, count=len(user_ids)
    )
    return rows, rows >= 0


def _scores_from_rows(
    candidates: CandidateSet,
    query_rows: np.ndarray,
    score_batch,
    batch_size: int,
) -> list[np.ndarray]:
    """Gather per-candidate scores, computing each distinct query row once.

    `score_batch(rows) -> (len(rows) x n_articles)` is the only thing that
    differs between BM25 and semantic, so the batching, the grouping and the
    gather are written once rather than twice.

    Batching is the memory wall, unchanged from D14/D21: a full 37,777 x 65,238
    float32 score matrix is 9.9 GB against 7 GB of RAM. Peak memory here is a
    function of `batch_size`, not of impression count - the thing that grows at
    10x scale.
    """
    scores: list[np.ndarray] = [None] * len(candidates)  # type: ignore[list-item]

    # Impressions with no query row score a flat zero: every candidate ties, and
    # the pessimistic policy then ranks the clicked ones last. That is the
    # honest reading of "the scorer distinguished nothing", and it is why this
    # is 0.0 rather than NaN - NaN would be rejected by the metrics as unrankable.
    for i in np.flatnonzero(query_rows < 0):
        scores[i] = np.zeros(len(candidates.candidate_rows[i]), dtype=np.float32)

    have = np.flatnonzero(query_rows >= 0)
    # Group impressions by query row so a user shared by many impressions is
    # scored once. MIND: 51,205 impressions behind 37,777 users; EB-NeRD:
    # 17,749 behind 1,437, a 12x saving there.
    order = have[np.argsort(query_rows[have], kind="stable")]
    unique_rows, starts = np.unique(query_rows[order], return_index=True)
    groups = np.split(order, starts[1:])

    for start in range(0, len(unique_rows), batch_size):
        chunk_rows = unique_rows[start : start + batch_size]
        block = score_batch(chunk_rows)
        for offset, group in enumerate(groups[start : start + batch_size]):
            row = block[offset]
            for i in group:
                scores[i] = row[candidates.candidate_rows[i]].astype(np.float32)

    return scores


def score_bm25(
    candidates: CandidateSet,
    index: BM25Index,
    queries: QuerySet,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[np.ndarray]:
    """BM25 score of each candidate against its user's query.

    Identical formula, index and queries as Q2 - only the candidate set changes.
    A candidate sharing no term with the query scores exactly 0.0, which is what
    makes D23's tie policy load-bearing rather than decorative.
    """
    weights_t = index.doc_term.T.tocsc()
    query_rows, has_query = _rows_by_query(candidates.user_ids, queries.user_ids)
    # A user present in the history table but with an empty query is query-less
    # too - detected from the query matrix, not from history length.
    query_rows = np.where(
        has_query & queries.has_query[np.clip(query_rows, 0, None)], query_rows, -1
    )

    def batch(rows: np.ndarray) -> np.ndarray:
        return (queries.matrix[rows] @ weights_t).toarray()

    return _scores_from_rows(candidates, query_rows, batch, batch_size)


def score_semantic(
    candidates: CandidateSet,
    users: UserVectors,
    embeddings: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[np.ndarray]:
    """Cosine similarity between the mean-pooled user vector and each candidate.

    Both sides are L2-normalised upstream, so the dot product *is* cosine. No
    -inf masking is needed here, unlike retrieval: nothing is being excluded, so
    there is no "worse than the worst real score" sentinel to place.
    """
    query_rows, has_query = _rows_by_query(candidates.user_ids, users.user_ids)
    query_rows = np.where(
        has_query & users.has_query[np.clip(query_rows, 0, None)], query_rows, -1
    )

    def batch(rows: np.ndarray) -> np.ndarray:
        return users.matrix[rows] @ embeddings.T

    return _scores_from_rows(candidates, query_rows, batch, batch_size)


def train_click_counts(
    train_impressions: pl.DataFrame, article_ids: list[str]
) -> np.ndarray:
    """Clicks per article over the TRAIN split only, aligned to article rows.

    Train-only is not a detail. Counting clicks over val would mean the
    popularity baseline had seen the answers it is being scored against - a
    label leak, and precisely the kind Q9 asks for a test about. The caller
    passes the train partition; this function refuses anything else rather than
    trusting it.
    """
    splits = set(train_impressions.get_column("split").unique().to_list())
    if splits != {"train"}:
        raise ValueError(
            f"popularity must be counted over train only, got splits {sorted(splits)}; "
            "counting val or test clicks would leak the labels being scored"
        )

    row_of_article = {aid: i for i, aid in enumerate(article_ids)}
    counts = np.zeros(len(article_ids), dtype=np.float32)
    for clicks in train_impressions.get_column("clicked_article_ids").to_list():
        for article in clicks:
            row = row_of_article.get(article)
            if row is not None:
                counts[row] += 1.0
    return counts


def score_popularity(candidates: CandidateSet, counts: np.ndarray) -> list[np.ndarray]:
    """Rank candidates by how often they were clicked during training.

    Not personalised at all - every user sees the same ordering of the same
    candidates. An article never clicked in train scores 0.0, and there are many
    of those, so this baseline leans hard on the tie policy: under D23's
    pessimistic rule it gets no credit for guessing among them.
    """
    return [counts[rows].copy() for rows in candidates.candidate_rows]


def score_random(candidates: CandidateSet, seed: int = RANDOM_SEED) -> list[np.ndarray]:
    """A seeded random score per candidate - the interpretive floor.

    The seed is derived per impression from a CRC32 of its id rather than drawn
    from one stream in iteration order, so the result depends on the impression
    itself and not on how many impressions happened to be processed before it.
    Python's built-in `hash` is salted per process and would make this
    reproducible within a run but not between runs.
    """
    out: list[np.ndarray] = []
    for imp_id, rows in zip(candidates.impression_ids, candidates.candidate_rows):
        rng = np.random.default_rng(seed ^ zlib.crc32(imp_id.encode("utf-8")))
        out.append(rng.random(len(rows)).astype(np.float32))
    return out
