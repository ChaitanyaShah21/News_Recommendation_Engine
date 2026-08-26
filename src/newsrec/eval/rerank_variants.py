"""Phase 5b - re-ranking variants tried after the first MIND leaderboard result.

Written as a separate module from `rerank.py` deliberately: that file's four
scorers are the ones every Q4 number in the design note was produced with, and
they are pinned by 240 tests. Nothing here modifies them.

Three ideas, each aimed at a weakness our own val measurements identified rather
than at "try things until the number moves":

1. **max-similarity user representation** (`score_semantic_maxsim`). Finding 4
   recorded a concrete failure of mean pooling: MIND user `mind:U13132` read
   three political stories and one about a Starbucks latte, and semantic
   retrieval returned five near-duplicate Popeyes articles. The mechanism is
   that nearest-neighbour search is won by *dense* regions, so a mean sitting
   nearer politics overall still lands inside a tight food cluster. Scoring a
   candidate by its similarity to the user's **single most similar** history
   article has no mean to hijack - a user with several distinct interests keeps
   all of them instead of being collapsed to their centroid.

2. **N, the history window** (`n_recent` throughout). D12 chose N = 10 for
   *retrieval*, where a long query causes topic drift. Re-ranking is a different
   job - the candidate list is already topically plausible, so more history may
   be signal rather than noise. `PROGRESS.md` has listed this sweep as "never
   run" since Phase 3.

3. **Score fusion** (`normalise_per_impression`, `blend`). BM25, semantic and
   popularity fail differently: BM25 needs term overlap, semantic is blind to
   popularity, popularity is blind to the user. Fusing them is the standard way
   to buy the union of three partial signals.

**Why fusion needs normalisation.** The three scorers live on incompatible
scales - cosine on [-1, 1], BM25 on [0, ~50], popularity on raw click counts in
the thousands. Summing them raw is not a blend; it is popularity with rounding
error. Both offered normalisations are computed **within each impression**,
which is also the unit every metric and both leaderboards score on.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .rerank import CandidateSet

# Below this, a per-impression standard deviation is treated as "every candidate
# scored the same". Mirrors semantic_search.MIN_NORM's reasoning: a tiny non-zero
# value is float residue, and dividing by it manufactures confident nonsense from
# noise.
MIN_STD = 1e-9


def build_history_rows(
    history: pl.DataFrame, article_ids: list[str], n_recent: int
) -> dict[str, np.ndarray]:
    """user_id -> the article-matrix rows of that user's last `n_recent` clicks.

    Same tail semantics as `build_user_vectors` (end of the list is most
    recent, D12) and the same de-duplication: an article clicked twice must not
    count twice, or a repeated click silently becomes a stronger vote.
    """
    row_of = {a: i for i, a in enumerate(article_ids)}
    out: dict[str, np.ndarray] = {}
    for uid, ids in zip(
        history.get_column("user_id").to_list(),
        history.get_column("history_article_ids").to_list(),
    ):
        tail = (ids or [])[-n_recent:]
        rows = np.unique(
            np.asarray([row_of[a] for a in tail if a in row_of], dtype=np.int32)
        )
        out[uid] = rows
    return out


def score_semantic_maxsim(
    candidates: CandidateSet,
    history_rows: dict[str, np.ndarray],
    embeddings: np.ndarray,
) -> list[np.ndarray]:
    """Cosine to the user's *nearest* history article, not to their centroid.

    Impressions are grouped by user so each user's history-embedding block is
    gathered once rather than once per impression - MIND val has 51,205
    impressions behind 37,777 users.

    A user with no usable history scores a flat zero, exactly as
    `score_semantic` does, so the two variants treat cold start identically and
    a comparison between them measures the representation and nothing else.
    """
    scores: list[np.ndarray] = [None] * len(candidates)  # type: ignore[list-item]

    by_user: dict[str, list[int]] = {}
    for i, uid in enumerate(candidates.user_ids):
        by_user.setdefault(uid, []).append(i)

    for uid, idxs in by_user.items():
        rows = history_rows.get(uid)
        if rows is None or len(rows) == 0:
            for i in idxs:
                scores[i] = np.zeros(len(candidates.candidate_rows[i]), dtype=np.float32)
            continue

        hist = embeddings[rows]  # (k, 384), already L2-normalised upstream
        for i in idxs:
            cand = embeddings[candidates.candidate_rows[i]]  # (n, 384)
            # (k, n) similarities, then the best history article per candidate.
            scores[i] = np.asarray((hist @ cand.T).max(axis=0), dtype=np.float32)

    return scores


def normalise_per_impression(
    scores: list[np.ndarray], mode: str = "rank"
) -> list[np.ndarray]:
    """Put one scorer's outputs on a common scale, impression by impression.

    `mode="rank"`: percentile rank in [0, 1]. Discards magnitude entirely, so no
        single scorer can dominate a blend through scale alone, and an outlier
        cannot drag a weighted sum. Ties share their average rank.
    `mode="zscore"`: (s - mean) / std. Keeps margin, so "far and away the best
        candidate" outvotes "narrowly the best" - which rank normalisation
        cannot express. Both are offered because which one wins is an empirical
        question, and guessing it is exactly the sort of silent choice R6 exists
        to prevent.
    """
    if mode not in ("rank", "zscore"):
        raise ValueError(f"mode must be 'rank' or 'zscore', got {mode!r}")

    out: list[np.ndarray] = []
    for s in scores:
        s = np.asarray(s, dtype=np.float64)
        n = s.shape[0]
        if n == 1:
            # A single candidate has no spread to normalise and no ranking to
            # affect; 0.0 keeps it neutral under any weighting.
            out.append(np.zeros(1, dtype=np.float32))
            continue

        if mode == "zscore":
            std = s.std()
            out.append(
                np.zeros(n, dtype=np.float32)
                if std < MIN_STD
                else np.asarray((s - s.mean()) / std, dtype=np.float32)
            )
        else:
            order = np.argsort(s, kind="stable")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            # Average tied ranks, so an all-tied impression maps to a constant
            # rather than to the arbitrary order argsort happened to produce.
            uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
            if len(uniq) < n:
                sums = np.zeros(len(uniq));  np.add.at(sums, inv, ranks)
                ranks = (sums / counts)[inv]
            out.append(np.asarray(ranks / (n - 1), dtype=np.float32))

    return out


def blend(
    components: dict[str, list[np.ndarray]], weights: dict[str, float]
) -> list[np.ndarray]:
    """Weighted sum of already-normalised component scores.

    Every weighted component must cover every impression, checked rather than
    assumed: a length mismatch would otherwise broadcast or truncate silently
    and blend impression i's scores with impression j's.
    """
    used = [k for k, w in weights.items() if w != 0.0]
    if not used:
        raise ValueError("all blend weights are zero")
    missing = [k for k in used if k not in components]
    if missing:
        raise KeyError(f"no component scores for {missing}")

    n = len(components[used[0]])
    for k in used:
        if len(components[k]) != n:
            raise ValueError(
                f"component {k!r} has {len(components[k])} impressions, expected {n}"
            )

    out: list[np.ndarray] = []
    for i in range(n):
        acc = None
        for k in used:
            part = np.asarray(components[k][i], dtype=np.float32) * weights[k]
            if acc is None:
                acc = part
            else:
                if part.shape != acc.shape:
                    raise ValueError(
                        f"impression {i}: component {k!r} has {part.shape[0]} "
                        f"candidates, expected {acc.shape[0]}"
                    )
                acc = acc + part
        out.append(acc)
    return out
