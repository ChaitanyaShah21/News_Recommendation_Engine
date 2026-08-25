"""Ranking metrics for Q4.1: AUC, MRR, nDCG@5, nDCG@10.

These grade a *re-ranking* of the candidate list the platform actually showed
(`candidate_article_ids`), not a retrieval from the whole corpus. That is
forced, not chosen: every metric here needs to know, per item, whether it was
clicked or not, and only the shown list carries that label. A corpus article
the user never saw is unlabelled, not negative - counting it as a negative
would invent ~65,000 facts per impression the log never recorded.

Every function here works on one impression at a time and takes two aligned
arrays:

    scores  (n_candidates,)  float - our BM25 or cosine score for each candidate
    labels  (n_candidates,)  0/1   - was that candidate clicked

Aggregation across impressions is deliberately *not* done here. Q4.4's
bootstrap resamples impressions, so it needs the per-impression values, not a
mean that has already thrown them away.

Tie handling
------------
AUC has a tie rule built into its own definition (a tied pair scores 0.5).
MRR and nDCG do not - they need a total order, so a tie must be broken by
something outside the score.

Measured on real val data before choosing: BM25 scores 2.4% (MIND) / 4.0%
(EB-NeRD) of candidates exactly 0, the largest tie group averages 10.0% / 12.7%
of the rack, and 0.10% of MIND impressions have *every* candidate at 0 - there,
the tiebreak alone decides MRR and nDCG.

The tempting default is the dangerous one: `np.argsort` is stable, so "do
nothing" silently means "rank ties by their position in the raw candidate
list". That is only safe if raw order carries no click signal. Verified, rather
than assumed - mean normalised position of a clicked item is 0.5017 (MIND) and
0.4961 (EB-NeRD) against 0.5 for a uniform shuffle, so both platforms do
pre-shuffle. But that is a fact about these two val splits, not a property the
code would notice changing.

So we break ties explicitly, and by default `PESSIMISTIC` - clicked items go
*last* within their tie group, making every metric a lower bound that no
tie-luck can inflate. `OPTIMISTIC` (clicked first) is provided so the two can
be reported together: if the bounds agree to a fraction of a percent, that one
sentence retires the whole question for the design note, which a single number
could never do.

Undefined values
----------------
Returned as `float("nan")`, never as 0.0. A metric that cannot be computed and
a metric that scored zero are different facts, and collapsing them silently
drags every mean downward. `macro_mean` therefore reports how many impressions
were undefined rather than quietly skipping them.

    P = number of clicked candidates, N = number of unclicked.
    AUC   undefined when P == 0 or N == 0   (no positive-negative pair exists)
    MRR   undefined when P == 0             (no first correct item to find)
    nDCG  undefined when P == 0             (the ideal DCG is 0, so 0/0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

PESSIMISTIC = "pessimistic"
OPTIMISTIC = "optimistic"


def _validate(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to aligned float/int arrays and reject anything unrankable.

    NaN is rejected loudly rather than tolerated. `np.argsort` places NaN
    *last* without complaint, so a single NaN score would quietly become "worst
    candidate" - and NaN is exactly what a mean over zero history vectors
    produces, the cold-start trap Phase 3 already hit once.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(np.int8)

    if scores.shape != labels.shape:
        raise ValueError(
            f"scores has {scores.shape[0]} candidates but labels has "
            f"{labels.shape[0]}; they must be aligned position for position"
        )
    if not np.isfinite(scores).all():
        raise ValueError(
            "scores contain NaN or inf; argsort would silently rank those last "
            "instead of failing, so they are rejected here"
        )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0/1 - these metrics assume binary clicks")

    return scores, labels


def rank_order(
    scores: np.ndarray, labels: np.ndarray, tie: str = PESSIMISTIC
) -> np.ndarray:
    """Candidate indices sorted best-first, with ties broken by `tie`.

    `np.lexsort` sorts by its *last* key first, so the primary key goes last in
    the tuple. `-scores` ascending is scores descending. The secondary key then
    orders within a tie group only:

        PESSIMISTIC: key `labels` ascending -> 0 before 1 -> clicked items last
        OPTIMISTIC:  key `-labels`          -> 1 before 0 -> clicked items first
    """
    scores, labels = _validate(scores, labels)
    if tie == PESSIMISTIC:
        return np.lexsort((labels, -scores))
    if tie == OPTIMISTIC:
        return np.lexsort((-labels, -scores))
    raise ValueError(f"unknown tie policy {tie!r}; expected {PESSIMISTIC!r} or {OPTIMISTIC!r}")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve for one impression.

    Computed through ranks rather than by enumerating pairs. The pair form is

        AUC = (1/(P*N)) * sum over (clicked p, unclicked n) of
              1 if s_p > s_n, 0.5 if s_p == s_n, 0 otherwise

    which is O(P*N). The identical rank form is O(n log n):

        AUC = (sum of the clicked items' ranks - P*(P+1)/2) / (P*N)

    `rankdata(..., method="average")` gives tied items their *mean* rank, and
    that averaging is precisely what produces the 0.5 credit - so AUC needs no
    tie policy of its own and is unaffected by `rank_order`'s choice. Worth
    stating explicitly, because it means an AUC that moves between the
    pessimistic and optimistic runs is a bug, not a finding.

    Ranks are taken ascending (worst = 1), so a clicked item scoring above
    everything else earns the largest rank, and subtracting P*(P+1)/2 removes
    the ranks the clicked items spend on each other.
    """
    scores, labels = _validate(scores, labels)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def reciprocal_rank(
    scores: np.ndarray, labels: np.ndarray, tie: str = PESSIMISTIC
) -> float:
    """1 / (rank of the first clicked candidate), ranks 1-indexed."""
    order = rank_order(scores, labels, tie)
    ordered_labels = np.asarray(labels).ravel().astype(np.int8)[order]
    hits = np.flatnonzero(ordered_labels)
    if hits.size == 0:
        return float("nan")
    return float(1.0 / (hits[0] + 1))


def _dcg(gains: np.ndarray) -> float:
    """sum of gain_i / log2(i + 1), i 1-indexed over the array as given."""
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float((gains / discounts).sum())


def ndcg_at_k(
    scores: np.ndarray, labels: np.ndarray, k: int, tie: str = PESSIMISTIC
) -> float:
    """Normalised discounted cumulative gain over the top `k` candidates.

    The ideal ranking is the same impression's clicked items packed into the
    top slots, so nDCG is 1.0 exactly when every clicked candidate that *can*
    fit inside k is inside k, in the top positions.

    Note `min(n_pos, k)` in the ideal: an impression with 21 clicks cannot score
    more than 10 hits inside nDCG@10, and normalising by an unreachable ideal
    would cap that impression below 1.0 for a reason that is about the cutoff,
    not about our ranking.

    When k exceeds the number of candidates this stops being a top-k metric and
    quietly becomes nDCG@all - a full-list ordering measure, close to what AUC
    already reports. That is the normal case on EB-NeRD, whose median rack is 9
    candidates, so nDCG@10 there is not measuring what its name suggests.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    order = rank_order(scores, labels, tie)
    labels = np.asarray(labels).ravel().astype(np.int8)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    actual = _dcg(labels[order][:k].astype(np.float64))
    ideal = _dcg(np.ones(min(n_pos, k), dtype=np.float64))
    return float(actual / ideal)


@dataclass
class ImpressionMetrics:
    """Per-impression metric values, aligned row-for-row with the input.

    Kept as arrays rather than means because Q4.4's bootstrap resamples
    impressions - it needs the individual values, and a mean has already
    discarded them. NaN marks undefined, never zero.
    """

    auc: np.ndarray
    mrr: np.ndarray
    ndcg_at_5: np.ndarray
    ndcg_at_10: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "AUC": self.auc,
            "MRR": self.mrr,
            "nDCG@5": self.ndcg_at_5,
            "nDCG@10": self.ndcg_at_10,
        }


def evaluate_impressions(
    scores: list[np.ndarray],
    labels: list[np.ndarray],
    tie: str = PESSIMISTIC,
) -> ImpressionMetrics:
    """All four metrics for a list of impressions, one value each per impression.

    Args:
        scores: per-impression score arrays, one per candidate in that
            impression's own candidate list.
        labels: per-impression 0/1 arrays, aligned position-for-position with
            `scores`. Different impressions have different lengths - MIND's
            racks run 2 to 295 - so these are lists of arrays, not a matrix.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores has {len(scores)} impressions but labels has {len(labels)}"
        )

    n = len(scores)
    out = {name: np.full(n, np.nan) for name in ("auc", "mrr", "ndcg5", "ndcg10")}

    for i, (s, y) in enumerate(zip(scores, labels)):
        out["auc"][i] = auc(s, y)
        out["mrr"][i] = reciprocal_rank(s, y, tie)
        out["ndcg5"][i] = ndcg_at_k(s, y, 5, tie)
        out["ndcg10"][i] = ndcg_at_k(s, y, 10, tie)

    return ImpressionMetrics(
        auc=out["auc"],
        mrr=out["mrr"],
        ndcg_at_5=out["ndcg5"],
        ndcg_at_10=out["ndcg10"],
    )


def macro_mean(values: np.ndarray) -> tuple[float, int, int]:
    """Mean over the impressions where the metric is defined.

    Returns (mean, n_defined, n_undefined). The undefined count is returned
    rather than logged or dropped: `np.nanmean` on an all-NaN array warns and
    returns NaN, and a metric quietly averaged over 60% of the impressions is
    the kind of number that reaches a design note unchallenged.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = ~np.isnan(values)
    n_defined = int(defined.sum())
    n_undefined = int(values.size - n_defined)
    mean = float(values[defined].mean()) if n_defined else float("nan")
    return mean, n_defined, n_undefined
