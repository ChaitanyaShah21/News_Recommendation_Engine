"""slicing: cold-start vs warm users, head vs tail articles.

A slice is an evaluation-time grouping, not a feature. That distinction sets the
one rule these definitions must obey: **a slice may use anything except the
labels it is used to evaluate.** Grouping impressions by whether their clicked
article was popular *among val clicks* would be circular - the grouping would
already encode the answer. Exposure counted from `candidate_article_ids` (what
was shown) is not label-derived and is safe, which is the same predicate
discipline D19 imposed on availability.

COLD START (D26, threshold = 5)
The two datasets barely overlap on history length:

    history length per val impression   p0   p10   p25   p50   p75   p90
       MIND                              0     3     8    19    42    77
       EB-NeRD                           5    37    94   225   400   604

EB-NeRD's *coldest* user has more history than MIND's 25th percentile. An
absolute threshold therefore selects 17.7% of MIND's impressions and 0.3% of
EB-NeRD's (55 of them) - and that asymmetry is the finding, not a defect. A
per-dataset quantile would produce equal-sized slices by making "cold" mean
<= 8 articles on MIND and <= 94 on EB-NeRD, two different concepts sharing a
label. EB-NeRD's cold slice is too small to carry a useful confidence interval,
which is reported rather than hidden.

HEAD VS TAIL (D26, two definitions)
The textbook definition - head = articles carrying the top 50% of *training*
clicks - is degenerate on news data. Measured: it puts **97.9% of val clicks in
the tail on both datasets**, because 88.2% (MIND) / 90.5% (EB-NeRD) of the
corpus was never clicked during training at all, and 39.7% / 94.4% of val
clicked articles were never clicked in training either. The catalogue turns over
between windows, so training popularity barely predicts what gets clicked later.

Exposure-based head - articles accounting for 50% of all val impression slots -
splits both datasets usefully (67.3% / 54.9% of clicks land in head). Both are
computed; the contrast between them is itself a result.

MULTI-CLICK IMPRESSIONS
29.3% of MIND's val impressions carry more than one click, and those clicks can
straddle a head/tail boundary. Rather than invent a rule (majority? first
click?), such impressions are assigned to neither slice and counted separately.
A stated default under the Phase 3 pacing agreement, not a silent one.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl

DEFAULT_COLD_THRESHOLD = 5
DEFAULT_HEAD_FRACTION = 0.5


def cold_start_mask(
    history_len: np.ndarray, threshold: int = DEFAULT_COLD_THRESHOLD
) -> np.ndarray:
    """True where the user had at most `threshold` articles of history.

    Inclusive of the threshold, and inclusive of zero - D17's query-less users
    are the coldest case of the same thing, not a separate category.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    return np.asarray(history_len) <= threshold


def exposure_counts(impressions: pl.DataFrame) -> Counter:
    """How many impression slots each article occupied.

    Counted from `candidate_article_ids` - what the platform *showed* - and
    never from `clicked_article_ids`. Counting exposure from clicks would
    produce a very similar-looking ranking and would make the slice circular:
    impressions would be grouped by a quantity derived from their own labels.
    """
    if "clicked_article_ids" in impressions.columns and "candidate_article_ids" not in impressions.columns:
        raise ValueError("exposure must be counted from candidates, not clicks")
    counts: Counter = Counter()
    for candidates in impressions.get_column("candidate_article_ids").to_list():
        counts.update(candidates)
    return counts


def head_set_from_counts(counts: Counter | dict, fraction: float = DEFAULT_HEAD_FRACTION) -> set:
    """The smallest set of articles accounting for `fraction` of the total count.

    Articles are added most-frequent-first until the cumulative share reaches
    `fraction`, so the article that crosses the line is included. Ties in count
    are broken by article id, so the set is deterministic rather than dependent
    on Counter iteration order.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    total = sum(counts.values())
    if total == 0:
        return set()

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    target = total * fraction
    head, cumulative = set(), 0
    for article, count in ordered:
        head.add(article)
        cumulative += count
        if cumulative >= target:
            break
    return head


def head_tail_masks(
    clicked_lists: list[list[str]], head: set
) -> tuple[np.ndarray, np.ndarray, int]:
    """(head_mask, tail_mask, n_mixed) over impressions.

    An impression is head if *every* clicked article is in `head`, tail if none
    is, and mixed otherwise. Mixed impressions are in neither mask and are
    counted, so a reader can see how much of the data the slice pair excludes
    instead of it silently vanishing into one side.

    An impression with no clicks is in neither mask either - there is no article
    whose popularity could classify it.
    """
    n = len(clicked_lists)
    head_mask = np.zeros(n, dtype=bool)
    tail_mask = np.zeros(n, dtype=bool)
    mixed = 0

    for i, clicks in enumerate(clicked_lists):
        if not clicks:
            continue
        in_head = sum(1 for c in clicks if c in head)
        if in_head == len(clicks):
            head_mask[i] = True
        elif in_head == 0:
            tail_mask[i] = True
        else:
            mixed += 1

    return head_mask, tail_mask, mixed


def train_popularity_head_set(
    train_counts: np.ndarray, article_ids: list[str], fraction: float = DEFAULT_HEAD_FRACTION
) -> set:
    """The textbook head: articles carrying `fraction` of all *training* clicks.

    Kept alongside the exposure-based definition specifically to report that it
    does not work here - it leaves 97.9% of val clicks in the tail on both
    datasets. Reporting the degeneracy is more useful than quietly substituting
    a definition the spec did not name.
    """
    counts = {a: float(c) for a, c in zip(article_ids, train_counts) if c > 0}
    return head_set_from_counts(counts, fraction)
