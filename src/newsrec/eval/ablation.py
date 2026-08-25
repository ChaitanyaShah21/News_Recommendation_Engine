"""features unavailable at serving time, and what they would buy.

The assignment asks us to *"report metrics with and without features
unavailable at serving time"*. Note the direction: the point is not to remove
legitimate features, it is to **add an illegitimate one and measure how much it
inflates the score** - so the design note can say what a leak is worth here
rather than asserting that leaks are bad.

THE TEST FOR "AVAILABLE AT SERVING TIME"
Could this number have been computed at the moment the recommendation was made?
Not "is it in the file" - the file contains the whole logged history, including
things that had not happened yet when the impression was served.

  available    the user's click history before now, an article's text, its
               category, how often it had already been shown before now,
               whether this user has already read it
  NOT available how many clicks this article receives during the window we are
               evaluating, how long the user will read it, whether they will
               click at all

`future_click_counts` computes the second kind on purpose. It exists only for
this ablation and is named so it cannot be mistaken for a feature - nothing in
the pipeline imports it, and `train_click_counts` in `rerank.py` raises if it is
ever pointed at a non-train split.

THE CLEANEST ARM
`popularity` run twice, once from train-window counts and once from val-window
counts. Identical algorithm, identical candidates, identical everything except
whether the clicks it counts had happened yet. That isolates the variable
exactly, which no blended arm can.

BLENDING, AND WHY IT IS RANK-BASED
BM25 scores run 0 to ~40, cosine runs [-1, 1] and click counts run 0 to 4,316.
Adding them directly would let the widest scale win regardless of signal. Each
score vector is therefore converted to its within-impression rank, scaled to
[0, 1], before combining - which discards magnitude but keeps ordering, and
ordering is all the metrics read anyway.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import rankdata

DEFAULT_BLEND_WEIGHT = 0.5


def future_click_counts(
    evaluated_impressions: pl.DataFrame, article_ids: list[str]
) -> np.ndarray:
    """Clicks per article **within the split being evaluated**. Deliberately leaky.

    This is the mistake it models: computing item popularity over the whole
    dataset, including the evaluation window, instead of over training only. It
    is a single line's difference in a notebook and it is undetectable from the
    results, because its only symptom is that the score improves.

    Unlike `rerank.train_click_counts`, this function does **not** guard the
    split - guarding it is the whole point of the other one. It is quarantined
    here instead, in a module nothing else imports.
    """
    row_of_article = {article: i for i, article in enumerate(article_ids)}
    counts = np.zeros(len(article_ids), dtype=np.float32)
    for clicks in evaluated_impressions.get_column("clicked_article_ids").to_list():
        for article in clicks:
            row = row_of_article.get(article)
            if row is not None:
                counts[row] += 1.0
    return counts


def seen_before_feature(
    candidate_rows: list[np.ndarray], history_rows: list[np.ndarray]
) -> list[np.ndarray]:
    """1.0 where the user had already read that candidate, else 0.0.

    **Available at serving time** - it is derived from the user's own history,
    which is known before the recommendation is made, and never from clicks on
    the impression being predicted. Included as the honest counterpart to the
    leaky arm: the contrast between what a legal feature buys and what an
    illegal one buys is the actual anti-gaming argument.

    Measured before being used: such a candidate is clicked 3.5x more often than
    average on MIND (14.3% vs 4.1%) and 0.49x as often on EB-NeRD (4.1% vs
    8.4%) - real signal, in opposite directions on the two datasets.
    """
    if len(candidate_rows) != len(history_rows):
        raise ValueError(
            f"{len(candidate_rows)} candidate lists but {len(history_rows)} histories"
        )
    out = []
    for candidates, history in zip(candidate_rows, history_rows):
        seen = np.isin(candidates, history)
        out.append(seen.astype(np.float32))
    return out


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Within-impression ranks scaled to [0, 1], ties averaged.

    A one-candidate impression has no spread to normalise, so it returns 0.5 -
    the neutral value - rather than dividing by zero.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size <= 1:
        return np.full(scores.shape, 0.5)
    ranks = rankdata(scores, method="average")
    return (ranks - 1.0) / (scores.size - 1.0)


def blend(
    base: list[np.ndarray],
    feature: list[np.ndarray],
    weight: float = DEFAULT_BLEND_WEIGHT,
) -> list[np.ndarray]:
    """`rank_normalise(base) + weight * feature`, per impression.

    The base is rank-normalised so the feature's contribution is on a
    comparable scale; the feature is used as given, because both features here
    are already bounded (0/1, or rank-normalised by the caller).

    `weight` is a stated default of 0.5, not a tuned value: at that setting the
    feature can move a candidate by half the full width of the base ordering,
    which is enough to show an effect without letting it simply overwrite the
    base. Tuning it would make this a system-design exercise rather than an
    ablation.
    """
    if len(base) != len(feature):
        raise ValueError(f"{len(base)} base lists but {len(feature)} feature lists")
    out = []
    for b, f in zip(base, feature):
        if len(b) != len(f):
            raise ValueError("base and feature disagree on candidate count")
        out.append((rank_normalise(b) + weight * np.asarray(f, dtype=np.float64)))
    return out
