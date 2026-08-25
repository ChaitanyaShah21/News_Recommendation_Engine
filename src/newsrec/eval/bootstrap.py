"""bootstrap 95% confidence intervals.

WHAT A BOOTSTRAP IS DOING
We have one validation set and one number from it. The question a confidence
interval answers is "if we had drawn a different sample of impressions from the
same process, how different would that number be?" - and we cannot answer it by
collecting more data. The bootstrap answers it by treating our sample as a stand-
in for the population: draw a new sample *of the same size, with replacement*
from the impressions we have, recompute the metric, and repeat. The spread of
those recomputed values estimates the spread of the real one.

Resampling is done over **impressions**, which is the unit D18 already chose for
every metric in this project and the unit both leaderboards score. Resampling
clicks or users instead would give a different (and, for our purposes, wrong)
interval, because it would answer a question about a different population.

ONE INDEX DRAW, SHARED ACROSS METRICS
Every metric in a run is recomputed on the *same* resampled impressions rather
than on independent draws. It costs nothing and it means a later comparison
between two metrics, or between two methods, sees the same resampled worlds -
without which a difference-of-means interval would be wrong.

NaN IS EXCLUDED, NOT TREATED AS ZERO
Undefined metric values (an impression with no clicks, a retrieval list too
short to have pairs) are dropped inside each resample rather than counted as 0,
matching `macro_mean`. A resample containing nothing but NaN yields NaN rather
than a warning and a fabricated number.

COVERAGE IS NOT A MEAN, SO IT NEEDS ITS OWN FUNCTION
Diversity and novelty are per-list values whose statistic is a mean. Coverage is
a property of the union of the whole run: the mean of per-list coverages is not
the coverage of the union (there is a test pinning exactly that). So it must be
*recomputed* inside each resample from the union of the resampled lists, which
`bootstrap_coverage` does.

METHOD
Percentile interval: the 2.5th and 97.5th percentiles of the resampled statistic.
Simple, assumption-light and standard. It is not bias-corrected (BCa would be),
which matters mainly for strongly skewed statistics - worth naming in the design
note rather than implying more rigour than the method has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 20260825


@dataclass
class Interval:
    point: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"

    def pivotal(self) -> tuple[float, float]:
        """The basic (pivotal) interval: [2*point - high, 2*point - low].

        Reflects the bootstrap distribution through the point estimate, which is
        the textbook correction when that distribution is location-shifted
        rather than centred. Computed for coverage and carried in the reports,
        but **not** used as coverage's headline interval - see
        `bootstrap_coverage` for why coverage is reported without one.
        """
        return 2 * self.point - self.high, 2 * self.point - self.low


def _percentiles(alpha: float) -> tuple[float, float]:
    return 100 * (alpha / 2), 100 * (1 - alpha / 2)


def bootstrap_mean(
    values: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    indices: np.ndarray | None = None,
) -> Interval:
    """Percentile CI for the mean of `values`, ignoring NaN.

    Args:
        indices: an optional pre-drawn (n_resamples x n) array of resample
            indices, so several metrics can share one draw. When omitted a
            fresh one is drawn from `seed`.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = ~np.isnan(values)
    n = values.size
    if n == 0 or not defined.any():
        return Interval(float("nan"), float("nan"), float("nan"), 0)

    point = float(values[defined].mean())
    if indices is None:
        indices = draw_indices(n, n_resamples, seed)

    resampled = values[indices]
    ok = ~np.isnan(resampled)
    counts = ok.sum(axis=1)
    totals = np.where(ok, resampled, 0.0).sum(axis=1)
    # A resample in which every drawn impression was undefined has no mean;
    # np.where keeps it NaN instead of dividing by zero.
    means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    lo_p, hi_p = _percentiles(alpha)
    usable = means[~np.isnan(means)]
    if usable.size == 0:
        return Interval(point, float("nan"), float("nan"), int(defined.sum()))
    return Interval(
        point,
        float(np.percentile(usable, lo_p)),
        float(np.percentile(usable, hi_p)),
        int(defined.sum()),
    )


def draw_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples x n) resample indices, drawn with replacement."""
    if n <= 0:
        raise ValueError(f"cannot resample {n} items")
    rng = np.random.default_rng(seed)
    # int32, not int64: 1,000 resamples over MIND's 51,205 impressions is 205 MB
    # instead of 410 MB, and this array is held while every metric reuses it.
    return rng.integers(0, n, size=(n_resamples, n), dtype=np.int32)


def bootstrap_coverage(
    lists: list[np.ndarray],
    catalogue_size: int,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    indices: np.ndarray | None = None,
) -> Interval:
    """Coverage, plus the spread of coverage across resamples.

    **The returned `low`/`high` are NOT a valid 95% confidence interval, and are
    deliberately not reported as one (D27).** They are kept because they are the
    honest raw output and the reports carry them as clearly-named columns.

    Why the bootstrap fails here, when it works for every other metric in this
    project: drawing n items with replacement from n items yields only **63.2%**
    distinct items (1 - 1/e; measured at n = 1,562, 17,749 and 50,000 - all
    63.2%). For a *mean* that is harmless, because a duplicated impression still
    contributes its value and the mean scatters around the truth unbiased. For
    coverage it is fatal: coverage is the size of a *union*, a duplicate
    contributes nothing new, so every resample can only lose articles. The whole
    bootstrap distribution shifts downward and the percentile interval ends up
    entirely below the point estimate.

    The bias is a property of the statistic, not of this code. What these
    numbers actually measure is "coverage when a third of the users are thrown
    away", which is a different and smaller quantity than "how much would
    coverage wobble".

    There is also a question of what coverage even estimates. "Our system
    surfaced 31.1% of the catalogue to these 50,000 users" is a **census** - we
    counted it, and there is no sampling error to report. The other metrics are
    per-impression averages, where "what if we had drawn different impressions?"
    is a real question.

    The union is computed with a boolean scratch array rather than a Python set:
    marking `seen[flat] = True` and summing is O(list length) per resample with
    no hashing, which is what makes 1,000 resamples over 51,205 lists tractable.
    The array is refilled rather than reallocated for the same reason.
    """
    if catalogue_size <= 0:
        raise ValueError(f"catalogue_size must be positive, got {catalogue_size}")
    n = len(lists)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)

    # Flatten once: `flat[start[i]:start[i+1]]` is list i's article rows.
    lengths = np.fromiter((len(x) for x in lists), dtype=np.int64, count=n)
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=start[1:])
    flat = (
        np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
        if start[-1] > 0
        else np.zeros(0, dtype=np.int64)
    )

    seen = np.zeros(catalogue_size, dtype=bool)
    seen[flat] = True
    point = float(seen.sum() / catalogue_size)

    if indices is None:
        indices = draw_indices(n, n_resamples, seed)

    values = np.empty(len(indices), dtype=np.float64)
    for r in range(len(indices)):
        draw = indices[r]
        seen[:] = False
        # Ragged gather, vectorised. The obvious inner `for i in draw` loop is
        # 51,205 Python iterations per resample - 51 million over 1,000
        # resamples, which does not finish. Instead build every gather position
        # at once: each drawn list contributes `lengths[i]` consecutive
        # positions starting at `start[i]`, so
        #     repeat(start[draw], lens) + (arange(total) - repeat(list_start, lens))
        # walks 0..len-1 within each drawn list's own span.
        lens = lengths[draw]
        total = int(lens.sum())
        if total:
            list_start = np.cumsum(lens) - lens
            within = np.arange(total, dtype=np.int64) - np.repeat(list_start, lens)
            seen[flat[np.repeat(start[draw], lens) + within]] = True
        # Repeated draws mark the same articles again, which is correct: a union
        # does not double-count.
        values[r] = seen.sum() / catalogue_size

    lo_p, hi_p = _percentiles(alpha)
    return Interval(point, float(np.percentile(values, lo_p)), float(np.percentile(values, hi_p)), n)
