"""click-history features with multi-scale recency decay (D35).

Three scales per dataset - short, medium, infinite - become separate features
and the re-ranker learns their mix. Ages are measured from the user's *newest*
history click, not from the impression:

    weight_i = 0.5 ** ((t_newest - t_i) / half_life)

A normalised weighted average is unchanged when every weight is scaled by the
same factor, so this equals measuring from the impression time T - which
cancels. Two consequences (D35): the decayed profile is per user rather than
per impression, and it cannot underflow, because the newest click always has
weight exactly 1. Measured from T instead, a 600-hour-old click at a 1-hour
half-life weighs 0.5**600 ~ 1e-181, and a little further every weight is 0.0
and the average is 0/0.

Time away from the site is the one thing that cancellation discards, so it is
recovered as its own per-impression feature, `hours_since_last_click` - which
is also where the no-future-click guard lives.

Age is time on EB-NeRD (hours) and list position on MIND, whose history
timestamps are 100% null. The two are NOT comparable across datasets.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import sparse

from newsrec.retrieval.semantic_search import MIN_NORM, UserVectors

# D31: the shipped re-ranking system pools the last 100 clicks. NOT the library's
# DEFAULT_N_RECENT (10, D12), which was tuned for retrieval - importing that would
# make the 'infinite half-life equals A1' claim compare against the wrong A1.
N_RECENT = 100

# D35. Units: hours for EB-NeRD, list positions for MIND. math.inf is the plain
# mean, i.e. exactly A1's user vector.
HALF_LIVES = {
    "ebnerd": {"short": 24.0, "medium": 168.0, "inf": math.inf},
    "mind": {"short": 3.0, "medium": 10.0, "inf": math.inf},
}


def decay_weights(ages: np.ndarray, half_life: float) -> np.ndarray:
    """Weights 0.5**((age - min age) / half_life), so the newest item gets exactly 1.

    Raises on a negative or non-finite age. An age can only be negative if a
    history click is dated after the thing it is being compared to, which is the
    signature of a leak (joining the wrong history snapshot, Landmine 5). Clipping
    to zero would hide that and hand the model the answer, so it is refused.
    """
    ages = np.asarray(ages, dtype=np.float64)
    if ages.size == 0:
        return ages
    if not np.all(np.isfinite(ages)) or np.any(ages < 0):
        raise ValueError(f"ages must be finite and >= 0; got min={np.nanmin(ages)!r}")
    if not half_life > 0:
        raise ValueError(f"half_life must be > 0 (math.inf for no decay), got {half_life!r}")
    if math.isinf(half_life):
        return np.ones_like(ages)
    return np.power(0.5, (ages - ages.min()) / half_life)


def _tail_with_ages(
    article_ids: list[str] | None,
    timestamps: list | None,
    n_recent: int,
) -> tuple[list[str], np.ndarray]:
    """The last `n_recent` clicks (history is oldest-first) and each one's age.

    Age is hours before the newest click when timestamps exist (EB-NeRD), else
    list position counted back from the newest (MIND: timestamps 100% null).
    """
    ids = list(article_ids or [])
    if not ids:
        return [], np.zeros(0)
    ids = ids[-n_recent:]
    if timestamps is not None:
        ts = list(timestamps)[-n_recent:]
        if len(ts) != len(ids) or any(t is None for t in ts):
            raise ValueError("history_timestamps must align 1:1 with history_article_ids")
        newest = max(ts)
        ages = np.array([(newest - t).total_seconds() / 3600.0 for t in ts])
    else:
        ages = np.arange(len(ids) - 1, -1, -1, dtype=np.float64)
    return ids, ages


def _dedupe_keep_newest(ids: list[str], ages: np.ndarray) -> tuple[list[str], np.ndarray]:
    """An article clicked twice counts once, at its most recent click.

    A1's user vector de-duplicates too (an article re-read must not be counted
    twice), so this keeps the infinite scale identical to it. Keeping the
    *newest* occurrence is the choice that matters under decay: the older copy
    would under-weight an article the user has just come back to.
    """
    best: dict[str, float] = {}
    for a, age in zip(ids, ages):
        if a not in best or age < best[a]:
            best[a] = float(age)
    return list(best), np.array(list(best.values()))


def build_decayed_user_vectors(
    history: pl.DataFrame,
    article_ids: list[str],
    embeddings: np.ndarray,
    half_life: float,
    n_recent: int = N_RECENT,
) -> UserVectors:
    """Recency-weighted mean of each user's last `n_recent` clicked embeddings.

    Identical in shape and guards to A1's `semantic_search.build_user_vectors`,
    with decay weights in place of its equal 1/n weights. At half_life=math.inf
    every weight is equal and the result reproduces A1's vectors exactly - the
    test that pins it. `history` is one dataset and one split.
    """
    row_of_article = {aid: i for i, aid in enumerate(article_ids)}
    user_ids = history.get_column("user_id").to_list()
    histories = history.get_column("history_article_ids").to_list()
    stamps = (
        history.get_column("history_timestamps").to_list()
        if "history_timestamps" in history.columns
        else [None] * len(user_ids)
    )

    indptr = np.zeros(len(user_ids) + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[float] = []
    for i, (ids, ts) in enumerate(zip(histories, stamps)):
        tail, ages = _tail_with_ages(ids, ts, n_recent)
        tail, ages = _dedupe_keep_newest(tail, ages)
        keep = [(row_of_article[a], age) for a, age in zip(tail, ages) if a in row_of_article]
        if keep:
            rows = np.array([r for r, _ in keep], dtype=np.int32)
            w = decay_weights(np.array([age for _, age in keep]), half_life)
            order = np.argsort(rows)  # CSR wants sorted column indices per row
            indices.extend(rows[order].tolist())
            data.extend((w[order] / w.sum()).tolist())
        indptr[i + 1] = len(indices)

    selection = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), indptr),
        shape=(len(user_ids), embeddings.shape[0]),
        dtype=np.float32,
    )
    pooled = np.asarray(selection @ embeddings, dtype=np.float32)

    # Same three-way guard as A1: no articles, exact cancellation, and
    # near-cancellation that `norms > 0` would miss. See semantic_search.
    norms = np.linalg.norm(pooled, axis=1)
    has_query = norms > MIN_NORM
    safe = np.where(has_query, norms, 1.0).astype(np.float32)
    matrix = (pooled / safe[:, None]).astype(np.float32)
    matrix[~has_query] = 0.0
    return UserVectors(user_ids=user_ids, matrix=matrix, has_query=has_query)


def user_category_shares(
    history: pl.DataFrame,
    article_category: dict[str, str],
    half_life: float,
    n_recent: int = N_RECENT,
) -> pl.DataFrame:
    """Long table (user_id, category, share): the recency-weighted share of each
    user's recent clicks that went to each category. Shares sum to 1 per user.

    Joined later on (user_id, candidate's category); a category the user never
    clicked has no row and becomes share 0. Built only from the user's own
    history, never from catalogue-wide counts (Landmine 6).
    """
    users: list[str] = []
    cats: list[str] = []
    shares: list[float] = []
    stamps = (
        history.get_column("history_timestamps").to_list()
        if "history_timestamps" in history.columns
        else [None] * history.height
    )
    for user, ids, ts in zip(
        history.get_column("user_id").to_list(),
        history.get_column("history_article_ids").to_list(),
        stamps,
    ):
        tail, ages = _tail_with_ages(ids, ts, n_recent)
        tail, ages = _dedupe_keep_newest(tail, ages)
        pairs = [(article_category[a], age) for a, age in zip(tail, ages) if a in article_category]
        if not pairs:
            continue
        w = decay_weights(np.array([age for _, age in pairs]), half_life)
        total: dict[str, float] = {}
        for (cat, _), wi in zip(pairs, w):
            total[cat] = total.get(cat, 0.0) + wi
        norm = w.sum()
        for cat, v in total.items():
            users.append(user)
            cats.append(cat)
            shares.append(v / norm)
    return pl.DataFrame(
        {"user_id": users, "category": cats, "share": shares},
        schema={"user_id": pl.Utf8, "category": pl.Utf8, "share": pl.Float64},
    )


def hours_since_last_click(impressions: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """Per impression: hours between the user's newest history click and the
    impression. EB-NeRD only (MIND has no history timestamps).

    This is the per-impression half of D35 (the decayed profile cannot see time
    away), and it is where the behaviour-window boundary is enforced. Raises if
    ANY history click is at or after its impression: a tie is refused as well as
    a future click, the conservative direction (D34a's rule). `impressions` and
    `history` must be one dataset and one split, so the join is on the right
    snapshot (Landmine 5).
    """
    for frame, name in ((impressions, "impressions"), (history, "history")):
        if frame.select(pl.col("split").n_unique()).item() > 1:
            raise ValueError(f"{name} spans more than one split; join per split (Landmine 5)")
    newest = history.select(
        "user_id", pl.col("history_timestamps").list.max().alias("newest_click")
    )
    out = impressions.select("impression_id", "user_id", "timestamp").join(
        newest, on="user_id", how="left"
    )
    bad = out.filter(pl.col("newest_click") >= pl.col("timestamp"))
    if bad.height:
        raise ValueError(
            f"{bad.height} impressions have a history click at or after the impression "
            f"(first: {bad.row(0, named=True)}) - future-click leakage"
        )
    return out.select(
        "impression_id",
        ((pl.col("timestamp") - pl.col("newest_click")).dt.total_seconds() / 3600.0).alias(
            "hours_since_last_click"
        ),
    )
