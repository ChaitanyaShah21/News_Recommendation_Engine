"""article features: freshness (D34c) and label-free exposure popularity (D34b).

Both are computed per (impression, candidate) row, from facts strictly before
the impression - and neither reads a click label, so both can be computed on the
Codabench testset exactly as on train/val.

Popularity is a SHARE, never a count: the feature store is a 5% user sample and
the leaderboard is the full population, so a raw count would be ~20x larger at
submission time than at training time (D34b).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from newsrec.retrieval.availability import first_seen_times

EXPOSURE_WINDOWS_HOURS = (1, 24)  # D34b


def candidate_rows(impressions: pl.DataFrame) -> pl.DataFrame:
    """One row per (impression, candidate): impression_id, timestamp, article_id.

    A candidate listed twice in the same impression becomes one row - it was
    shown once, and must count once in the exposure share.
    """
    return (
        impressions.select("impression_id", "timestamp", "candidate_article_ids")
        .explode("candidate_article_ids", empty_as_null=True)  # pinned, as availability.py
        .rename({"candidate_article_ids": "article_id"})
        .drop_nulls("article_id")
        .unique(subset=["impression_id", "article_id"], maintain_order=True)
    )


def freshness_hours(
    rows: pl.DataFrame, articles: pl.DataFrame, all_impressions: pl.DataFrame
) -> pl.DataFrame:
    """Hours between each candidate's earliest evidence of existence and the impression.

    Earliest evidence = min(published_time, first_seen) (D34c). published_time
    is null for every MIND article, so MIND reduces to first_seen. first_seen is
    taken over `all_impressions` - every split - which is safe: every row here is
    a candidate AT its own timestamp, so its earliest appearance is at or before
    that timestamp by construction. The final check makes that an assertion.
    """
    seen = first_seen_times(all_impressions)
    out = (
        rows.join(articles.select("article_id", "published_time"), on="article_id", how="left")
        .join(seen, on="article_id", how="left")
        .with_columns(
            pl.min_horizontal("published_time", "first_seen").alias("earliest_evidence")
        )
        .with_columns(
            ((pl.col("timestamp") - pl.col("earliest_evidence")).dt.total_seconds() / 3600.0)
            .alias("freshness_hours")
        )
    )
    if out["freshness_hours"].null_count():
        raise ValueError(
            f"{out['freshness_hours'].null_count()} candidates have no first_seen - "
            "all_impressions must include the impressions these rows came from"
        )
    if (out["freshness_hours"] < 0).any():
        raise ValueError("negative freshness: an article is dated after the impression showing it")
    return out.select("impression_id", "article_id", "freshness_hours")


def exposure_shares(
    rows: pl.DataFrame,
    all_impressions: pl.DataFrame,
    windows_hours: tuple[int, ...] = EXPOSURE_WINDOWS_HOURS,
) -> pl.DataFrame:
    """Share of impressions in [T - w, T) that showed the candidate (D34b).

    Strictly before T: an impression in the same second is excluded, so the
    current impression never counts itself. Denominator and numerator both come
    from `all_impressions` (every split, one dataset, label-free). A window with
    no impressions at all gives null, not 0 - "no data" is not "never shown".

    Vectorised with searchsorted over one int64 key per (article, second):
        key = article_index * SPAN + seconds_since_start
    so "entries of article a with time in [lo, hi)" is a difference of two
    searchsorted positions. SPAN exceeds the whole time range, so one article's
    keys can never reach another's - PROVIDED lo is clamped at 0. An unclamped
    window starting before the data would spill into the previous article's key
    range and silently borrow its counts; that boundary has its own test.
    """
    shown = candidate_rows(all_impressions)
    t0 = all_impressions["timestamp"].min()

    def secs(col: pl.Series) -> np.ndarray:
        return ((col - t0).dt.total_seconds()).to_numpy().astype(np.int64)

    # Denominator: every impression's time, sorted.
    imp_t = np.sort(secs(all_impressions["timestamp"]))
    span = int(imp_t.max()) + max(windows_hours) * 3600 + 1

    # Numerator: one key per (article, impression) showing it, sorted.
    vocab = {a: i for i, a in enumerate(shown["article_id"].unique().sort().to_list())}
    shown_key = np.sort(
        np.fromiter((vocab[a] for a in shown["article_id"]), np.int64, shown.height) * span
        + secs(shown["timestamp"])
    )

    q_t = secs(rows["timestamp"])
    q_a = np.fromiter((vocab.get(a, -1) for a in rows["article_id"]), np.int64, rows.height)
    if (q_a < 0).any():
        raise ValueError("a candidate never appears in all_impressions - pass every split")

    out = {"impression_id": rows["impression_id"], "article_id": rows["article_id"]}
    for w in windows_hours:
        lo = np.maximum(q_t - w * 3600, 0)  # the clamp that keeps keys inside one article
        denom = np.searchsorted(imp_t, q_t, "left") - np.searchsorted(imp_t, lo, "left")
        numer = (np.searchsorted(shown_key, q_a * span + q_t, "left")
                 - np.searchsorted(shown_key, q_a * span + lo, "left"))
        share = np.where(denom > 0, numer / np.maximum(denom, 1), np.nan)
        out[f"exposure_share_{w}h"] = pl.Series(share).fill_nan(None)
    return pl.DataFrame(out)
