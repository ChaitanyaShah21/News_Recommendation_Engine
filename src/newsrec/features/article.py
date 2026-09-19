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
    return freshness_from_seen(rows, articles, first_seen_times(all_impressions))


def freshness_from_seen(
    rows: pl.DataFrame, articles: pl.DataFrame, seen: pl.DataFrame, strict: bool = True
) -> pl.DataFrame:
    """`freshness_hours` with first_seen precomputed, so a chunked caller (the
    leaderboard, 206M rows) computes it once rather than per chunk.

    strict=False is for rows that are deliberately NOT in the context the
    first_seen table came from (EB-NeRD's synthetic beyond-accuracy rows, D38):
    the impression itself is then taken as evidence the article exists at T,
    so freshness is floored at 0 rather than raising. With strict=True the
    context contains every row's own impression, so that floor is a no-op and
    the checks below stay armed.
    """
    evidence = ["published_time", "first_seen"] + ([] if strict else ["timestamp"])
    out = (
        rows.join(articles.select("article_id", "published_time"), on="article_id", how="left")
        .join(seen, on="article_id", how="left")
        .with_columns(pl.min_horizontal(*evidence).alias("earliest_evidence"))
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
    """
    return ExposureIndex.build(all_impressions, windows_hours).shares(rows)


class ExposureIndex:
    """The sorted arrays behind `exposure_shares`, built once and queried per chunk.

    Vectorised with searchsorted over one int64 key per (article, second):
        key = article_index * SPAN + seconds_since_start
    so "entries of article a with time in [lo, hi)" is a difference of two
    searchsorted positions. SPAN exceeds the whole time range, so one article's
    keys can never reach another's - PROVIDED lo is clamped at 0. An unclamped
    window starting before the data would spill into the previous article's key
    range and silently borrow its counts; that boundary has its own test.

    `build` accepts a LazyFrame and never materialises the exploded
    (impression, article) strings: at leaderboard scale that is ~220M rows,
    ~10 GB as strings, but 1.8 GB as the one int64 key column kept here.
    """

    def __init__(self, imp_t, shown_key, vocab, t0, span, windows_hours):
        self.imp_t, self.shown_key, self.vocab = imp_t, shown_key, vocab
        self.t0, self.span, self.windows_hours = t0, span, windows_hours

    @classmethod
    def build(cls, impressions, windows_hours=EXPOSURE_WINDOWS_HOURS) -> "ExposureIndex":
        lf = impressions.lazy()
        t0 = lf.select(pl.col("timestamp").min()).collect().item()
        secs = (pl.col("timestamp") - t0).dt.total_seconds().cast(pl.Int64)

        # Denominator: every impression's time, sorted.
        imp_t = (lf.select(secs.alias("t")).sort("t").collect(engine="streaming")
                 ["t"].to_numpy())
        span = int(imp_t.max()) + max(windows_hours) * 3600 + 1

        # Numerator: one key per (article, impression) showing it, sorted. A
        # candidate listed twice in one impression counts once (list.unique),
        # which equals candidate_rows' dedup because impression ids are unique.
        shown = (lf.select(secs.alias("t"),
                           pl.col("candidate_article_ids").list.unique().alias("article_id"))
                   .explode("article_id", empty_as_null=True).drop_nulls("article_id"))
        vocab_list = (shown.select(pl.col("article_id").unique()).collect(engine="streaming")
                      ["article_id"].sort().to_list())
        vocab = {a: i for i, a in enumerate(vocab_list)}
        vocab_df = pl.DataFrame({"article_id": vocab_list,
                                 "_a": pl.Series(range(len(vocab_list)), dtype=pl.Int64)})
        shown_key = (shown.join(vocab_df.lazy(), on="article_id", how="inner")
                     .select((pl.col("_a") * span + pl.col("t")).alias("k"))
                     .sort("k").collect(engine="streaming")["k"].to_numpy())
        return cls(imp_t, shown_key, vocab, t0, span, tuple(windows_hours))

    def shares(self, rows: pl.DataFrame, strict: bool = True) -> pl.DataFrame:
        """strict=False: a candidate never shown in the context has share 0 (it
        was shown to nobody before T) instead of raising - for rows kept out of
        the context on purpose (D38). Strict mode refuses, because there it
        means the caller passed the wrong context."""
        q_t = ((rows["timestamp"] - self.t0).dt.total_seconds()).to_numpy().astype(np.int64)
        q_a = np.fromiter((self.vocab.get(a, -1) for a in rows["article_id"]), np.int64, rows.height)
        unseen = q_a < 0
        if unseen.any() and strict:
            raise ValueError("a candidate never appears in all_impressions - pass every split")
        if (q_t < 0).any():
            raise ValueError("a row predates the whole context - its window cannot be read")
        q_a = np.where(unseen, 0, q_a)

        out = {"impression_id": rows["impression_id"], "article_id": rows["article_id"]}
        for w in self.windows_hours:
            lo = np.maximum(q_t - w * 3600, 0)  # the clamp that keeps keys inside one article
            denom = (np.searchsorted(self.imp_t, q_t, "left")
                     - np.searchsorted(self.imp_t, lo, "left"))
            numer = (np.searchsorted(self.shown_key, q_a * self.span + q_t, "left")
                     - np.searchsorted(self.shown_key, q_a * self.span + lo, "left"))
            numer = np.where(unseen, 0, numer)
            share = np.where(denom > 0, numer / np.maximum(denom, 1), np.nan)
            out[f"exposure_share_{w}h"] = pl.Series(share).fill_nan(None)
        return pl.DataFrame(out)
