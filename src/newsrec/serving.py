"""D44: the online scoring path - one request, no DataFrame.

WHY THIS EXISTS
Q4 measured the shipped pipeline at p50 237 ms (MIND) / 452 ms (EB-NeRD) per
request against an example SLA of p99 < 100 ms, and the design note's answer was
a projection rather than a build. Profiling said exactly where the time goes,
and it is not the arithmetic:

    66 separate Polars query collections per request, for 100 candidates.

`features/assemble.build_rows` is a batch pipeline. It is the right shape for
the 206 million candidate rows of a leaderboard run, where a fixed per-query
cost of a few milliseconds disappears into work that lasts minutes. Served one
request at a time it pays that fixed cost 66 times over on tables of 100 rows,
and the dataframe overhead - not the dot products, not the searchsorted, not the
trees - IS the latency. cProfile over 20 real requests: 52% of the time inside
`PyLazyFrame.collect`, 23% in joins, and joins against 65,238-row article tables
to decorate 100 candidates.

WHAT THIS MODULE CHANGES, AND WHAT IT DELIBERATELY DOES NOT
It changes the data structures, not one feature definition. Every value here is
required to equal what `build_rows` produces for the same request - that is the
test, run against real data, and it is the only reason this is safe to serve.

Three kinds of work are separated, which is the actual insight:

  1. CORPUS-LIFETIME, built once at process start and shared by every request:
     article embeddings, the BM25 index, the exposure index, earliest-evidence
     times, the category map. Already amortised in the Q4 benchmark; here they
     are additionally flattened into arrays indexed by article row, so a lookup
     is `a[rows]` rather than a join against a 65,238-row frame.

  2. PER-USER, and precomputable offline: the three decayed profile vectors, the
     three category-share tables and the BM25 query. These depend only on the
     user's click history, never on the request - the Q4 benchmark measured them
     at 13.5% (MIND) / 16.8% (EB-NeRD) of the feature stage. They are built HERE
     by calling the existing, tested builders in `features/history.py` and
     `retrieval/bm25_search.py`, not reimplemented: a second implementation of a
     decay weight is a second thing that can be wrong.

  3. PER-REQUEST, which is all that is left: gather rows, six dot products, two
     searchsorted pairs, and the rack normalisation. Pure NumPy, no Polars.

WHAT THIS DOES NOT FIX
The per-user work still has to happen somewhere. `UserProfile.build` is the
nightly job, and a user whose profile is stale gets stale features - which is a
freshness/latency trade-off a real system has to make explicitly, and is stated
in SCALE_NOTES.md rather than hidden behind the improved number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

from newsrec.features import history as hf
from newsrec.features import rack as rk
from newsrec.features.assemble import ALL_FEATURES
from newsrec.retrieval import bm25_search

SECONDS_PER_HOUR = 3600.0


# --------------------------------------------------------------- corpus tier

@dataclass
class ServingIndex:
    """Corpus-lifetime state, flattened to arrays indexed by article row.

    Every per-article array is indexed by the SAME row space - the embedding
    matrix's - so one `art_idx` lookup serves all of them. The BM25 index and
    the exposure index have their own internal orderings; `bm25_row` and
    `exp_row` translate, with -1 for an article that ordering does not contain.
    Mapping them once here is what removes the joins from the request path.
    """

    dataset: str
    art_idx: dict[str, int]
    emb: np.ndarray            # (n_art, dim) float32, unit norm
    cat_id: np.ndarray         # (n_art,) int32, -1 = no category
    n_categories: int
    earliest_s: np.ndarray     # (n_art,) float64, epoch seconds, NaN = unknown
    bm25_row: np.ndarray       # (n_art,) int64 into doc_term, -1 = absent
    doc_term: sparse.csr_matrix
    exp_row: np.ndarray        # (n_art,) int64 into the exposure vocabulary, -1
    exposure: object           # features.article.ExposureIndex
    features: list[str]

    @classmethod
    def build(cls, ctx) -> "ServingIndex":
        """From the batch `assemble.Context`, so the two cannot drift apart:
        there is one definition of the corpus state and this is a view of it."""
        art_idx = ctx.art_idx
        n = len(ctx.emb_ids)

        cats = sorted({c for c in ctx.category.values() if c is not None})
        cat_of = {c: i for i, c in enumerate(cats)}
        cat_id = np.full(n, -1, np.int32)
        for a, c in ctx.category.items():
            if c is not None and a in art_idx:
                cat_id[art_idx[a]] = cat_of[c]

        # Earliest evidence, as epoch seconds: min(published_time, first_seen),
        # exactly D34c. NaN marks an article with neither, which `features`
        # refuses rather than silently scoring - the same refusal
        # `freshness_from_seen` makes.
        earliest = np.full(n, np.nan, np.float64)
        seen = ctx.first_seen.join(
            ctx.articles.select("article_id", "published_time"), on="article_id", how="left"
        ).with_columns(
            pl.min_horizontal("published_time", "first_seen").alias("e")
        ).select("article_id", pl.col("e").dt.epoch("s").cast(pl.Float64))
        for a, e in zip(seen["article_id"], seen["e"]):
            if a in art_idx and e is not None:
                earliest[art_idx[a]] = e

        bm25_row = np.full(n, -1, np.int64)
        for a, i in ctx.b_idx.items():
            if a in art_idx:
                bm25_row[art_idx[a]] = i
        exp_row = np.full(n, -1, np.int64)
        for a, i in ctx.exposure.vocab.items():
            if a in art_idx:
                exp_row[art_idx[a]] = i

        return cls(dataset=ctx.dataset, art_idx=art_idx, emb=ctx.emb, cat_id=cat_id,
                   n_categories=len(cats), earliest_s=earliest, bm25_row=bm25_row,
                   doc_term=ctx.index.doc_term, exp_row=exp_row, exposure=ctx.exposure,
                   features=list(ALL_FEATURES[ctx.dataset]))


# ----------------------------------------------------------------- user tier

@dataclass
class UserProfile:
    """One user's precomputable state. `None` means "this user has no usable
    query at this scale", which becomes a null feature - never a zero. A cold
    user and a user whose profile happens to average to nothing are the same
    fact to the model and a different fact from "score 0".
    """

    vec: dict[str, np.ndarray | None]        # scale -> (dim,) unit-norm
    cat_share: dict[str, np.ndarray | None]  # scale -> (n_categories,)
    bm25_q: sparse.csr_matrix | None         # (1, vocab)
    extra: dict                              # EB-NeRD's impression-level values

    @classmethod
    def build_many(cls, index: ServingIndex, ctx, history: pl.DataFrame
                   ) -> dict[str, "UserProfile"]:
        """The nightly job: every user's profile, from the existing builders.

        Batched on purpose. Per user these builders cost ~30 ms (measured, Q4);
        over a whole split they cost a few seconds, because the expensive parts
        are matrix products that vectorise. That asymmetry is the argument for
        precomputing at all, and it is a measurement rather than a belief.
        """
        scales = list(hf.HALF_LIVES[index.dataset])
        users = history["user_id"].to_list()
        out = {u: cls({}, {}, None, {}) for u in users}

        for scale in scales:
            half = hf.HALF_LIVES[index.dataset][scale]
            uv = hf.build_decayed_user_vectors(history, ctx.emb_ids, ctx.emb, half)
            for i, u in enumerate(uv.user_ids):
                out[u].vec[scale] = uv.matrix[i] if uv.has_query[i] else None

            shares = hf.user_category_shares(history, ctx.category, half)
            cat_of = {}
            for a, c in ctx.category.items():
                if c is not None and a in index.art_idx:
                    cat_of[c] = int(index.cat_id[index.art_idx[a]])
            dense: dict[str, np.ndarray] = {}
            for u, c, v in zip(shares["user_id"], shares["category"], shares["share"]):
                arr = dense.setdefault(u, np.zeros(index.n_categories, np.float64))
                if c in cat_of:
                    arr[cat_of[c]] = v
            for u in users:
                out[u].cat_share[scale] = dense.get(u)  # None = user unknown here

        q = bm25_search.build_queries(history, ctx.index, ctx.title_term, n_recent=hf.N_RECENT)
        for i, u in enumerate(q.user_ids):
            out[u].bm25_q = q.matrix[i] if q.has_query[i] else None
        return out


# -------------------------------------------------------------- request tier

def features(index: ServingIndex, profile: UserProfile | None,
             article_ids: list[str], timestamp, extra: dict | None = None) -> np.ndarray:
    """The feature matrix for one request: (n_candidates, len(index.features)).

    Column order is `index.features`, which is `ALL_FEATURES[dataset]` - the
    same order `gbdt.feature_matrix` builds, so the model's columns line up by
    construction rather than by convention.

    `timestamp` is the moment the request is served. `extra` supplies EB-NeRD's
    impression-level values, which a serving system knows from the request
    itself (device, session) rather than computing.
    """
    rows = np.fromiter((index.art_idx.get(a, -1) for a in article_ids), np.int64,
                       len(article_ids))
    if (rows < 0).any():
        raise ValueError(f"{int((rows < 0).sum())} candidates have no embedding")
    n = len(rows)
    t_s = float(pl.Series([timestamp]).dt.epoch("s")[0])
    cols: dict[str, np.ndarray] = {}

    # --- semantic: one (dim,) . (n, dim) product per scale ---
    cand_emb = index.emb[rows]
    for scale in hf.HALF_LIVES[index.dataset]:
        v = profile.vec.get(scale) if profile else None
        cols[f"cos_{scale}"] = (cand_emb @ v).astype(np.float64) if v is not None \
            else np.full(n, np.nan)

    # --- category share: a gather out of the user's dense share vector ---
    cat = index.cat_id[rows]
    for scale in hf.HALF_LIVES[index.dataset]:
        s = profile.cat_share.get(scale) if profile else None
        if s is None:
            cols[f"cat_share_{scale}"] = np.full(n, np.nan)
        else:
            # A candidate with no category, or a category this user never
            # clicked, scores 0 - the same value the left join plus fill_null(0)
            # gives in the batch path, and different from the null above.
            cols[f"cat_share_{scale}"] = np.where(cat >= 0, s[np.clip(cat, 0, None)], 0.0)

    # --- lexical: one sparse row times the candidates' rows ---
    b = index.bm25_row[rows]
    if profile is None or profile.bm25_q is None:
        cols["bm25"] = np.full(n, np.nan)
    else:
        sc = np.zeros(n)
        ok = b >= 0
        if ok.any():
            sc[ok] = np.asarray(
                index.doc_term[b[ok]].multiply(profile.bm25_q).sum(axis=1)).ravel()
        cols["bm25"] = np.where(ok, sc, np.nan)

    # --- freshness (D34c) ---
    e = index.earliest_s[rows]
    if np.isnan(e).any():
        raise ValueError(f"{int(np.isnan(e).sum())} candidates have no earliest evidence")
    fresh = (t_s - e) / SECONDS_PER_HOUR
    if (fresh < 0).any():
        raise ValueError("negative freshness: an article is dated after the request")
    cols["freshness_hours"] = fresh

    # --- exposure share (D34b), through the batch path's own numpy core ---
    q_t = np.full(n, int(t_s - float(pl.Series([index.exposure.t0]).dt.epoch("s")[0])), np.int64)
    shares = index.exposure.shares_arrays(q_t, index.exp_row[rows])
    for w, s in zip(index.exposure.windows_hours, shares):
        cols[f"exposure_share_{w}h"] = s

    # --- EB-NeRD's impression-level values, supplied by the request ---
    # A missing OR null value becomes NaN, not 0. `hours_since_last_click` is
    # genuinely null for a user with no click history, so `None` here is a real
    # serving case and not a caller error - and a zero would claim the user
    # clicked something this instant.
    for name in index.features:
        if name not in cols and not name.endswith(("_pct", "_z")):
            v = (extra or {}).get(name)
            cols[name] = np.full(n, np.nan if v is None else float(v))

    # --- D42 rack normalisation, over this one rack ---
    for f in rk.RACK_BASE:
        p, z = _rack(cols[f])
        cols[f"{f}_pct"], cols[f"{f}_z"] = p, z

    return np.stack([cols[f] for f in index.features], axis=1).astype(np.float32)


def _rack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Percentile rank and z-score over one rack, matching `features/rack.py`.

    NaN is this path's spelling of null - it is what LightGBM reads as missing,
    and what `feature_matrix` turns the batch path's nulls into - so NaNs are
    excluded from the statistics and stay NaN, and a rack with fewer than two
    finite values, or no spread, gets the neutral value rather than 0/0.
    """
    p = np.full(len(x), np.nan)
    z = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    m = int(ok.sum())
    if m == 0:
        return p, z
    v = x[ok]
    if m == 1:
        p[ok], z[ok] = rk.NEUTRAL_PCT, rk.NEUTRAL_Z
        return p, z
    # Average rank over ties, the same rule as Polars' rank("average"):
    # rank the sorted values, then average each tied run's ranks.
    order = np.argsort(v, kind="stable")
    ranks = np.empty(m, np.float64)
    s = v[order]
    i = 0
    while i < m:
        j = i
        while j + 1 < m and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    p[ok] = (ranks - 1.0) / (m - 1.0)
    sd = v.std(ddof=1)
    z[ok] = rk.NEUTRAL_Z if sd == 0 else (v - v.mean()) / sd
    return p, z
