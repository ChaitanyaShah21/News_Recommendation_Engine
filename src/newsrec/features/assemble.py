"""assemble one row per (impression, candidate) with every allowlisted feature.

The allowlist is FEATURES below - explicit, per dataset (D34). The builder ends
by asserting its output columns are exactly keys + label + that list, so a
feature cannot join the model by accident: it has to be named here.

The label (`clicked`) is carried for training and evaluation but is never a
feature, and no feature is computed from it - asserted in test_no_leakage.py by
shuffling the labels and requiring every feature to stay identical.

A1's scorers are features, not competitors: `cos_inf` IS A1's semantic score
(the infinite half-life reproduces its user vector exactly) and `bm25` is A1's
lexical score. Q2's "before" system is therefore one column of the "after".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from newsrec.features import article as af
from newsrec.features import history as hf
from newsrec.features import session as sf
from newsrec.retrieval import bm25, bm25_search

KEYS = ["impression_id", "user_id", "article_id"]
LABEL = "clicked"

_COMMON = [
    "cos_short", "cos_medium", "cos_inf",
    "cat_share_short", "cat_share_medium", "cat_share_inf",
    "bm25",
    "freshness_hours",
    "exposure_share_1h", "exposure_share_24h",
]
FEATURES = {
    "mind": _COMMON,
    "ebnerd": _COMMON + [
        "hours_since_last_click",
        "session_position", "minutes_since_session_start", "device_type",
    ],
}
CHUNK = 500_000


def _rowwise_dot(left: np.ndarray, li: np.ndarray, right: np.ndarray, ri: np.ndarray) -> np.ndarray:
    """out[k] = left[li[k]] . right[ri[k]], chunked: 14M rows x 384 floats at once is ~21 GB."""
    out = np.empty(len(li), dtype=np.float32)
    for s in range(0, len(li), CHUNK):
        e = s + CHUNK
        out[s:e] = np.einsum("ij,ij->i", left[li[s:e]], right[ri[s:e]])
    return out


def _sparse_rowwise_dot(left, li: np.ndarray, right, ri: np.ndarray) -> np.ndarray:
    out = np.empty(len(li), dtype=np.float32)
    for s in range(0, len(li), CHUNK // 5):
        e = s + CHUNK // 5
        out[s:e] = np.asarray(left[li[s:e]].multiply(right[ri[s:e]]).sum(axis=1)).ravel()
    return out


@dataclass
class Context:
    """Everything a feature row reads that is NOT specific to one chunk of target
    impressions: the article side (embeddings, BM25 index, categories) and the
    label-free log of past impressions (first_seen, exposure, sessions).

    Built once. `build_feature_table` builds it and immediately uses it, so the
    tested path and the chunked leaderboard path run the same code.
    """

    dataset: str
    articles: pl.DataFrame
    emb_ids: list[str]
    emb: np.ndarray
    art_idx: dict
    index: object
    title_term: object
    b_idx: dict
    category: dict
    first_seen: pl.DataFrame
    exposure: af.ExposureIndex
    sessions: pl.DataFrame | None


def build_context(
    dataset: str,
    context_impressions,
    articles: pl.DataFrame,
    emb_ids: list[str],
    emb: np.ndarray,
) -> Context:
    """context_impressions: DataFrame or LazyFrame of every impression whose
    existence (never its clicks) a feature may read - all splits of one dataset."""
    lf = context_impressions.lazy()
    ds_articles = articles.filter(pl.col("dataset") == dataset)
    index = bm25.build_index(ds_articles)
    sessions = None
    if dataset == "ebnerd":
        sessions = sf.session_features(
            lf.select("dataset", "impression_id", "user_id", "session_id", "timestamp",
                      "device_type").collect(engine="streaming"))
    return Context(
        dataset=dataset,
        articles=ds_articles,
        emb_ids=emb_ids,
        emb=emb,
        art_idx={a: i for i, a in enumerate(emb_ids)},
        index=index,
        title_term=bm25_search.build_title_term_matrix(ds_articles, index.vocab),
        b_idx={a: i for i, a in enumerate(index.article_ids)},
        category=dict(zip(ds_articles["article_id"], ds_articles["category"])),
        first_seen=af.first_seen_times(lf).collect(engine="streaming"),
        exposure=af.ExposureIndex.build(lf),
        sessions=sessions,
    )


def build_feature_table(
    dataset: str,
    target: pl.DataFrame,
    all_impressions: pl.DataFrame,
    history: pl.DataFrame,
    articles: pl.DataFrame,
    emb_ids: list[str],
    emb: np.ndarray,
) -> pl.DataFrame:
    """Features for every (impression, candidate) in `target`.

    target:          the impressions to featurise - one dataset, ONE split.
    all_impressions: every split of the same dataset. Exposure, freshness and
                     session features read the past from here, never labels.
    history:         the history table; only `target`'s split is used, so the
                     snapshot always matches (Landmine 5).
    """
    ctx = build_context(dataset, all_impressions, articles, emb_ids, emb)
    return build_rows(ctx, target, history)


def build_rows(ctx: Context, target: pl.DataFrame, history: pl.DataFrame,
               strict: bool = True) -> pl.DataFrame:
    """The per-chunk half of `build_feature_table`.

    strict=False only for rows deliberately left out of the context (D38): an
    article never shown before gets exposure 0 and freshness floored at T,
    instead of raising.
    """
    dataset = ctx.dataset
    splits = target["split"].unique().to_list()
    if len(splits) != 1:
        raise ValueError(f"target must be one split, got {splits} (Landmine 5)")
    # Every join below is on impression_id. A duplicated id would silently
    # multiply rows - EB-NeRD's leaderboard file has 200,000 rows sharing id 0.
    if target["impression_id"].n_unique() != target.height:
        raise ValueError("target impression_ids are not unique; give each row its own key")
    hist = history.filter(pl.col("split") == splits[0])

    rows = af.candidate_rows(target).with_row_index("_row").join(
        target.select("impression_id", "user_id"), on="impression_id", how="left"
    )
    if "clicked_article_ids" in target.columns:
        clicks = (target.select("impression_id", "clicked_article_ids")
                  .explode("clicked_article_ids", empty_as_null=True)
                  .drop_nulls("clicked_article_ids")
                  .rename({"clicked_article_ids": "article_id"})
                  .unique()
                  .with_columns(pl.lit(1, dtype=pl.Int8).alias(LABEL)))
        rows = rows.join(clicks, on=["impression_id", "article_id"], how="left").with_columns(
            pl.col(LABEL).fill_null(0))
    else:  # the unlabelled leaderboard split
        rows = rows.with_columns(pl.lit(None, dtype=pl.Int8).alias(LABEL))
    rows = rows.sort("_row")

    # --- semantic: cosine against the three decayed user vectors (D35) ---
    ai = np.fromiter((ctx.art_idx.get(a, -1) for a in rows["article_id"]), np.int64, rows.height)
    if (ai < 0).any():
        raise ValueError(f"{int((ai < 0).sum())} candidates have no embedding")
    cols: dict[str, pl.Series] = {}
    for scale, h in hf.HALF_LIVES[dataset].items():
        users = hf.build_decayed_user_vectors(hist, ctx.emb_ids, ctx.emb, h)
        u_idx = {u: i for i, u in enumerate(users.user_ids)}
        ui = np.fromiter((u_idx.get(u, -1) for u in rows["user_id"]), np.int64, rows.height)
        ok = (ui >= 0) & users.has_query[np.clip(ui, 0, None)]
        dots = _rowwise_dot(users.matrix, np.clip(ui, 0, None), ctx.emb, ai)
        cols[f"cos_{scale}"] = pl.Series(np.where(ok, dots, np.nan)).fill_nan(None)

    # --- lexical: A1's BM25 score of each candidate for the user's query ---
    queries = bm25_search.build_queries(hist, ctx.index, ctx.title_term, n_recent=hf.N_RECENT)
    q_idx = {u: i for i, u in enumerate(queries.user_ids)}
    qi = np.fromiter((q_idx.get(u, -1) for u in rows["user_id"]), np.int64, rows.height)
    bi = np.fromiter((ctx.b_idx.get(a, -1) for a in rows["article_id"]), np.int64, rows.height)
    okq = (qi >= 0) & (bi >= 0) & queries.has_query[np.clip(qi, 0, None)]
    b = _sparse_rowwise_dot(queries.matrix, np.clip(qi, 0, None), ctx.index.doc_term, np.clip(bi, 0, None))
    cols["bm25"] = pl.Series(np.where(okq, b, np.nan)).fill_nan(None)

    rows = rows.with_columns(**cols)

    # --- category share at three scales; 0 for a known user's unseen category,
    #     null for a user with no categorised history at all ---
    rows = rows.with_columns(
        pl.col("article_id").replace_strict(ctx.category, default=None).alias("_cat"))
    for scale, h in hf.HALF_LIVES[dataset].items():
        shares = hf.user_category_shares(hist, ctx.category, h)
        known = shares.select("user_id").unique().with_columns(pl.lit(True).alias("_known"))
        rows = (rows.join(shares.rename({"category": "_cat", "share": f"cat_share_{scale}"}),
                          on=["user_id", "_cat"], how="left")
                    .join(known, on="user_id", how="left")
                    .with_columns(pl.when(pl.col("_known"))
                                  .then(pl.col(f"cat_share_{scale}").fill_null(0.0))
                                  .otherwise(None).alias(f"cat_share_{scale}"))
                    .drop("_known"))

    # --- article features (D34b, D34c) ---
    keyed = rows.select("impression_id", "timestamp", "article_id")
    rows = rows.join(af.freshness_from_seen(keyed, ctx.articles, ctx.first_seen, strict=strict),
                     on=["impression_id", "article_id"], how="left")
    rows = rows.join(ctx.exposure.shares(keyed, strict=strict),
                     on=["impression_id", "article_id"], how="left")

    # --- EB-NeRD only: time away (D35) and session context (D34a) ---
    if dataset == "ebnerd":
        rows = rows.join(hf.hours_since_last_click(target, hist), on="impression_id", how="left")
        rows = rows.join(ctx.sessions, on="impression_id", how="left")

    out = rows.sort("_row").select(KEYS + [LABEL] + FEATURES[dataset])
    expected = KEYS + [LABEL] + FEATURES[dataset]
    assert out.columns == expected, f"allowlist violated: {out.columns} != {expected}"
    assert out.height == rows.height, "a join multiplied rows"
    return out
