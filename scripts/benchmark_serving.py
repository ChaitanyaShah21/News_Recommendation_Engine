"""serving cost of the FULL two-stage pipeline - memory, p99 latency, cost/QPS.

A1's benchmark_engineering.py measured stage 1 alone (BM25 and embedding
retrieval). A2 adds the two stages that did not exist then: building the Q1
features for the candidates, and scoring them with the LambdaRank model. Those
are measured here per single user request, because that is what a serving system
pays.

WHAT IS COUNTED AS PER-REQUEST, AND WHAT IS NOT
Not per-request (process start-up, amortised over every query): the BM25 index,
the embedding matrix, the exposure index, first-seen times, the session table,
the category map, the model. Their resident sizes are reported instead.
Per-request: retrieve top-K, build features for those K candidates, score them.

The request is one impression for one user, K = 100 (D31/Q2), on the local test
split. Runs single-threaded on purpose: a p99 measured while 8 threads fight
over 8 cores is a throughput number wearing a latency costume, and A1's error
log already records one wrong conclusion drawn from a contended measurement.

    python scripts/benchmark_serving.py [--dataset mind] [--requests 300]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.features import assemble  # noqa: E402
from newsrec.features import history as hf  # noqa: E402
from newsrec.retrieval import bm25_search  # noqa: E402
from newsrec.rank import gbdt  # noqa: E402
from newsrec.retrieval import availability as avail_mod  # noqa: E402
from newsrec.retrieval import semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
K = 100
N_RECENT_RETRIEVAL = 10  # D12, A1's generator
SLA_MS = 100.0           # Q4.3's example SLA: p99 < 100 ms
VCPU_HOUR_USD = 0.04     # stated assumption: ~1 vCPU of a c7g.xlarge, on-demand


def sizeof(obj) -> int:
    """Resident bytes of the things we actually hold, measured not guessed."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if hasattr(obj, "data") and hasattr(obj, "indices") and hasattr(obj, "indptr"):
        return obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
    if isinstance(obj, pl.DataFrame):
        return obj.estimated_size()
    if isinstance(obj, dict):
        return sum(len(k) + 64 for k in obj) + 64 * len(obj)  # rough, and labelled so
    return sys.getsizeof(obj)


def percentiles(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mind")
    ap.add_argument("--requests", type=int, default=300)
    args = ap.parse_args()
    ds = args.dataset

    imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == ds)
    hist = pl.read_parquet(PROCESSED / "history.parquet").filter(pl.col("dataset") == ds)
    arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == ds)
    ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset=ds)

    t0 = time.perf_counter()
    ctx = assemble.build_context(ds, imps, arts, ids, emb)
    startup_ctx_s = time.perf_counter() - t0
    booster = lgb.Booster(model_file=str(REPO_ROOT / "data" / "models" / f"lgbm_{ds}.txt"))
    feats = assemble.FEATURES[ds]

    target = imps.filter(pl.col("split") == "test")
    hist_split = hist.filter(pl.col("split") == "test")
    users = semantic_search.build_user_vectors(hist_split, ids, emb, n_recent=N_RECENT_RETRIEVAL)
    q_of = {u: i for i, u in enumerate(users.user_ids)}
    row_of = {a: i for i, a in enumerate(ids)}
    hist_of = dict(zip(hist_split["user_id"], hist_split["history_article_ids"].to_list()))
    _, bucket_id, masks = avail_mod.build_availability(imps, target, ids, "1h")
    startup_total_s = time.perf_counter() - t0

    footprint = {
        "bm25 index (sparse doc-term)": sizeof(ctx.index.doc_term),
        "bm25 title-term matrix": sizeof(ctx.title_term),
        "article embeddings": sizeof(emb),
        "user vectors (test split)": sizeof(users.matrix),
        "exposure index (2 int64 arrays)": sizeof(ctx.exposure.imp_t) + sizeof(ctx.exposure.shown_key),
        "first-seen table": sizeof(ctx.first_seen),
        "availability masks (1h buckets)": sum(m.nbytes for m in masks),
        "session table": sizeof(ctx.sessions) if ctx.sessions is not None else 0,
        "category map (rough)": sizeof(ctx.category),
        "model (trees, on disk)": (REPO_ROOT / "data" / "models" / f"lgbm_{ds}.txt").stat().st_size,
    }

    sample = target.sample(min(args.requests, target.height), seed=gbdt.SEED)
    stage = {"retrieve": [], "features": [], "score": [], "total": [], "profile_part": []}
    served = 0
    for row in sample.iter_rows(named=True):
        u = row["user_id"]
        if u not in q_of or not users.has_query[q_of[u]]:
            continue  # a cold user skips stage 1 entirely (D17); not a latency sample
        one = target.filter(pl.col("impression_id") == row["impression_id"])
        t_req = time.perf_counter()

        drop = np.asarray([row_of[a] for a in (hist_of.get(u) or []) if a in row_of], np.int32)
        bucket = masks[bucket_id[row["timestamp"].replace(minute=0, second=0, microsecond=0)]]
        top = semantic_search.retrieve_bucketed(
            users, emb, np.array([q_of[u]], np.int64), np.array([0], np.int64),
            [bucket], k=K, exclude_rows=[drop] * len(users.user_ids), batch_size=1)[0]
        t_retr = time.perf_counter()

        one = one.with_columns(
            pl.Series("candidate_article_ids", [[ids[r] for r in top]], dtype=pl.List(pl.Utf8)))
        ft = assemble.build_rows(ctx, one, hist_split.filter(pl.col("user_id") == u))
        t_feat = time.perf_counter()

        booster.predict(gbdt.feature_matrix(ft, feats))
        t_end = time.perf_counter()

        # How much of the feature stage is PER-USER work - three decayed
        # profiles, three category-share tables and the BM25 query. It depends
        # only on the user's history, never on the request, so a production
        # system precomputes it nightly. Measured separately (repeating it, so
        # it is not subtracted from a number it was part of) to make the
        # "precompute this" claim in the design note a measurement.
        one_hist = hist_split.filter(pl.col("user_id") == u)
        t_p = time.perf_counter()
        for scale, half in hf.HALF_LIVES[ds].items():
            hf.build_decayed_user_vectors(one_hist, ids, emb, half)
            hf.user_category_shares(one_hist, ctx.category, half)
        bm25_search.build_queries(one_hist, ctx.index, ctx.title_term, n_recent=hf.N_RECENT)
        stage["profile_part"].append((time.perf_counter() - t_p) * 1e3)

        stage["retrieve"].append((t_retr - t_req) * 1e3)
        stage["features"].append((t_feat - t_retr) * 1e3)
        stage["score"].append((t_end - t_feat) * 1e3)
        stage["total"].append((t_end - t_req) * 1e3)
        served += 1

    out = {"dataset": ds, "k": K, "requests_measured": served,
           "startup_context_s": round(startup_ctx_s, 1),
           "startup_total_s": round(startup_total_s, 1),
           "footprint_bytes": footprint,
           "footprint_total_mb": round(sum(footprint.values()) / 1e6, 1),
           "latency_ms": {k: {m: round(v, 2) for m, v in percentiles(x).items()}
                          for k, x in stage.items()},
           "sla_ms": SLA_MS, "vcpu_hour_usd": VCPU_HOUR_USD}
    prof = out["latency_ms"]["profile_part"]["p50"]
    out["precomputable_share_of_features"] = round(
        prof / out["latency_ms"]["features"]["p50"], 3)
    out["p99_if_profiles_precomputed_ms"] = round(
        out["latency_ms"]["total"]["p99"] - out["latency_ms"]["profile_part"]["p99"], 1)
    p99 = out["latency_ms"]["total"]["p99"]
    mean = out["latency_ms"]["total"]["mean"]
    # Cost: one request occupies one core for `mean` ms, so a core serves
    # 1000/mean requests per second. Little's law with N=1 caps a single
    # in-flight request at 1/R, which is the honest ceiling to quote.
    qps_per_core = 1000.0 / mean
    out["qps_per_core"] = round(qps_per_core, 1)
    out["cost_per_1000_queries_usd"] = round(VCPU_HOUR_USD / 3600 * (1000 / qps_per_core), 6)
    out["meets_sla"] = bool(p99 < SLA_MS)
    out["cores_for_1000_qps"] = int(np.ceil(1000 / qps_per_core))

    path = REPORTS / f"serving_benchmark_{ds}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "footprint_bytes"}, indent=2))
    print(f"footprint: {out['footprint_total_mb']} MB total")
    for k, v in sorted(footprint.items(), key=lambda kv: -kv[1]):
        print(f"  {k:38s} {v / 1e6:8.1f} MB")
    print(f"-> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
