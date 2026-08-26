#!/usr/bin/env python3
"""Engineering benchmarks: index build cost, per-query latency, and a MEASURED
comparison against the alternatives the decision log rejected by argument.

    python scripts/benchmark_engineering.py --dataset mind

Written after the course email of 2026-08-26 made the grading criterion explicit:
"tool/db/index choices, their impact on engineering metrics, how they compare to
alternatives, functional metrics, optimizations improving latency/throughputs".

Two things in `ARCHITECTURE.md` were arguments rather than measurements, and this
exists to convert them:

  - D14 rejected `rank_bm25` claiming 37,777 queries x 65,238 documents "would
    not finish". Plausible, never timed. If it is wrong we report that it is
    wrong - an overstated justification for a correct decision is still a defect
    in the reasoning, and it is exactly what a viva question finds.
  - The batching in D14/D21 is described as a memory necessity. Its *throughput*
    effect was never quantified, so the optimisation has no number attached.

Latency is reported per single query (batch size 1), which is the serving-time
number, alongside batched throughput, which is the offline-evaluation number.
They answer different questions and the gap between them IS the optimisation.
"""

from __future__ import annotations

import argparse
import gc
import resource
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.eval import rerank  # noqa: E402
from newsrec.retrieval import bm25, bm25_search, semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def pct(xs, p):
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 2 else max(xs)


def report(name, times_ms):
    print(f"  {name:<38} p50={statistics.median(times_ms):8.2f}ms  "
          f"p95={pct(times_ms,95):8.2f}ms  p99={pct(times_ms,99):8.2f}ms  "
          f"mean={statistics.mean(times_ms):8.2f}ms  n={len(times_ms)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mind")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--rank-bm25-docs", type=int, default=65238)
    ap.add_argument("--rank-bm25-queries", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()
    ds = args.dataset

    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == ds)
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == ds) & (pl.col("split") == "val"))
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == ds) & (pl.col("split") == "val"))
    article_ids = articles.get_column("article_id").to_list()
    n_docs = len(article_ids)
    print(f"=== {ds}: {n_docs:,} articles, {impressions.height:,} val impressions ===\n")

    # ---------- 1. index construction -------------------------------------
    print("INDEX BUILD")
    gc.collect(); t = time.perf_counter()
    index = bm25.build_index(articles)
    t_bm25 = time.perf_counter() - t
    nnz = index.doc_term.nnz
    bytes_bm25 = (index.doc_term.data.nbytes + index.doc_term.indices.nbytes
                  + index.doc_term.indptr.nbytes)
    print(f"  BM25 inverted index      {t_bm25:6.2f}s   {len(index.vocab):,} terms, "
          f"{nnz:,} non-zeros, {bytes_bm25/1e6:.0f} MB")

    t = time.perf_counter()
    embed_ids, emb = semantic.load_article_embeddings(
        PROCESSED / "embeddings.parquet", dataset=ds)
    t_emb = time.perf_counter() - t
    print(f"  embedding matrix load    {t_emb:6.2f}s   {emb.shape}, "
          f"{emb.nbytes/1e6:.0f} MB  (dense: {100*emb.nbytes/max(bytes_bm25,1):.0f}% of BM25 index)")

    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    t = time.perf_counter()
    queries = bm25_search.build_queries(history, index, title_term, n_recent=100)
    t_q = time.perf_counter() - t
    t = time.perf_counter()
    users = semantic_search.build_user_vectors(history, embed_ids, emb, n_recent=100)
    t_u = time.perf_counter() - t
    print(f"  BM25 queries (all users) {t_q:6.2f}s   {len(queries.user_ids):,} users")
    print(f"  user vectors (all users) {t_u:6.2f}s   {users.matrix.shape}, "
          f"{users.matrix.nbytes/1e6:.0f} MB")
    print(f"  peak RSS so far          {rss_gb():6.2f} GB\n")

    # ---------- 2. per-query retrieval latency (batch size 1) -------------
    print(f"RETRIEVAL LATENCY, single query, top-{args.top_k} over {n_docs:,} docs")
    rows = np.random.default_rng(0).choice(
        len(queries.user_ids), size=min(args.n_queries, len(queries.user_ids)), replace=False)

    lat = []
    for r in rows:
        t = time.perf_counter()
        s = (queries.matrix[r] @ index.doc_term.T).toarray().ravel()
        np.argpartition(-s, args.top_k)[:args.top_k]
        lat.append((time.perf_counter() - t) * 1000)
    report("BM25 sparse (ours)", lat)
    bm25_p50 = statistics.median(lat)

    lat = []
    for r in rows:
        t = time.perf_counter()
        s = emb @ users.matrix[r]
        np.argpartition(-s, args.top_k)[:args.top_k]
        lat.append((time.perf_counter() - t) * 1000)
    report("semantic brute-force (ours, D21)", lat)
    sem_p50 = statistics.median(lat)

    # ---------- 3. batched throughput, i.e. what batching bought ----------
    print(f"\nBATCHED THROUGHPUT (the D14/D21 optimisation, quantified)")
    for bs in (1, 32, 256):
        sel = rows[:256]
        t = time.perf_counter()
        for i in range(0, len(sel), bs):
            blk = users.matrix[sel[i:i + bs]]
            s = blk @ emb.T
            np.argpartition(-s, args.top_k, axis=1)[:, :args.top_k]
        el = time.perf_counter() - t
        print(f"  semantic batch={bs:<4} {len(sel)/el:9,.0f} queries/s   "
              f"({1000*el/len(sel):6.2f} ms/query)")

    # ---------- 4. re-ranking latency -------------------------------------
    print(f"\nRE-RANKING LATENCY, single impression (the leaderboard task)")
    cands = rerank.build_candidate_set(impressions.head(2000), article_ids, history)
    qr, _ = rerank._rows_by_query(cands.user_ids, users.user_ids)
    lat = []
    for i in range(len(cands)):
        if qr[i] < 0:
            continue
        t = time.perf_counter()
        emb[cands.candidate_rows[i]] @ users.matrix[qr[i]]
        lat.append((time.perf_counter() - t) * 1000)
    report("semantic re-rank (per impression)", lat)
    n_c = np.mean([len(r) for r in cands.candidate_rows])
    print(f"  mean candidates per impression: {n_c:.1f}")

    # ---------- 5. the rejected alternative, MEASURED ---------------------
    print(f"\nREJECTED ALTERNATIVE: rank_bm25 (D14 claimed it 'would not finish')")
    from rank_bm25 import BM25Okapi

    sub = articles.head(args.rank_bm25_docs)
    texts = semantic.build_document_texts(sub).get_column("text").to_list()
    toks = [bm25.tokenize(t) for t in texts]
    gc.collect(); t = time.perf_counter()
    okapi = BM25Okapi(toks)
    t_build = time.perf_counter() - t
    print(f"  build over {len(toks):,} docs: {t_build:6.2f}s  "
          f"(ours: {t_bm25:.2f}s over {n_docs:,} -> {t_build/max(t_bm25,1e-9):.1f}x)")

    qtoks = []
    hist_ids = history.get_column("history_article_ids").to_list()
    id_to_title = dict(zip(articles.get_column("article_id").to_list(),
                           articles.get_column("title").to_list()))
    for ids in hist_ids[:args.rank_bm25_queries]:
        text = " ".join((id_to_title.get(a) or "") for a in (ids or [])[-100:])
        qtoks.append(bm25.tokenize(text) or ["news"])

    lat = []
    for q in qtoks:
        t = time.perf_counter()
        s = okapi.get_scores(q)
        np.argpartition(-s, args.top_k)[:args.top_k]
        lat.append((time.perf_counter() - t) * 1000)
    report("rank_bm25 (pure-Python loop)", lat)
    rb_p50 = statistics.median(lat)

    n_users = len(queries.user_ids)
    print(f"\n  ratio at p50: rank_bm25 / ours = {rb_p50/bm25_p50:,.0f}x")
    print(f"  full val run ({n_users:,} unique users, {n_docs:,} docs):")
    print(f"     ours      {n_users*bm25_p50/1000/60:8.1f} min")
    print(f"     rank_bm25 {n_users*rb_p50/1000/60:8.1f} min "
          f"({n_users*rb_p50/1000/3600:.1f} h)")
    print(f"\n  peak RSS {rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
