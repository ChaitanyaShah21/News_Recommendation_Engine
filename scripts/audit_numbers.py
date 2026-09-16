#!/usr/bin/env python3
"""Measure every system quantity `reports/NUMBERS.md` quotes. Read-only.

Written after Quiz-1 (2026-09-16) asked for corpus sizes, index sizes against the
raw corpus, the memory tier each index lives on, p50/p99 latency, throughput and a
Little's-law consistency check - and made clear those answers are cross-checked
against this repository.

    python scripts/audit_numbers.py                      # this repo's store (A2)
    python scripts/audit_numbers.py --processed /home/csharp/IRE/A1/data/processed \
        --raw /home/csharp/IRE/A1/data/raw                # the A1 system as submitted

Writes nothing anywhere: pointing it at a frozen repo cannot modify it.

Latency here is machine-state-dependent - see PROGRESS.md's 2026-09-16 error-log
entry, where the same code on identical data ran ~2.4x slower than three weeks
earlier. Always record the date beside the number, and prefer ratios to absolutes.
"""
from __future__ import annotations
import argparse, gc, resource, statistics, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import polars as pl
from newsrec.eval import rerank
from newsrec.retrieval import bm25, bm25_search, semantic, semantic_search

N_RECENT = 100      # D31: the shipped system uses the 100 most recent clicks
TOP_K = 200
N_QUERIES = 300

# Raw sources per dataset, so "index size vs raw corpus" has a denominator.
# EB-NeRD's bundle differs between A1 (demo) and A2 (large): whichever exists is used.
def raw_sources(raw_root: Path) -> dict[str, list[Path]]:
    ebnerd = [raw_root / "ebnerd/ebnerd_large/articles.parquet",
              raw_root / "ebnerd/ebnerd_demo/articles.parquet"]
    return {
        "mind": [raw_root / "mind/MINDsmall_train/news.tsv",
                 raw_root / "mind/MINDsmall_dev/news.tsv"],
        "ebnerd": [p for p in ebnerd if p.exists()][:1],
    }

def pctile(xs, p):
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 2 else max(xs)

def stats(name, ms):
    print(f"  {name:<34} p50={statistics.median(ms):7.2f}  p95={pctile(ms,95):7.2f}  "
          f"p99={pctile(ms,99):7.2f}  mean={statistics.mean(ms):7.2f} ms  n={len(ms)}")
    return statistics.mean(ms)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=REPO_ROOT / "data" / "processed",
                    help="feature store to measure (default: this repo's)")
    ap.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw",
                    help="raw-data root, for the index-vs-corpus size ratio")
    ap.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    processed, RAW_BYTES = args.processed, raw_sources(args.raw)
    print(f"store: {processed}\nraw:   {args.raw}\nsplit: {args.split}")

    articles_all = pl.read_parquet(processed / "articles.parquet")
    imps_all = pl.read_parquet(processed / "impressions.parquet")
    hist_all = pl.read_parquet(processed / "history.parquet")

    for ds in args.datasets:
        articles = articles_all.filter(pl.col("dataset") == ds)
        imps = imps_all.filter((pl.col("dataset") == ds) & (pl.col("split") == args.split))
        hist = hist_all.filter((pl.col("dataset") == ds) & (pl.col("split") == args.split))
        raw = sum(p.stat().st_size for p in RAW_BYTES[ds])
        print(f"\n{'='*70}\n=== {ds} ===")
        print(f"CORPUS  docs={articles.height:,}  raw source bytes={raw/1e6:.1f} MB "
              f"({', '.join(p.name for p in RAW_BYTES[ds])})")

        gc.collect(); t = time.perf_counter()
        index = bm25.build_index(articles)
        t_build = time.perf_counter() - t
        nnz = index.doc_term.nnz
        idx_bytes = (index.doc_term.data.nbytes + index.doc_term.indices.nbytes
                     + index.doc_term.indptr.nbytes + index.doc_len.nbytes)
        print(f"LEXICAL fields=title+abstract  terms(vocab)={len(index.vocab):,}  "
              f"postings(nnz)={nnz:,}  postings/doc={nnz/articles.height:.1f}")
        print(f"        avg tokens/doc={float(index.doc_len.mean()):.1f}  "
              f"total tokens={float(index.doc_len.sum()):,.0f}")
        print(f"        index={idx_bytes/1e6:.1f} MB = {100*idx_bytes/raw:.0f}% of raw  "
              f"build={t_build:.2f}s = {articles.height/t_build:,.0f} docs/s")
        print(f"        structure=scipy.sparse.csr_matrix float32, resident in RAM")

        t = time.perf_counter()
        emb_ids, emb = semantic.load_article_embeddings(
            processed / "embeddings.parquet", dataset=ds)
        t_load = time.perf_counter() - t
        print(f"SEMANTIC model={semantic.MODEL_NAME} dim={emb.shape[1]}  rows={emb.shape[0]:,}  "
              f"{emb.nbytes/1e6:.1f} MB = {100*emb.nbytes/raw:.0f}% of raw  load={t_load:.2f}s")
        print(f"        dense/sparse size ratio = {emb.nbytes/idx_bytes:.1f}x")

        title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
        t = time.perf_counter()
        queries = bm25_search.build_queries(hist, index, title_term, n_recent=N_RECENT)
        t_q = time.perf_counter() - t
        t = time.perf_counter()
        users = semantic_search.build_user_vectors(hist, emb_ids, emb, n_recent=N_RECENT)
        t_u = time.perf_counter() - t
        qnnz = np.diff(queries.matrix.indptr)
        print(f"QUERIES {args.split} impressions={imps.height:,}  users with a query={len(queries.user_ids):,}  "
              f"history rows={hist.height:,}")
        print(f"        mean distinct query terms={qnnz.mean():.1f} (N={N_RECENT} recent titles)  "
              f"mean candidates/impression={imps['candidate_article_ids'].list.len().mean():.1f}")
        print(f"        build: queries {t_q:.2f}s, user vectors {t_u:.2f}s "
              f"({users.matrix.nbytes/1e6:.0f} MB, RAM)")

        rows = np.random.default_rng(0).choice(
            len(queries.user_ids), size=min(N_QUERIES, len(queries.user_ids)), replace=False)
        print(f"LATENCY single query, top-{TOP_K}, batch size 1 (two independent passes)")
        for _ in range(3):  # warm-up: first BLAS call pays thread-pool setup
            _ = emb @ users.matrix[rows[0]]
            _ = (queries.matrix[rows[0]] @ index.doc_term.T).toarray()
        lat = []
        for r in rows:
            t = time.perf_counter()
            s = (queries.matrix[r] @ index.doc_term.T).toarray().ravel()
            np.argpartition(-s, TOP_K)[:TOP_K]
            lat.append((time.perf_counter() - t) * 1000)
        mean_bm25 = stats("BM25 sparse", lat)
        lat = []
        for r in rows:
            t = time.perf_counter()
            s = emb @ users.matrix[r]
            np.argpartition(-s, TOP_K)[:TOP_K]
            lat.append((time.perf_counter() - t) * 1000)
        mean_sem = stats("semantic brute force", lat)

        lat2 = []
        for r in rows:
            t = time.perf_counter()
            s2 = (queries.matrix[r] @ index.doc_term.T).toarray().ravel()
            np.argpartition(-s2, TOP_K)[:TOP_K]
            lat2.append((time.perf_counter() - t) * 1000)
        stats("BM25 sparse (pass 2)", lat2)
        lat2 = []
        for r in rows:
            t = time.perf_counter()
            s2 = emb @ users.matrix[r]
            np.argpartition(-s2, TOP_K)[:TOP_K]
            lat2.append((time.perf_counter() - t) * 1000)
        stats("semantic brute force (pass 2)", lat2)

        # Little's law with N=1: a single worker cannot exceed 1/R.
        print("LITTLE'S LAW  N=1, so X <= 1/R")
        for name, mean_ms in (("bm25", mean_bm25), ("semantic", mean_sem)):
            sel = rows[:256]
            t = time.perf_counter()
            if name == "semantic":
                for i in range(0, len(sel), 32):
                    blk = users.matrix[sel[i:i+32]]
                    sc = emb @ blk.T
                    np.argpartition(-sc, TOP_K, axis=0)[:TOP_K]
            else:
                for i in range(0, len(sel), 32):
                    sc = (queries.matrix[sel[i:i+32]] @ index.doc_term.T).toarray()
                    np.argpartition(-sc, TOP_K, axis=1)[:, :TOP_K]
            batched = len(sel) / (time.perf_counter() - t)
            print(f"  {name:<9} serial 1/R = {1000/mean_ms:8.1f} q/s   "
                  f"batched(32) = {batched:8.1f} q/s   speed-up x{batched*mean_ms/1000:.2f}")

        # Re-ranking the supplied candidate list: the leaderboard-shaped operation.
        cands = rerank.build_candidate_set(imps, emb_ids, hist)
        t = time.perf_counter()
        rerank.score_semantic(cands, users, emb)
        t_rr = time.perf_counter() - t
        print(f"RE-RANK semantic over supplied candidates: {t_rr:.2f}s for {len(cands):,} "
              f"impressions = {len(cands)/t_rr:,.0f} impressions/s, "
              f"{1000*t_rr/len(cands):.4f} ms each")
        print(f"PEAK RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
        del index, emb, users, queries, title_term, cands
        gc.collect()
    return 0

if __name__ == "__main__":
    sys.exit(main())
