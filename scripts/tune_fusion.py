#!/usr/bin/env python3
"""Phase 5b - fuse the transferable re-ranking scorers, tuned on val.

    python scripts/tune_fusion.py --dataset mind --n-recent 100

Rewritten after a first attempt ran over an hour without finishing. Three
corrections, each of which was a real design error rather than bad luck:

  1. **AUC only.** The first version called `evaluate_impressions`, which
     computes seven metrics (AUC, MRR, nDCG@5, nDCG@10 and three optimistic
     variants) for every weight combination when the search reads exactly one
     of them.
  2. **Popularity dropped.** Measured before deciding: only **5.7%** of MIND
     leaderboard-test candidates (1,701 of 30,043) have a train-window click
     count, because our train split is MIND-small (9-14 Nov 2019) and the test
     bundle starts 19 Nov. A weight tuned on a signal that is absent for 94% of
     test candidates flatters val and does nothing on the leaderboard. Fusing
     only what transfers - text and history, both present in the test bundle.
  3. **One weight pinned.** AUC depends only on the within-impression ordering,
     which is invariant to positive rescaling of the whole blend, so a free
     4-way grid was 5x redundant. `mean` is fixed at 1.0 and the others are
     measured relative to it.

Together: ~1,248 seven-metric evaluations became ~72 one-metric ones.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from newsrec.eval import rerank  # noqa: E402
from newsrec.eval import rerank_variants as rv  # noqa: E402
from newsrec.retrieval import bm25, bm25_search, semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"


def auc_fast(scores, labels, mask) -> float:
    """Macro-mean AUC only, via the rank identity.

    Same formula `metrics.py` uses and is mutation-tested against an
    independent O(P*N) pair-counting implementation - reproduced here purely to
    skip the six metrics the weight search does not read. `rankdata` defaults to
    average ranks, which is AUC's own tie rule (a tied pair counts 0.5).
    """
    vals = []
    for i in np.flatnonzero(mask):
        y = labels[i]
        n_pos = int(y.sum())
        n_neg = y.shape[0] - n_pos
        if n_pos == 0 or n_neg == 0:
            continue  # undefined, not zero - same convention as metrics.py
        r = rankdata(scores[i])
        vals.append((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mind")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-recent", type=int, default=100)
    args = ap.parse_args()
    ds, n = args.dataset, args.n_recent

    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == ds)
    imps = pl.read_parquet(PROCESSED / "impressions.parquet")
    impressions = imps.filter((pl.col("dataset") == ds) & (pl.col("split") == args.split))
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == ds) & (pl.col("split") == args.split))
    del imps

    article_ids = articles.get_column("article_id").to_list()
    embed_ids, emb = semantic.load_article_embeddings(
        PROCESSED / "embeddings.parquet", dataset=ds)
    if embed_ids != article_ids:
        raise ValueError("embedding row order does not match the articles table")

    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    cands = rerank.build_candidate_set(impressions, article_ids, history)
    labels = cands.labels

    t0 = time.perf_counter()
    q = bm25_search.build_queries(history, index, title_term, n_recent=n)
    u = semantic_search.build_user_vectors(history, embed_ids, emb, n_recent=n)
    hr = rv.build_history_rows(history, article_ids, n)
    qr, present = rerank._rows_by_query(cands.user_ids, q.user_ids)
    mask = present & q.has_query[np.clip(qr, 0, None)]

    comp = {
        "mean": rerank.score_semantic(cands, u, emb),
        "max": rv.score_semantic_maxsim(cands, hr, emb),
        "bm25": rerank.score_bm25(cands, index, q),
    }
    print(f"{ds} {args.split}  N={n}  {len(cands):,} impressions, "
          f"{int(mask.sum()):,} with a query   (scored in {time.perf_counter()-t0:.0f}s)")
    for k, v in comp.items():
        print(f"  {k:<5} alone: {auc_fast(v, labels, mask):.4f}")

    grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    results = []
    for mode in ("rank", "zscore"):
        norm = {k: rv.normalise_per_impression(v, mode=mode) for k, v in comp.items()}
        t = time.perf_counter()
        for w_max, w_bm in itertools.product(grid, repeat=2):
            weights = {"mean": 1.0, "max": w_max, "bm25": w_bm}
            results.append((auc_fast(rv.blend(norm, weights), labels, mask), mode, weights))
        print(f"  {mode:>6} grid: {len(grid)**2} combos in {time.perf_counter()-t:.0f}s")

    results.sort(reverse=True, key=lambda r: r[0])
    base = auc_fast(comp["mean"], labels, mask)
    print(f"\n{'AUC':>8}  {'delta':>8}  {'norm':>7}  weights")
    for auc, mode, w in results[:10]:
        ws = " ".join(f"{k}={v:g}" for k, v in w.items() if v)
        print(f"{auc:8.4f}  {auc-base:+8.4f}  {mode:>7}  {ws}")
    print(f"\nmean-only baseline at N={n}: {base:.4f}")
    print(f"best fused                 : {results[0][0]:.4f} "
          f"({results[0][0]-base:+.4f})  {results[0][1]} {results[0][2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
