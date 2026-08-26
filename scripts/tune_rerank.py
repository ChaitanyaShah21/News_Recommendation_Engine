#!/usr/bin/env python3
"""Phase 5b - tune the re-ranker on val, offline, before spending a submission.

    python scripts/tune_rerank.py --dataset mind

Val is a trustworthy proxy here, which is what makes this worth doing: our val
AUC of 0.6338 predicted a leaderboard AUC of 0.6037 - close, and more
importantly monotone, so an improvement measured here should survive.

Runs three experiments and prints one table per stage:
  1. N sweep - the history window, never tuned for re-ranking (D12 chose it for
     retrieval, where long queries cause topic drift; re-ranking is a different
     job).
  2. mean-pooled vs max-similarity user representation, at each N.
  3. weighted fusion of the best variants with BM25 and popularity, over both
     normalisations.

Everything is scored ONCE per (variant, N) and cached in memory, so the fusion
weight search costs no re-scoring at all.
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

from newsrec.eval import metrics as met  # noqa: E402
from newsrec.eval import rerank  # noqa: E402
from newsrec.eval import rerank_variants as rv  # noqa: E402
from newsrec.retrieval import bm25, bm25_search, semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"


def macro_auc(scores, labels, mask) -> float:
    """Macro-mean AUC over the impressions `mask` selects (D17/D18 headline)."""
    r = met.evaluate_impressions(scores, labels, tie=met.PESSIMISTIC)
    return float(met.macro_mean(np.asarray(r.auc)[mask])[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mind")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-values", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    args = ap.parse_args()
    ds, split = args.dataset, args.split

    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == ds)
    imps = pl.read_parquet(PROCESSED / "impressions.parquet")
    impressions = imps.filter((pl.col("dataset") == ds) & (pl.col("split") == split))
    train_impressions = imps.filter((pl.col("dataset") == ds) & (pl.col("split") == "train"))
    history = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == ds) & (pl.col("split") == split))
    del imps

    article_ids = articles.get_column("article_id").to_list()
    embed_ids, emb = semantic.load_article_embeddings(
        PROCESSED / "embeddings.parquet", dataset=ds)
    if embed_ids != article_ids:
        raise ValueError("embedding row order does not match the articles table")

    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    cands = rerank.build_candidate_set(impressions, article_ids, history)
    counts = rerank.train_click_counts(train_impressions, article_ids)
    labels = cands.labels
    print(f"{ds} {split}: {len(cands):,} impressions, {len(article_ids):,} articles")

    # D17: the headline excludes impressions whose user has no usable query.
    # Computed at N=10 and held FIXED across every variant below, so the sweep
    # compares scorers over an identical impression set rather than letting the
    # denominator move with N.
    q10 = bm25_search.build_queries(history, index, title_term, n_recent=10)
    qr, present = rerank._rows_by_query(cands.user_ids, q10.user_ids)
    mask = present & q10.has_query[np.clip(qr, 0, None)]
    print(f"has-query impressions (fixed denominator): {int(mask.sum()):,}\n")

    comp: dict[str, list] = {}

    # ---- stage 1+2: N sweep, mean vs max ---------------------------------
    print(f"{'N':>5}  {'bm25':>8} {'sem-mean':>9} {'sem-max':>9}     (val macro AUC)")
    best = {"bm25": (-1, None), "mean": (-1, None), "max": (-1, None)}
    for n in args.n_values:
        t0 = time.perf_counter()
        q = bm25_search.build_queries(history, index, title_term, n_recent=n)
        u = semantic_search.build_user_vectors(history, embed_ids, emb, n_recent=n)
        hr = rv.build_history_rows(history, article_ids, n)

        s_bm = rerank.score_bm25(cands, index, q)
        s_mean = rerank.score_semantic(cands, u, emb)
        s_max = rv.score_semantic_maxsim(cands, hr, emb)

        a_bm, a_mean, a_max = (macro_auc(s, labels, mask) for s in (s_bm, s_mean, s_max))
        comp[f"bm25@{n}"], comp[f"mean@{n}"], comp[f"max@{n}"] = s_bm, s_mean, s_max
        for key, a, s in (("bm25", a_bm, f"bm25@{n}"), ("mean", a_mean, f"mean@{n}"),
                          ("max", a_max, f"max@{n}")):
            if a > best[key][0]:
                best[key] = (a, s)
        print(f"{n:>5}  {a_bm:8.4f} {a_mean:9.4f} {a_max:9.4f}   "
              f"({time.perf_counter()-t0:.0f}s)")

    comp["pop"] = rerank.score_popularity(cands, counts)
    a_pop = macro_auc(comp["pop"], labels, mask)
    print(f"\npopularity: {a_pop:.4f}")
    print(f"best bm25 : {best['bm25'][1]}  {best['bm25'][0]:.4f}")
    print(f"best mean : {best['mean'][1]}  {best['mean'][0]:.4f}")
    print(f"best max  : {best['max'][1]}  {best['max'][0]:.4f}")

    # ---- stage 3: fusion --------------------------------------------------
    picks = [best["mean"][1], best["max"][1], best["bm25"][1], "pop"]
    print(f"\nfusing {picks}")
    results = []
    for mode in ("rank", "zscore"):
        norm = {k: rv.normalise_per_impression(comp[k], mode=mode) for k in picks}
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        for w in itertools.product(grid, repeat=4):
            if sum(w) == 0:
                continue
            weights = dict(zip(picks, w))
            auc = macro_auc(rv.blend(norm, weights), labels, mask)
            results.append((auc, mode, weights))

    results.sort(reverse=True, key=lambda r: r[0])
    print(f"\n{'AUC':>8}  {'norm':>7}  weights")
    for auc, mode, weights in results[:12]:
        ws = " ".join(f"{k}={v:g}" for k, v in weights.items() if v)
        print(f"{auc:8.4f}  {mode:>7}  {ws}")

    baseline = best["mean"][0]
    top = results[0]
    print(f"\nbaseline (current submission, mean@10-style best): {baseline:.4f}")
    print(f"best fused                                       : {top[0]:.4f}  "
          f"({top[0]-baseline:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
