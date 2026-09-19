"""train the LambdaRank re-ranker and report before/after on the supplied inview lists.

Train on the train split, early-stop on val, report on the local labelled test
split (D7) - the one split that chose nothing (D36).

"Before" is A1's re-ranker, which is one column of the feature table:
`cos_inf` (semantic, A1's submitted system) and `bm25` (lexical). "After" is the
trained model. Every "after minus before" gain ships a paired bootstrap CI.

    python scripts/run_reranker.py --datasets mind ebnerd

Needs data/processed/features/*.parquet (scripts/build_feature_tables.py).
Writes reports/rerank_{dataset}_test.csv, reports/rerank_importance_{dataset}.csv,
the model to data/models/ and per-row test scores to data/processed/scores/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.eval import bootstrap, metrics  # noqa: E402
from newsrec.rank import gbdt  # noqa: E402

FEATS = REPO_ROOT / "data" / "processed" / "features"
MODELS = REPO_ROOT / "data" / "models"
SCORES = REPO_ROOT / "data" / "processed" / "scores"
REPORTS = REPO_ROOT / "reports"
METRICS = ["AUC", "MRR", "nDCG@5", "nDCG@10"]


def evaluate(ft: pl.DataFrame, scores: np.ndarray) -> dict[str, np.ndarray]:
    _, s, y = gbdt.per_impression(ft, scores)
    return metrics.evaluate_impressions(s, y).as_dict()


def baseline_scores(ft: pl.DataFrame, col: str) -> np.ndarray:
    # A user with no history has no score; all-tied, which D23's pessimistic
    # tie-break grades as no better than random - exactly what A1 did.
    return ft[col].fill_null(0.0).to_numpy()


def run(dataset: str, features: list[str], tag: str = "") -> pl.DataFrame:
    tr = pl.read_parquet(FEATS / f"{dataset}_train.parquet")
    va = pl.read_parquet(FEATS / f"{dataset}_val.parquet")
    te = pl.read_parquet(FEATS / f"{dataset}_test.parquet")
    t0 = time.perf_counter()
    ranker = gbdt.train(tr, va, features)
    fit_s = time.perf_counter() - t0
    del tr
    print(f"{dataset}{tag}: trained {ranker.best_iteration} trees in {fit_s:.0f} s", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    ranker.booster.save_model(str(MODELS / f"lgbm_{dataset}{tag}.txt"),
                              num_iteration=ranker.best_iteration)
    ranker.importance().write_csv(REPORTS / f"rerank_importance_{dataset}{tag}.csv")

    after = ranker.score(te)
    SCORES.mkdir(parents=True, exist_ok=True)
    te.select("impression_id", "user_id", "article_id", "clicked").with_columns(
        pl.Series("score", after)).write_parquet(SCORES / f"lgbm_{dataset}{tag}_test.parquet")

    systems = {"random": np.random.default_rng(gbdt.SEED).random(te.height),
               "bm25 (A1)": baseline_scores(te, "bm25"),
               "semantic (A1)": baseline_scores(te, "cos_inf"),
               "lgbm": after}
    per = {name: evaluate(te, s) for name, s in systems.items()}

    rows = []
    idx = None
    for name, vals in per.items():
        row = {"dataset": dataset, "system": name + tag, "n_impressions": te["impression_id"].n_unique()}
        for m in METRICS:
            if idx is None:
                idx = bootstrap.draw_indices(len(vals[m]), bootstrap.DEFAULT_RESAMPLES, bootstrap.DEFAULT_SEED)
            iv = bootstrap.bootstrap_mean(vals[m], indices=idx)
            row.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4), f"{m}_hi": round(iv.high, 4)})
        rows.append(row)
    for base in ("semantic (A1)", "bm25 (A1)"):
        row = {"dataset": dataset, "system": f"lgbm{tag} - {base}",
               "n_impressions": te["impression_id"].n_unique()}
        for m in METRICS:
            iv = bootstrap.paired_bootstrap_diff(per["lgbm"][m], per[base][m])
            row.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4), f"{m}_hi": round(iv.high, 4)})
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    args = ap.parse_args()
    for ds in args.datasets:
        out = run(ds, gbdt.default_features(ds))
        path = REPORTS / f"rerank_{ds}_test.csv"
        out.write_csv(path)
        with pl.Config(tbl_cols=-1, tbl_width_chars=200, tbl_rows=20):
            print(out.select("system", *METRICS))
        print(f"-> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
