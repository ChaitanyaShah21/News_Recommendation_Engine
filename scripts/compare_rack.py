"""D42: the shipped model vs the rack-normalised model, paired, on the test split.

Both models scored the SAME impressions and the same candidate lists, so the
comparison is paired: impression i contributes d_i = rack_i - base_i and the
bootstrap resamples those differences (Q3.4's method, reused). Shared
per-impression difficulty cancels, which is the whole point - two overlapping
unpaired intervals can still hide a real and consistent gain.

    python scripts/compare_rack.py --datasets mind ebnerd

Writes reports/rack_vs_base_{dataset}_test.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.eval import bootstrap, metrics  # noqa: E402
from newsrec.rank import gbdt  # noqa: E402

SCORES = REPO_ROOT / "data" / "processed" / "scores"
FEATS = REPO_ROOT / "data" / "processed" / "features"
REPORTS = REPO_ROOT / "reports"
METRICS = ["AUC", "MRR", "nDCG@5", "nDCG@10"]


def per_metric(ft: pl.DataFrame, scores: np.ndarray) -> dict[str, np.ndarray]:
    _, s, y = gbdt.per_impression(ft, scores)
    return metrics.evaluate_impressions(s, y).as_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    args = ap.parse_args()

    for ds in args.datasets:
        ft = pl.read_parquet(FEATS / f"{ds}_test.parquet")
        arms = {}
        for tag, name in (("", "lgbm (base, shipped)"), ("_rack", "lgbm + rack (D42)")):
            s = pl.read_parquet(SCORES / f"lgbm_{ds}{tag}_test.parquet")
            # Row alignment is the thing that would silently invalidate a paired
            # test: mismatched rows still produce a plausible number.
            if s.height != ft.height or not (s["article_id"] == ft["article_id"]).all():
                raise SystemExit(f"{ds}{tag}: saved scores are not row-aligned with the "
                                 "feature table; re-run run_reranker.py")
            arms[name] = per_metric(ft, s["score"].to_numpy())

        rows, idx = [], None
        for name, vals in arms.items():
            r = {"dataset": ds, "system": name,
                 "n_impressions": ft["impression_id"].n_unique()}
            for m in METRICS:
                if idx is None:
                    idx = bootstrap.draw_indices(len(vals[m]), bootstrap.DEFAULT_RESAMPLES,
                                                 bootstrap.DEFAULT_SEED)
                iv = bootstrap.bootstrap_mean(vals[m], indices=idx)
                r.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4),
                          f"{m}_hi": round(iv.high, 4)})
            rows.append(r)

        r = {"dataset": ds, "system": "rack - base (paired)",
             "n_impressions": ft["impression_id"].n_unique()}
        for m in METRICS:
            iv = bootstrap.paired_bootstrap_diff(arms["lgbm + rack (D42)"][m],
                                                 arms["lgbm (base, shipped)"][m])
            r.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4),
                      f"{m}_hi": round(iv.high, 4)})
            excludes = (iv.low > 0) or (iv.high < 0)
            r[f"{m}_excludes_zero"] = excludes
        rows.append(r)

        out = pl.DataFrame(rows)
        path = REPORTS / f"rack_vs_base_{ds}_test.csv"
        out.write_csv(path)
        with pl.Config(tbl_cols=-1, tbl_width_chars=220, tbl_rows=10):
            print(out.select("system", *[c for m in METRICS
                                         for c in (m, f"{m}_lo", f"{m}_hi")]))
        print(f"-> {path.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
