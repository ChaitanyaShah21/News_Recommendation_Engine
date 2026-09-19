"""metrics WITH and WITHOUT features unavailable at serving time.

Q9 asks for both numbers, not a promise. So we train the same LambdaRank model
twice: once on the honest allowlist (D34), once on the allowlist plus the
quarantined columns from features/unavailable.py:

  read_time, scroll_percentage   measured AFTER the impression was served -
                                 EB-NeRD's test file still ships them, which is
                                 exactly why "present" is not "available".
  total_pageviews                a whole-dataset aggregate: it contains every
                                 impression's future.
  future_exposure_share_24h      D34b's window run forwards, (T, T+24h]: what
                                 the site WILL push.

The gap between the two is the price of cheating, measured rather than asserted,
with a paired bootstrap CI. The honest model is the one that ships and the one
both leaderboard files came from.

    python scripts/run_q9_ablation.py [--datasets mind ebnerd]
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

from newsrec.eval import bootstrap as boot  # noqa: E402
from newsrec.eval import metrics as met  # noqa: E402
from newsrec.features import assemble  # noqa: E402
from newsrec.features.unavailable import UNAVAILABLE, unavailable_features  # noqa: E402
from newsrec.rank import gbdt  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
METRICS = ["AUC", "MRR", "nDCG@5", "nDCG@10"]


def augment(ds: str, split: str, imps: pl.DataFrame, arts: pl.DataFrame) -> pl.DataFrame:
    """The cached honest table plus the quarantined columns, same row order."""
    ft = pl.read_parquet(PROCESSED / "features" / f"{ds}_{split}.parquet")
    target = imps.filter(pl.col("split") == split)
    t0 = time.perf_counter()
    out = unavailable_features(ft, target, imps, arts)
    missing = [c for c in UNAVAILABLE if c not in out.columns]
    if missing:
        raise ValueError(f"quarantine columns absent: {missing}")
    print(f"  {ds} {split}: +{len(UNAVAILABLE)} unavailable columns "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return out


def per_impression_metrics(ft: pl.DataFrame, scores: np.ndarray) -> dict[str, np.ndarray]:
    _, s, y = gbdt.per_impression(ft, scores)
    return met.evaluate_impressions(s, y).as_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    args = ap.parse_args()

    for ds in args.datasets:
        imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == ds)
        arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == ds)
        tables = {s: augment(ds, s, imps, arts) for s in ("train", "val", "test")}

        honest = assemble.FEATURES[ds]
        # Columns that are entirely null carry no signal and LightGBM ignores
        # them; naming them keeps the table honest about what each dataset
        # actually ships (MIND has no dwell fields and no pageview counts).
        usable = [c for c in UNAVAILABLE if tables["train"][c].null_count() < tables["train"].height]
        print(f"{ds}: unavailable columns with any data: {usable}", flush=True)

        runs = {"honest (ships)": honest, "+ serving-unavailable": honest + usable}
        per = {}
        for name, feats in runs.items():
            r = gbdt.train(tables["train"], tables["val"], feats)
            per[name] = per_impression_metrics(tables["test"], r.score(tables["test"]))
            imp = r.importance().head(5)
            print(f"{ds} {name}: {r.best_iteration} trees; top gains "
                  f"{list(zip(imp['feature'], imp['share'].round(3)))}", flush=True)

        rows, idx = [], None
        for name in runs:
            row = {"dataset": ds, "system": name, "n": len(per[name]["AUC"])}
            for m in METRICS:
                if idx is None:
                    idx = boot.draw_indices(len(per[name][m]), boot.DEFAULT_RESAMPLES,
                                            boot.DEFAULT_SEED)
                iv = boot.bootstrap_mean(per[name][m], indices=idx)
                row.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4),
                            f"{m}_hi": round(iv.high, 4)})
            rows.append(row)
        row = {"dataset": ds, "system": "price of cheating [paired]", "n": len(per[name]["AUC"])}
        for m in METRICS:
            iv = boot.paired_bootstrap_diff(per["+ serving-unavailable"][m], per["honest (ships)"][m])
            row.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4),
                        f"{m}_hi": round(iv.high, 4)})
        rows.append(row)

        out = pl.DataFrame(rows)
        path = REPORTS / f"q9_serving_ablation_{ds}_test.csv"
        out.write_csv(path)
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(out.select("system", *METRICS))
        print(f"-> {path.relative_to(REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
