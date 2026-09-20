"""D42a's arm selection: base / +pct / +z / +both, graded on VALIDATION.

The arm is chosen here and nowhere else. Test is opened only afterwards, by
`scripts/run_reranker.py --features rack` and `scripts/compare_rack.py`, so the
split that chose nothing stays that way (D36).

Selection rule, fixed before the numbers were read: validation nDCG@10 — the
metric LambdaRank optimises and the one early stopping already uses. Choosing on
a different metric after seeing the table would be A1's Finding 5 in new clothes.

    python scripts/diagnostics/rack_arms.py mind
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from newsrec.eval import metrics  # noqa: E402
from newsrec.features import rack  # noqa: E402
from newsrec.rank import gbdt  # noqa: E402

FEATS = REPO_ROOT / "data" / "processed" / "features"
METRICS = ("AUC", "MRR", "nDCG@5", "nDCG@10")


def main() -> None:
    ds = sys.argv[1] if len(sys.argv) > 1 else "mind"
    base = gbdt.default_features(ds)
    pct = [f"{f}_pct" for f in rack.RACK_BASE]
    z = [f"{f}_z" for f in rack.RACK_BASE]
    arms = {"base": base, "+pct": base + pct, "+z": base + z, "+both": base + pct + z}

    t0 = time.perf_counter()
    tr = pl.read_parquet(FEATS / f"{ds}_train.parquet")
    va = pl.read_parquet(FEATS / f"{ds}_val.parquet")
    missing = [c for c in pct + z if c not in tr.columns]
    if missing:
        raise SystemExit(f"feature tables lack rack columns ({missing[:2]}...); "
                         "rebuild with scripts/build_feature_tables.py")
    print(f"{ds}: train {tr.height:,} val {va.height:,} rows, "
          f"loaded in {time.perf_counter() - t0:.0f} s", flush=True)

    for name, feats in arms.items():
        t0 = time.perf_counter()
        r = gbdt.train(tr, va, feats)
        _, sc, y = gbdt.per_impression(va, r.score(va))
        m = metrics.evaluate_impressions(sc, y).as_dict()
        vals = {k: metrics.macro_mean(m[k])[0] for k in METRICS}
        print(f"{ds:8s} {name:6s} trees={r.best_iteration:4d}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in vals.items())
              + f"   ({time.perf_counter() - t0:.0f} s)", flush=True)
        top = r.importance().head(6)
        print("         top gain: " + ", ".join(
            f"{a} {b:.0%}" for a, b in zip(top["feature"], top["share"])), flush=True)
        del r
        gc.collect()


if __name__ == "__main__":
    main()
