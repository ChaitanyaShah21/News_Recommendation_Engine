"""D42's diagnosis: how much of each feature's variance is BETWEEN racks?

A gradient-boosted tree splits on absolute values with one threshold shared by
every impression. A feature whose LEVEL drifts between impressions therefore
gives the tree a threshold that means different things in different racks, and
the within-rack ordering it should be reading is wasted. This quantifies that:

    between-rack share = Var(per-impression means, weighted by rack size)
                         / Var(feature over all rows)

Near 1: almost all the variation is "which rack is this". Near 0: the feature
already varies mostly within a rack, and normalising it buys little.

    python scripts/diagnostics/rack_variance.py [dataset] [split]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

FEATS = REPO_ROOT / "data" / "processed" / "features"
KEYS = ("impression_id", "user_id", "article_id", "clicked")


def main() -> None:
    ds = sys.argv[1] if len(sys.argv) > 1 else "mind"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    ft = pl.read_parquet(FEATS / f"{ds}_{split}.parquet")
    print(f"{ds}_{split}: {ft.height:,} rows, {ft['impression_id'].n_unique():,} impressions")

    sizes = ft.group_by("impression_id").len()["len"].to_numpy()
    print("rack size: min %d p10 %d p50 %d p90 %d max %d mean %.1f" % (
        sizes.min(), *np.percentile(sizes, [10, 50, 90]).astype(int),
        sizes.max(), sizes.mean()))

    feats = [c for c in ft.columns if c not in KEYS and not c.endswith(("_pct", "_z"))]
    print(f"\n{'feature':<28} {'null%':>6} {'between-rack var share':>22} {'racks constant':>15}")
    for f in feats:
        s = ft.select("impression_id", pl.col(f).cast(pl.Float64))
        nullpct = 100.0 * s[f].null_count() / s.height
        s = s.drop_nulls(f)
        if s.is_empty():
            print(f"{f:<28} {nullpct:6.1f}  (all null)")
            continue
        g = s.group_by("impression_id").agg(pl.col(f).mean().alias("m"),
                                            pl.col(f).std().alias("sd"),
                                            pl.len().alias("n"))
        total_var = s[f].var()
        n = g["n"].to_numpy().astype(float)
        m = g["m"].to_numpy()
        grand = (n * m).sum() / n.sum()
        between = (n * (m - grand) ** 2).sum() / n.sum()
        share = between / total_var if total_var else float("nan")
        const = 100.0 * float((g["sd"].fill_null(0.0).to_numpy() == 0).mean())
        print(f"{f:<28} {nullpct:6.1f} {share:22.3f} {const:14.1f}%")


if __name__ == "__main__":
    main()
