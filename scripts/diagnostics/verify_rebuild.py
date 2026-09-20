"""Prove a feature-table rebuild ADDED columns without CHANGING any.

Run after rebuilding `data/processed/features/` against a saved copy of the
previous tables. This is the check that caught the reused-MIND-impression-id bug
on 2026-09-19, which is why it is run rather than reasoned about.

    mv data/processed/features data/processed/features_pre_rack
    python scripts/build_feature_tables.py
    python scripts/diagnostics/verify_rebuild.py [old_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from newsrec.features.assemble import FEATURES, KEYS, LABEL, RACK_FEATURES  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"


def main() -> None:
    old = Path(sys.argv[1]) if len(sys.argv) > 1 else PROCESSED / "features_pre_rack"
    new = PROCESSED / "features"
    if not old.exists():
        raise SystemExit(f"no previous tables at {old} - nothing to compare against")
    bad = 0
    for ds in ("mind", "ebnerd"):
        base = KEYS + [LABEL] + FEATURES[ds]
        for split in ("train", "val", "test"):
            try:
                o = pl.read_parquet(old / f"{ds}_{split}.parquet")
                n = pl.read_parquet(new / f"{ds}_{split}.parquet")
            except FileNotFoundError as e:
                print(f"{ds} {split}: MISSING {e}")
                bad += 1
                continue
            added = [c for c in n.columns if c not in o.columns]
            identical = o.select(base).equals(n.select(base))
            print(f"{ds:7s} {split:5s}: rows {o.height:>10,} -> {n.height:>10,} "
                  f"{'OK' if o.height == n.height else 'CHANGED'}; base columns "
                  f"{'IDENTICAL' if identical else '*** DIFFER ***'}; +{len(added)} new")
            if not identical:
                bad += 1
                for c in base:
                    if not o[c].equals(n[c]):
                        print(f"           {c}: {(o[c] != n[c]).sum():,} rows differ")
            if sorted(added) != sorted(RACK_FEATURES[ds]):
                print(f"           unexpected new columns: "
                      f"{sorted(set(added) ^ set(RACK_FEATURES[ds]))}")
                bad += 1
    print("\nVERDICT:", "clean - rebuild adds columns only" if not bad else f"{bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
