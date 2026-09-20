"""Mutation-testing harness: plant a deliberate bug, require the tests to catch it.

A passing test suite proves the code does what the tests check. Mutation testing
asks the harder question — would these tests NOTICE if the code were wrong? A
mutation that SURVIVES is a gap in the tests, not a bug in the code.

`__pycache__` is cleared between mutations. The 2026-08-25 error-log entry
records a run where a stale bytecode cache made mutations look survived, because
restoring a file with `cp` did not make its mtime look newer to the import cache.
The failure direction is benign — a stale cache reports a FALSE survivor, which
gets investigated — but it wastes an hour.

Usage: import `run` and pass (name, old, new) triples.

    python scripts/diagnostics/mutate.py rack   # D42's 13 mutations
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: (source file, test file, [(description, find, replace), ...])
SUITES = {
    "rack": (
        REPO_ROOT / "src/newsrec/features/rack.py",
        "tests/test_rack_features.py",
        [
            ("pct off-by-one: (rank-1)/(n-1) -> rank/n",
             "(rank - 1.0) / (n - 1.0)", "rank / n"),
            ("singleton guard: n <= 1 -> n < 1",
             ".when(n <= 1).then(pl.lit(NEUTRAL_PCT))",
             ".when(n < 1).then(pl.lit(NEUTRAL_PCT))"),
            ("pct loses its rack: drop .over on rank",
             'rank = pl.col(f).rank("average").over("impression_id")',
             'rank = pl.col(f).rank("average")'),
            ("pct loses its rack: drop .over on count",
             'n = pl.col(f).count().over("impression_id")', 'n = pl.col(f).count()'),
            ("z loses its rack: drop .over on mean",
             'mu = pl.col(f).mean().over("impression_id")', 'mu = pl.col(f).mean()'),
            ("neutral percentile 0.5 -> 0.0", "NEUTRAL_PCT = 0.5", "NEUTRAL_PCT = 0.0"),
            ("constant rack unguarded: drop (sd == 0.0)",
             ".when(sd.is_null() | (sd == 0.0))", ".when(sd.is_null())"),
            ("nulls filled: drop the is_null guard in _pct",
             '.when(pl.col(f).is_null()).then(None)\n        .when(n <= 1)',
             '.when(n <= 1)'),
            ("nulls filled: drop the is_null guard in _z",
             '.when(pl.col(f).is_null()).then(None)\n        .when(sd.is_null()',
             '.when(sd.is_null()'),
            ('ties not shared: rank("average") -> rank("ordinal")',
             'rank("average")', 'rank("ordinal")'),
            ("z unscaled: drop the division by sd",
             ".otherwise((pl.col(f) - mu) / sd)", ".otherwise(pl.col(f) - mu)"),
            ("z uses variance not std", "sd = pl.col(f).std()", "sd = pl.col(f).var()"),
            ("RACK_BASE sweeps in an impression-level feature",
             '    "exposure_share_1h", "exposure_share_24h",\n]',
             '    "exposure_share_1h", "exposure_share_24h", "hours_since_last_click",\n]'),
        ],
    ),
}


def _clear_cache() -> None:
    for p in (REPO_ROOT / "src").rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)


def run(src: Path, test: str, mutants: list[tuple[str, str, str]]) -> int:
    backup = src.with_suffix(src.suffix + ".bak")
    shutil.copy(src, backup)
    original = backup.read_text()
    caught = survived = 0
    try:
        for name, old, new in mutants:
            if original.count(old) < 1:
                print(f"  SKIP (pattern absent): {name}")
                continue
            src.write_text(original.replace(old, new))
            _clear_cache()
            r = subprocess.run([".venv/bin/python", "-m", "pytest", test, "-q",
                                "--no-header", "-x"],
                               cwd=REPO_ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                caught += 1
                print(f"  caught   : {name}")
            else:
                survived += 1
                print(f"  SURVIVED : {name}")
    finally:
        shutil.copy(backup, src)
        backup.unlink()
        _clear_cache()
    print(f"\n{caught} caught, {survived} survived, of {caught + survived}")
    return 1 if survived else 0


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "rack"
    if which not in SUITES:
        raise SystemExit(f"unknown suite {which!r}; have {sorted(SUITES)}")
    src, test, mutants = SUITES[which]
    sys.exit(run(src, test, mutants))


if __name__ == "__main__":
    main()
