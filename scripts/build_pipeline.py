#!/usr/bin/env python3
"""One-command rebuild (Q1.5).

Reads raw-data locations from configs/mind.yaml and configs/ebnerd.yaml,
checks the raw files are actually present (printing manual download
instructions and exiting cleanly if not - MIND is gated, see PROGRESS.md),
then builds the feature store into data/processed/.

Usage:
    python scripts/build_pipeline.py
"""

import sys
from pathlib import Path

import yaml

# Every path in this script is anchored to REPO_ROOT, computed once here from
# this file's own location - not to the shell's current working directory.
# Found the hard way (Chaitanya asked, then we tested it): the sys.path fix
# below only made the *import* work from any directory; configs/mind.yaml and
# data/processed were still being resolved against the caller's cwd, so
# running this from outside the repo root failed with a plain
# FileNotFoundError even though the import succeeded. Fixed by using
# REPO_ROOT everywhere, not just for src/.
REPO_ROOT = Path(__file__).resolve().parent.parent

# scripts/ is not part of the installed src/newsrec package, so it is not
# on Python's import path automatically - add src/ ourselves before
# importing anything from newsrec.
sys.path.insert(0, str(REPO_ROOT / "src"))

from newsrec.build import build_feature_store

OUTPUT_DIR = REPO_ROOT / "data" / "processed"

MIND_DOWNLOAD_HELP = """
MIND-small raw files not found at {raw_root}

MIND is a gated HuggingFace dataset:
  1. Create a free account at huggingface.co (if you don't have one)
  2. Visit https://huggingface.co/datasets/yjw1029/MIND and request access
  3. Download MINDsmall_train.zip and MINDsmall_dev.zip
  4. Unzip into {raw_root}/MINDsmall_train/ and {raw_root}/MINDsmall_dev/
"""

EBNERD_DOWNLOAD_HELP = """
EB-NeRD-demo raw files not found at {raw_root}

Direct download, no login required:
  wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip

Unzip into {raw_root}/ (should contain articles.parquet, train/, validation/)
"""


def load_raw_root(config_filename: str) -> Path:
    """Read `raw_root` out of one configs/*.yaml file. The config's own
    `raw_root` value (e.g. "data/raw/mind") is written as repo-root-relative,
    so it gets resolved against REPO_ROOT too - not left relative to
    whatever the caller's cwd happens to be."""
    with open(REPO_ROOT / "configs" / config_filename) as f:
        config = yaml.safe_load(f)
    return REPO_ROOT / config["raw_root"]


def check_raw_data(mind_root: Path, ebnerd_root: Path) -> None:
    """Fail loudly and helpfully - not with a bare FileNotFoundError deep
    inside pl.read_csv - if raw data isn't where the configs say it is."""
    problems = []
    if not (mind_root / "MINDsmall_train" / "news.tsv").exists():
        problems.append(MIND_DOWNLOAD_HELP.format(raw_root=mind_root))
    if not (ebnerd_root / "articles.parquet").exists():
        problems.append(EBNERD_DOWNLOAD_HELP.format(raw_root=ebnerd_root))
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)


def main() -> None:
    mind_root = load_raw_root("mind.yaml")
    ebnerd_root = load_raw_root("ebnerd.yaml")

    check_raw_data(mind_root, ebnerd_root)

    print(f"Building feature store -> {OUTPUT_DIR} ...")
    build_feature_store(
        mind_root=mind_root, ebnerd_root=ebnerd_root, output_dir=OUTPUT_DIR
    )
    print("Done.")


if __name__ == "__main__":
    main()
