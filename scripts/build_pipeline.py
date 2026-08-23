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

# scripts/ is not part of the installed src/newsrec package, so it is not
# on Python's import path automatically - add src/ ourselves, relative to
# this file's own location, before importing anything from newsrec.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from newsrec.build import build_feature_store

OUTPUT_DIR = Path("data/processed")

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
    """Read `raw_root` out of one configs/*.yaml file."""
    with open(Path("configs") / config_filename) as f:
        config = yaml.safe_load(f)
    return Path(config["raw_root"])


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
