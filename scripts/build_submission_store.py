#!/usr/bin/env python3
"""Phase 5 / Q5 - build the submission-side store from the leaderboard test bundles.

Reads `test_root` from configs/{mind,ebnerd}.yaml and writes the unified-schema
test corpora into data/processed/submission/. Deliberately a separate command
from `build_pipeline.py`: see src/newsrec/submission.py for why the leaderboard
test set must not enter the feature store's `split == "test"`.

Usage:
    python scripts/build_submission_store.py                   # both datasets
    python scripts/build_submission_store.py --datasets mind
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from newsrec.submission import (  # noqa: E402
    SUBMISSION_SUBDIR,
    count_empty_texts,
    load_submission_articles,
)

OUTPUT_DIR = REPO_ROOT / "data" / "processed" / SUBMISSION_SUBDIR

CONFIG_FOR = {"mind": "mind.yaml", "ebnerd": "ebnerd.yaml"}

# What each bundle must contain for us to proceed, and what to say if it doesn't.
EXPECTED_FILE = {"mind": "news.tsv", "ebnerd": "articles.parquet"}

DOWNLOAD_HELP = {
    "mind": """
MINDlarge_test not found at {test_root}

  1. Visit https://huggingface.co/datasets/yjw1029/MIND and request access
  2. Download MINDlarge_test.zip
  3. Unzip into {test_root}/ (should contain news.tsv, behaviors.tsv)
""",
    "ebnerd": """
ebnerd_testset not found at {test_root}

  wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip

Unzip it, then point configs/ebnerd.yaml's test_root at the INNER directory -
the archive extracts to ebnerd_testset/ebnerd_testset/ (should contain
articles.parquet and test/).
""",
}


def load_test_root(dataset: str) -> Path:
    """Read `test_root` out of one config, resolved against REPO_ROOT - never
    against the caller's cwd. Same anchoring rule as build_pipeline.py, which
    shipped that exact bug once already."""
    with open(REPO_ROOT / "configs" / CONFIG_FOR[dataset]) as f:
        config = yaml.safe_load(f)
    if "test_root" not in config:
        raise KeyError(
            f"configs/{CONFIG_FOR[dataset]} has no `test_root` key - "
            f"Phase 5 added it; is this an older checkout?"
        )
    return REPO_ROOT / config["test_root"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    args = parser.parse_args()

    # Check every requested bundle BEFORE doing any work, so a missing second
    # dataset doesn't surface only after the first one has been written.
    roots = {}
    problems = []
    for dataset in args.datasets:
        root = load_test_root(dataset)
        roots[dataset] = root
        if not (root / EXPECTED_FILE[dataset]).exists():
            problems.append(DOWNLOAD_HELP[dataset].format(test_root=root))
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        t0 = time.perf_counter()
        articles = load_submission_articles(dataset, roots[dataset])
        n_blank = count_empty_texts(articles)

        out = OUTPUT_DIR / f"articles_{dataset}.parquet"
        articles.write_parquet(out)

        print(
            f"{dataset:<7} {articles.height:>8,} articles  "
            f"({n_blank} with no title and no abstract)  "
            f"{time.perf_counter() - t0:5.1f}s  -> {out.relative_to(REPO_ROOT)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
