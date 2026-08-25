#!/usr/bin/env python3
"""check a submission file before uploading it.

    python scripts/validate_submission.py --dataset mind
    python scripts/validate_submission.py --dataset ebnerd --reference some.zip

Every check here exists because the corresponding mistake produces a file that
looks fine. A submission cannot be debugged from the leaderboard: a wrong rank
direction, a dropped row, or a shuffled order all come back as one number that
is merely worse than expected.

Checked, in order of how quietly each fails:

  1. **Row order and identity.** Line i must carry impression i's own id, read
     from the raw bundle. Both competitions require "the row orders of the
     results should be consistent with those in the original files"; a chunked
     writer that emits chunks out of order satisfies every other check here.
  2. **Rank vectors are permutations of 1..n**, with n the number of candidates
     that impression actually had. Catches a truncated list, a 0-indexed list,
     duplicated ranks, and a rank vector built against the wrong impression.
  3. **Line syntax** - `<int> [<int>,<int>,...]`, no spaces inside the bracket.
  4. **The zip** contains exactly one flat entry with the competition's exact
     filename (`prediction.txt` for MIND, `predictions.txt` for EB-NeRD).
  5. Optionally, agreement with an official example file: same line count and
     the same impression ids in the same order.

What this deliberately does NOT check is whether the ranking is any *good* -
that is what the val-split evaluation in Q4 was for.
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from newsrec.predict import ID_PREFIX, PREDICTION_FILENAME  # noqa: E402
from newsrec.submission import load_submission_behaviors  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "submissions"
CONFIG_FOR = {"mind": "mind.yaml", "ebnerd": "ebnerd.yaml"}

LINE_RE = re.compile(r"^(\d+) \[(\d+(?:,\d+)*)\]$")


def load_test_root(dataset: str) -> Path:
    with open(REPO_ROOT / "configs" / CONFIG_FOR[dataset]) as f:
        return REPO_ROOT / yaml.safe_load(f)["test_root"]


def check_zip(zip_path: Path, dataset: str, problems: list[str]) -> None:
    expected = PREDICTION_FILENAME[dataset]
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    if names != [expected]:
        problems.append(
            f"zip must contain exactly ['{expected}'], contains {names}. "
            f"Both competitions reject folders and __MACOSX entries."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    parser.add_argument("--method", default="semantic")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--reference", type=Path, default=None,
                        help="official example .zip to cross-check ids against")
    args = parser.parse_args()

    ds = args.dataset
    txt_path = OUT_DIR / f"{ds}_{args.method}.txt"
    zip_path = OUT_DIR / f"{ds}_{args.method}.zip"
    if not txt_path.exists():
        print(f"error: {txt_path} not found - run scripts/run_submission.py first",
              file=sys.stderr)
        return 1

    problems: list[str] = []
    prefix = ID_PREFIX[ds]

    if zip_path.exists():
        check_zip(zip_path, ds, problems)
    else:
        problems.append(f"{zip_path} not found")

    behaviors = load_submission_behaviors(ds, load_test_root(ds))
    n_impressions = behaviors.select(pl.len()).collect(engine="streaming").item()

    n_lines = 0
    n_bad_syntax = 0
    n_bad_id = 0
    n_bad_perm = 0
    first_rank_hist: dict[int, int] = {}

    with open(txt_path, encoding="utf-8") as fh:
        for offset in range(0, n_impressions, args.chunk_size):
            chunk = behaviors.slice(offset, args.chunk_size).collect()
            if chunk.is_empty():
                break
            exp_ids = chunk.get_column("impression_id").to_list()
            exp_lens = chunk.get_column("candidate_article_ids").list.len().to_list()

            for exp_id, exp_len in zip(exp_ids, exp_lens):
                line = fh.readline()
                if not line:
                    break
                n_lines += 1
                m = LINE_RE.match(line.rstrip("\n"))
                if m is None:
                    if n_bad_syntax < 3:
                        problems.append(f"line {n_lines}: bad syntax {line[:80]!r}")
                    n_bad_syntax += 1
                    continue

                got_id, rank_str = m.group(1), m.group(2)
                if got_id != exp_id.removeprefix(prefix):
                    if n_bad_id < 3:
                        problems.append(
                            f"line {n_lines}: impression id {got_id!r}, "
                            f"bundle row {n_lines} is {exp_id.removeprefix(prefix)!r} "
                            f"- rows are out of order or one was dropped"
                        )
                    n_bad_id += 1
                    continue

                ranks = [int(x) for x in rank_str.split(",")]
                if sorted(ranks) != list(range(1, exp_len + 1)):
                    if n_bad_perm < 3:
                        problems.append(
                            f"line {n_lines} (impression {got_id}): ranks are not a "
                            f"permutation of 1..{exp_len}; got {len(ranks)} values"
                        )
                    n_bad_perm += 1
                    continue

                # Distribution of which POSITION won rank 1. A scorer that
                # always picks position 1 (or always the last) is a strong hint
                # that scores are constant or the vector is misbuilt - it is not
                # an error, so it is reported rather than raised.
                winner = ranks.index(1)
                first_rank_hist[winner] = first_rank_hist.get(winner, 0) + 1

        if fh.readline():
            problems.append("submission has MORE lines than the bundle has impressions")

    if n_lines != n_impressions:
        problems.append(
            f"line count {n_lines:,} != impression count {n_impressions:,}"
        )

    print(f"dataset            : {ds}")
    print(f"impressions        : {n_impressions:,}")
    print(f"lines              : {n_lines:,}")
    print(f"bad syntax         : {n_bad_syntax:,}")
    print(f"wrong id / order   : {n_bad_id:,}")
    print(f"not a permutation  : {n_bad_perm:,}")

    top = sorted(first_rank_hist.items(), key=lambda kv: -kv[1])[:5]
    total = sum(first_rank_hist.values()) or 1
    print("rank-1 position    : " + ", ".join(
        f"pos{p}={100 * c / total:.1f}%" for p, c in top))

    if args.reference and args.reference.exists():
        with zipfile.ZipFile(args.reference) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as ref:
                ref_n = 0
                mismatches = 0
                with open(txt_path, encoding="utf-8") as ours:
                    for ref_line in ref:
                        our_line = ours.readline()
                        if not our_line:
                            break
                        ref_n += 1
                        r_id = ref_line.decode().split(" ", 1)[0]
                        o_id = our_line.split(" ", 1)[0]
                        if r_id != o_id and mismatches < 3:
                            problems.append(
                                f"reference line {ref_n}: id {r_id} vs ours {o_id}")
                        mismatches += r_id != o_id
        print(f"reference lines    : {ref_n:,}  id mismatches: {mismatches:,}")
        if ref_n != n_lines:
            problems.append(
                f"reference has {ref_n:,} lines, ours has {n_lines:,}")

    if problems:
        print(f"\nFAILED - {len(problems)} problem(s):", file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("\nOK - safe to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
