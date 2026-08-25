#!/usr/bin/env python3
"""generate a Codabench submission for one dataset.

    python scripts/run_submission.py --dataset mind
    python scripts/run_submission.py --dataset ebnerd --chunk-size 250000

Writes reports/submissions/{dataset}_{method}.txt and .zip.

The scoring is the *same* re-ranking task Q4.2 already built and measured - the
platform supplies the candidate list, we order it. Nothing here is a new model;
`score_semantic` and `build_user_vectors` are imported unchanged from the
modules Phases 3 and 4 tested. What is new is only that the split is unlabeled,
enormous, and has to be streamed.

Memory design, against ~2.5 GB free (D29):
  - impressions are never materialised; the frame is sliced in chunks and each
    chunk's lines are appended to the output file immediately. Measured: a
    50,000-row slice at offset 13,000,000 costs 0.28 s, so deep offsets are
    cheap and this is O(n), not O(n^2).
  - user vectors are built in batches into a **memmap on disk** rather than a
    resident array. EB-NeRD's 807,677 users x 384 float32 is 1.24 GB, which
    does not coexist with everything else.
  - scoring uses an article matrix restricted to the articles that actually
    appear as candidates - 10,451 of 125,541 for EB-NeRD, 30,043 of 120,961 for
    MIND. The score block is (batch_users x n_candidates), so this is a 12x cut
    on the dominant memory term, and it changes no result because an article
    that never appears as a candidate can never be scored.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from newsrec.eval.rerank import score_semantic  # noqa: E402
from newsrec.predict import (  # noqa: E402
    PREDICTION_FILENAME,
    build_submission_candidate_set,
    format_lines,
    zip_submission,
)
from newsrec.retrieval.semantic import load_article_embeddings  # noqa: E402
from newsrec.retrieval.semantic_search import UserVectors, build_user_vectors  # noqa: E402
from newsrec.submission import (  # noqa: E402
    SUBMISSION_SUBDIR,
    load_submission_behaviors,
    load_submission_history,
)

PROCESSED = REPO_ROOT / "data" / "processed" / SUBMISSION_SUBDIR
OUT_DIR = REPO_ROOT / "reports" / "submissions"
CONFIG_FOR = {"mind": "mind.yaml", "ebnerd": "ebnerd.yaml"}

N_RECENT = 10  # D12, unchanged from Q2/Q3/Q4


def load_test_root(dataset: str) -> Path:
    with open(REPO_ROOT / "configs" / CONFIG_FOR[dataset]) as f:
        return REPO_ROOT / yaml.safe_load(f)["test_root"]


def distinct_candidates(behaviors: pl.LazyFrame) -> list[str]:
    """Every article that appears as a candidate anywhere in the split.

    One streaming pass, measured at 1 s / 1.03 GB peak over all 13,536,710
    EB-NeRD test impressions. Worth its own pass because it shrinks the score
    matrix by an order of magnitude.
    """
    return (
        behaviors.select(pl.col("candidate_article_ids").explode().unique().alias("a"))
        .collect(engine="streaming")
        .get_column("a")
        .drop_nulls()
        .to_list()
    )


def build_user_vector_memmap(
    dataset: str,
    test_root: Path,
    article_ids: list[str],
    embeddings: np.ndarray,
    path: Path,
    batch_users: int = 100_000,
) -> UserVectors:
    """Mean-pool every test user's last-N history into an on-disk float32 memmap.

    Batched for the reason SCALE_NOTES records: `build_user_vectors` calls
    `.to_list()`, and doing that for 807,677 users at once - even truncated to
    10 articles each - materialises ~8 million Python strings alongside a
    1.24 GB output array. Batching bounds the transient part; the memmap bounds
    the persistent part.

    The vectors themselves are identical to the unbatched result: pooling is
    per-user and touches no other row.
    """
    history = load_submission_history(dataset, test_root, n_recent=N_RECENT)
    n_users = history.height

    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(n_users, embeddings.shape[1])
    )
    has_query = np.zeros(n_users, dtype=bool)
    user_ids: list[str] = []

    for start in range(0, n_users, batch_users):
        block = history.slice(start, batch_users)
        vecs = build_user_vectors(block, article_ids, embeddings, n_recent=N_RECENT)
        matrix[start : start + block.height] = vecs.matrix
        has_query[start : start + block.height] = vecs.has_query
        user_ids.extend(vecs.user_ids)

    matrix.flush()
    return UserVectors(user_ids=user_ids, matrix=matrix, has_query=has_query)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    parser.add_argument("--method", default="semantic", choices=["semantic"],
                        help="Q4.2 measured semantic best on both datasets "
                             "(MIND AUC 0.6338, EB-NeRD 0.5331); it is our best "
                             "honest system, so it is what we submit")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--score-batch", type=int, default=512,
                        help="users per score block; peak memory is a function "
                             "of this, not of impression count")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N impressions (smoke runs)")
    args = parser.parse_args()

    ds = args.dataset
    test_root = load_test_root(ds)
    emb_path = PROCESSED / f"embeddings_{ds}.parquet"
    if not emb_path.exists():
        print(f"error: {emb_path} not found.\n"
              f"Run: python scripts/build_embeddings.py --datasets {ds} "
              f"--articles data/processed/{SUBMISSION_SUBDIR}/articles_{ds}.parquet "
              f"--output {emb_path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1

    t_start = time.perf_counter()
    print(f"dataset      : {ds}   method: {args.method}")

    # --- corpus -----------------------------------------------------------
    t0 = time.perf_counter()
    all_ids, all_emb = load_article_embeddings(emb_path)
    print(f"embeddings   : {len(all_ids):,} x {all_emb.shape[1]}  "
          f"({time.perf_counter() - t0:.1f}s)")

    behaviors = load_submission_behaviors(ds, test_root)
    n_impressions = behaviors.select(pl.len()).collect(engine="streaming").item()
    if args.limit:
        behaviors = behaviors.slice(0, args.limit)
        n_impressions = min(n_impressions, args.limit)

    # --- restrict the scoring matrix to articles that can actually be scored
    t0 = time.perf_counter()
    cand_ids = distinct_candidates(behaviors)
    row_of = {a: i for i, a in enumerate(all_ids)}
    missing = [a for a in cand_ids if a not in row_of]
    if missing:
        print(f"error: {len(missing):,} candidate articles have no embedding, "
              f"e.g. {missing[0]!r}", file=sys.stderr)
        return 1
    cand_rows = np.fromiter((row_of[a] for a in cand_ids), dtype=np.int64,
                            count=len(cand_ids))
    cand_emb = np.ascontiguousarray(all_emb[cand_rows])
    print(f"candidates   : {len(cand_ids):,} distinct of {len(all_ids):,} corpus "
          f"({100 * len(cand_ids) / len(all_ids):.1f}%)  "
          f"({time.perf_counter() - t0:.1f}s)")

    # --- user vectors -----------------------------------------------------
    t0 = time.perf_counter()
    users = build_user_vector_memmap(
        ds, test_root, all_ids, all_emb,
        path=PROCESSED / f"user_vectors_{ds}.npy",
    )
    n_cold = int((~users.has_query).sum())
    print(f"user vectors : {len(users.user_ids):,}  "
          f"({n_cold:,} with no usable history = "
          f"{100 * n_cold / max(len(users.user_ids), 1):.1f}%)  "
          f"({time.perf_counter() - t0:.1f}s)")

    # all_emb is no longer needed - the scorer uses cand_emb, and user vectors
    # are already pooled. Freeing 193 MB before the streaming loop starts.
    del all_emb, row_of, cand_rows

    # --- stream ------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUT_DIR / f"{ds}_{args.method}.txt"

    written = 0
    t0 = time.perf_counter()
    with open(txt_path, "w", encoding="utf-8") as fh:
        for offset in range(0, n_impressions, args.chunk_size):
            chunk = behaviors.slice(offset, args.chunk_size).collect()
            if chunk.is_empty():
                break

            cs = build_submission_candidate_set(chunk, cand_ids)
            scores = score_semantic(cs, users, cand_emb, batch_size=args.score_batch)
            fh.write("\n".join(format_lines(cs.impression_ids, scores, ds)))
            fh.write("\n")

            written += chunk.height
            rate = written / max(time.perf_counter() - t0, 1e-9)
            print(f"  {written:>10,} / {n_impressions:,}  "
                  f"({100 * written / n_impressions:5.1f}%)  "
                  f"{rate:,.0f} impressions/s", flush=True)

    if written != n_impressions:
        print(f"error: wrote {written:,} rows for {n_impressions:,} impressions",
              file=sys.stderr)
        return 1

    # --- package -----------------------------------------------------------
    zip_path = zip_submission(
        txt_path, OUT_DIR / f"{ds}_{args.method}.zip", PREDICTION_FILENAME[ds]
    )
    print(f"\nwrote {written:,} lines -> {txt_path.relative_to(REPO_ROOT)} "
          f"({txt_path.stat().st_size / 1e6:.0f} MB)")
    print(f"zipped as {PREDICTION_FILENAME[ds]} -> {zip_path.relative_to(REPO_ROOT)} "
          f"({zip_path.stat().st_size / 1e6:.0f} MB)")
    print(f"total {(time.perf_counter() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
