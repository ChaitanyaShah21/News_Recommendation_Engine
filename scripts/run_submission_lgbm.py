#!/usr/bin/env python3
"""the A2 leaderboard submission: the Q1 features + the Q2 LambdaRank model.

    python scripts/run_submission_lgbm.py --dataset mind
    python scripts/run_submission_lgbm.py --dataset ebnerd --only-groups 1   # smoke run

Writes reports/submissions/{dataset}_lgbm.txt and .zip; then run
scripts/validate_submission.py on it before uploading.

Scale: 93M (MIND) / 206M (EB-NeRD) candidate rows - never materialised at once.
The feature Context (BM25 index, first-seen times, the exposure index, the
session table) is built ONCE over everything a feature may read; rows are then
built, scored and written one chunk of impressions at a time, in file order.

WHAT COUNTS AS "THE PAST" HERE (D38)
The context is the local store (every split, for this dataset) plus the
leaderboard impressions themselves. Both leaderboard weeks start the second the
store ends (EB-NeRD 2023-06-01 07:00, MIND 2019-11-16 00:00), so the store is
exactly the history a live system would have had: the 24 h exposure window at
the start of the test week reads the store instead of an empty log. Only
existence of impressions is read, never a click - the store's labels are not
in any column this context touches.

Store impression ids are re-keyed "ctx:<split>:<id>" (MIND's train and dev files
reuse ids 1..N, and the leaderboard reuses them again), and every leaderboard
row gets an internal key "lb:<row>": EB-NeRD's file has 200,000 rows sharing
impression id 0.

EB-NeRD's 200,000 `is_beyond_accuracy` rows are one synthetic impression - the
same 250 articles, the same second, one per user - so they are kept OUT of the
context (they would read as 200,000 real exposures of those 250 articles) and
scored in non-strict mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.features import assemble  # noqa: E402
from newsrec.predict import PREDICTION_FILENAME, format_lines, zip_submission  # noqa: E402
from newsrec.rank.gbdt import feature_matrix  # noqa: E402
from newsrec.retrieval.semantic import load_article_embeddings  # noqa: E402
from newsrec.submission import (  # noqa: E402
    SUBMISSION_SUBDIR, load_submission_behaviors, load_submission_history)

PROCESSED = REPO_ROOT / "data" / "processed"
SUB = PROCESSED / SUBMISSION_SUBDIR
OUT_DIR = REPO_ROOT / "reports" / "submissions"
SPLIT = "leaderboard"
MAX_SESSION_MIN = 24 * 60  # the store's longest real session is 29 min


def context_columns(dataset: str) -> list[str]:
    cols = ["dataset", "impression_id", "user_id", "timestamp", "candidate_article_ids"]
    return cols + (["session_id", "device_type"] if dataset == "ebnerd" else [])


def prepare_leaderboard(ds: str, test_root: Path, groups: int) -> Path:
    """Stream the raw bundle once into a Parquet file with every column the run
    needs: `_r` (file row), an internal unique key, and `grp`, a user hash.

    Why: slicing the raw lazy frame at a deep offset re-scans everything before
    it once a row index is attached - measured 297 s for ONE 5,000-row chunk at
    offset 5M. A plain Parquet file slices and filters with pushdown instead.
    `grp` partitions by USER so every user's vectors are built exactly once
    (per-user work measured at ~3.4 ms; re-done per chunk it dominated)."""
    path = SUB / f"leaderboard_{ds}.parquet"
    if path.exists():
        return path
    lb = (load_submission_behaviors(ds, test_root).with_row_index("_r")
          .with_columns(pl.col("impression_id").alias("orig_id"),
                        (pl.lit("lb:") + pl.col("_r").cast(pl.Utf8)).alias("impression_id"),
                        pl.lit(SPLIT).alias("split"),
                        (pl.col("user_id").hash(seed=0) % groups).cast(pl.Int32).alias("grp")))
    if ds == "mind":
        lb = lb.with_columns(pl.lit(False).alias("is_beyond_accuracy"))
    lb.sink_parquet(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    ap.add_argument("--groups", type=int, default=64)
    ap.add_argument("--only-groups", type=int, default=0, help="smoke run: first N groups, no final file")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    ds = args.dataset
    t_start = time.perf_counter()

    model_path = Path(args.model) if args.model else REPO_ROOT / "data" / "models" / f"lgbm_{ds}.txt"
    booster = lgb.Booster(model_file=str(model_path))
    feats = assemble.FEATURES[ds]
    if booster.feature_name() != feats:
        raise SystemExit(f"model features {booster.feature_name()} != allowlist {feats}")

    with open(REPO_ROOT / "configs" / f"{ds}.yaml") as f:
        test_root = REPO_ROOT / yaml.safe_load(f)["test_root"]

    t0 = time.perf_counter()
    lb_path = prepare_leaderboard(ds, test_root, args.groups)
    lb = pl.scan_parquet(lb_path)
    n = lb.select(pl.len()).collect().item()
    print(f"leaderboard: {n:,} impressions ({time.perf_counter() - t0:.0f} s)", flush=True)

    # --- context: the store (re-keyed) + the real leaderboard impressions -----
    store = (pl.scan_parquet(PROCESSED / "impressions.parquet")
             .filter(pl.col("dataset") == ds)
             .with_columns((pl.lit("ctx:") + pl.col("split") + ":" + pl.col("impression_id"))
                           .alias("impression_id")))
    cols = context_columns(ds)
    context = pl.concat([store.select(cols),
                         lb.filter(~pl.col("is_beyond_accuracy")).select(cols)])
    arts = pl.read_parquet(SUB / f"articles_{ds}.parquet")
    emb_ids, emb = load_article_embeddings(SUB / f"embeddings_{ds}.parquet")
    t0 = time.perf_counter()
    ctx = assemble.build_context(ds, context, arts, emb_ids, emb)
    print(f"context built in {time.perf_counter() - t0:.0f} s "
          f"({len(ctx.exposure.shown_key):,} exposures, {len(ctx.exposure.imp_t):,} impressions)",
          flush=True)
    if ctx.sessions is not None:
        ctx.sessions = ctx.sessions.filter(pl.col("impression_id").str.starts_with("lb:"))
        longest = ctx.sessions["minutes_since_session_start"].max()
        print(f"longest leaderboard session: {longest:.1f} min", flush=True)
        if longest > MAX_SESSION_MIN:
            raise SystemExit("a (user, session) key spans more than a day - session ids "
                             "reused across weeks? Inspect before submitting.")

    t0 = time.perf_counter()
    history = load_submission_history(ds, test_root, n_recent=100).with_columns(
        pl.lit(SPLIT).alias("split"))
    print(f"history: {history.height:,} users ({time.perf_counter() - t0:.0f} s)", flush=True)

    # --- per user-group: build, score, save (resumable) -----------------------
    parts = SUB / f"lgbm_scores_{ds}"
    parts.mkdir(exist_ok=True)
    groups = range(args.only_groups or args.groups)
    done_imps = 0
    t0 = time.perf_counter()
    for g in groups:
        out = parts / f"{g:03d}.parquet"
        if out.exists():
            done_imps += pl.scan_parquet(out).select(pl.len()).collect().item()
            continue
        part = lb.filter(pl.col("grp") == g).collect()
        hist = history.filter(pl.col("user_id").is_in(part["user_id"].unique().implode()))
        gctx = dataclasses.replace(ctx, sessions=None if ctx.sessions is None else
                                   ctx.sessions.join(part.select("impression_id"),
                                                     on="impression_id", how="semi"))
        pieces = []
        for (is_ba,), sub in part.group_by("is_beyond_accuracy"):
            ft = assemble.build_rows(gctx, sub, hist, strict=not is_ba)
            sample = SUB / f"leaderboard_features_sample_{ds}.parquet"
            if not is_ba and not sample.exists():
                # Kept to compare against the local test table's distributions:
                # a feature silently null or shifted here would not error.
                ft.sample(min(ft.height, 300_000), seed=0).write_parquet(sample)
            s = booster.predict(feature_matrix(ft, feats))
            pieces.append(ft.select("impression_id").with_columns(pl.Series("score", s))
                          .group_by("impression_id", maintain_order=True).agg("score"))
        scored = pl.concat(pieces).join(part.select("impression_id", "_r", "candidate_article_ids"),
                                        on="impression_id", how="right")
        bad = scored.filter(pl.col("score").list.len().fill_null(-1)
                            != pl.col("candidate_article_ids").list.len())
        if bad.height:
            raise SystemExit(f"group {g}: {bad.height} impressions whose scores do not match "
                             f"their candidate list")
        scored.select("_r", "score").write_parquet(out)
        done_imps += part.height
        el = time.perf_counter() - t0
        print(f"  group {g + 1}/{args.groups}: {done_imps:,}/{n:,} impressions  "
              f"{el / 60:.1f} min elapsed", flush=True)
    if args.only_groups:
        print("smoke run: groups scored, no submission file written")
        return 0

    # --- write in file order ---------------------------------------------------
    scores = pl.scan_parquet(parts / "*.parquet").sort("_r").collect()
    if scores.height != n or not (scores["_r"] == pl.arange(0, n, eager=True)).all():
        raise SystemExit("scored rows do not cover the leaderboard exactly once")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUT_DIR / f"{ds}_lgbm.txt"
    ids = lb.select("orig_id").collect()["orig_id"]
    with open(txt_path, "w", encoding="utf-8") as fh:
        step = 500_000
        for off in range(0, n, step):
            sc = [np.asarray(x) for x in scores["score"].slice(off, step).to_list()]
            fh.write("\n".join(format_lines(ids.slice(off, step).to_list(), sc, ds)) + "\n")
    zip_submission(txt_path, txt_path.with_suffix(".zip"), PREDICTION_FILENAME[ds])
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min -> {txt_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
