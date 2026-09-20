"""reproduce NRMS, add ONE change (a time term), ablate it, paired bootstrap CI.

Arms (all identical except the candidate time inputs to the click score):
  nrms                  the reproduced baseline - no time input (D39)
  nrms+fresh            + log1p(freshness_hours)
  nrms+exposure         + exposure_share_1h, exposure_share_24h
  nrms+fresh+exposure   the improvement (D40): both
Train on train; the epoch is chosen by val AUC on a fixed seeded subsample of
val impressions; everything reported is on the local test split.

    python scripts/run_nrms.py --datasets mind [--epochs 4]
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from newsrec.eval import bootstrap, metrics  # noqa: E402
from newsrec.rank import nrms  # noqa: E402
from newsrec.retrieval import semantic  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
FEATS = PROCESSED / "features"
REPORTS = REPO_ROOT / "reports"
METRICS = ["AUC", "MRR", "nDCG@5", "nDCG@10"]
ARMS = {
    "nrms": [],
    "nrms+fresh": ["freshness_hours"],
    "nrms+exposure": ["exposure_share_1h", "exposure_share_24h"],
    "nrms+fresh+exposure": ["freshness_hours", "exposure_share_1h", "exposure_share_24h"],
}
VAL_SAMPLE = 30_000


def labels_of(b: nrms.Batchable) -> list[np.ndarray]:
    return [b.labels[b.cand_start[i]:b.cand_start[i + 1]] for i in range(len(b.cand_start) - 1)]


def sample_impressions(ft: pl.DataFrame, n: int) -> pl.DataFrame:
    ids = ft["impression_id"].unique(maintain_order=True)
    if len(ids) <= n:
        return ft
    keep = ids.sample(n, seed=nrms.SEED)
    return ft.filter(pl.col("impression_id").is_in(keep.implode()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mind"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--word-level", action="store_true",
                    help="D43: the paper's word-level news encoder over GloVe title "
                         "tokens, instead of D39's projection of the frozen sentence "
                         "embedding. Needs scripts/build_word_vocab.py to have run.")
    ap.add_argument("--tag", default="", help="suffix for the report and model filenames")
    ap.add_argument("--train-impressions", type=int, default=60_000,
                    help="seeded subsample of TRAIN impressions per arm (D41). 0 = all. "
                         "Measured 2026-09-19: a full MIND epoch is ~55 min on this CPU, "
                         "so 4 arms x 3 epochs would not fit the night.")
    args = ap.parse_args()
    torch.set_num_threads(8)

    for ds in args.datasets:
        hist = pl.read_parquet(PROCESSED / "history.parquet").filter(pl.col("dataset") == ds)
        word_emb = None
        if args.word_level:
            # `emb` stops being a matrix of article VECTORS and becomes a matrix of
            # article TOKEN IDS. Every downstream line is indifferent: prepare(),
            # train() and score() only ever gather rows out of it by article row and
            # hand the result to model.news, which is now the word encoder (D43).
            z = np.load(PROCESSED / f"words_{ds}.npz", allow_pickle=True)
            ids, emb, word_emb = list(z["article_ids"]), z["tokens"], z["emb"]
            print(f"{ds}: word-level, {word_emb.shape[0]:,} vocabulary x "
                  f"{word_emb.shape[1]} dims, titles padded to {emb.shape[1]} tokens",
                  flush=True)
        else:
            ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet",
                                                        dataset=ds)
        row = {a: i for i, a in enumerate(ids)}
        h = {s: hist.filter(pl.col("split") == s) for s in ("train", "val", "test")}
        ft = {"train": sample_impressions(pl.read_parquet(FEATS / f"{ds}_train.parquet"),
                                         args.train_impressions or 10**9),
              "val": sample_impressions(pl.read_parquet(FEATS / f"{ds}_val.parquet"), VAL_SAMPLE),
              "test": pl.read_parquet(FEATS / f"{ds}_test.parquet")}

        print(f"{ds}: train {ft['train']['impression_id'].n_unique():,} impressions / "
              f"{ft['train'].height:,} rows; val {ft['val']['impression_id'].n_unique():,}; "
              f"test {ft['test']['impression_id'].n_unique():,}", flush=True)
        per: dict[str, dict[str, np.ndarray]] = {}
        for arm in args.arms:
            cols = ARMS[arm]
            b = {s: nrms.prepare(ft[s], h[s], row, time_cols=cols or None) for s in ft}
            torch.manual_seed(nrms.SEED)
            model = nrms.NRMS(emb.shape[1], n_time=len(cols), word_embeddings=word_emb)
            best, best_auc = None, -1.0
            t0 = time.perf_counter()
            for ep in range(args.epochs):
                nrms.train(model, b["train"], emb, epochs=1, log=lambda m: print(f"{ds} {arm}{m}"))
                v = metrics.evaluate_impressions(nrms.score(model, b["val"], emb), labels_of(b["val"]))
                auc = float(np.nanmean(v.auc))
                print(f"{ds} {arm}: epoch {ep + 1} val AUC {auc:.4f} "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)
                if auc > best_auc:
                    best_auc, best = auc, copy.deepcopy(model.state_dict())
            model.load_state_dict(best)
            per[arm] = metrics.evaluate_impressions(
                nrms.score(model, b["test"], emb), labels_of(b["test"])).as_dict()
            torch.save(best, REPO_ROOT / "data" / "models" / f"nrms_{ds}_{arm}{args.tag}.pt")

        rows, idx = [], None
        for arm, vals in per.items():
            r = {"dataset": ds, "system": arm}
            for m in METRICS:
                if idx is None:
                    idx = bootstrap.draw_indices(len(vals[m]), bootstrap.DEFAULT_RESAMPLES,
                                                 bootstrap.DEFAULT_SEED)
                iv = bootstrap.bootstrap_mean(vals[m], indices=idx)
                r.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4), f"{m}_hi": round(iv.high, 4)})
            rows.append(r)
        if "nrms" in per:
            for arm in per:
                if arm == "nrms":
                    continue
                r = {"dataset": ds, "system": f"{arm} - nrms"}
                for m in METRICS:
                    iv = bootstrap.paired_bootstrap_diff(per[arm][m], per["nrms"][m])
                    r.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4), f"{m}_hi": round(iv.high, 4)})
                rows.append(r)
        out = pl.DataFrame(rows)
        path = REPORTS / f"nrms_ablation_{ds}_test{args.tag}.csv"
        out.write_csv(path)
        with pl.Config(tbl_cols=-1, tbl_width_chars=200, tbl_rows=20):
            print(out.select("system", *[c for m in METRICS for c in (m, f"{m}_lo", f"{m}_hi")]))
        print(f"-> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
