"""slices + Q4.4 bootstrap CIs over the re-ranking metrics.

Reads the per-impression parquet `run_rerank_eval.py` wrote - which is why that
script kept the individual values instead of means - and produces the accuracy
half of the Q4 evaluation table: AUC, MRR, nDCG@5 and nDCG@10 for four methods,
on four slices, each with a bootstrap 95% confidence interval.

Slices (D26):
  all          every impression with a query (D17's headline population)
  cold / warm  history length <= 5 vs > 5. Absolute, not a per-dataset quantile:
               "cold start" means we know little about this user, which is not a
               relative rank. EB-NeRD's coldest user has more history than
               MIND's 25th percentile, so its cold slice is tiny - that
               asymmetry is a result, and a quantile would have hidden it.
  head / tail  by article popularity, under two definitions:
               exposure  - articles filling 50% of val impression slots, counted
                           from `candidate_article_ids` (what was SHOWN) and
                           never from clicks, or the slice would be circular
               train-pop - the textbook definition, articles carrying 50% of
                           training clicks. Reported to show it is degenerate
                           here: it leaves ~98% of val clicks in the tail.

Every metric in one cell shares a single resample draw (Q4.4), so the four
intervals describe the same resampled worlds rather than four independent ones.

    python scripts/run_eval_report.py [--datasets mind ebnerd] [--resamples 1000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.eval import bootstrap as boot  # noqa: E402
from newsrec.eval import rerank, slices  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
METHODS = ("random", "popularity", "bm25", "semantic")
METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def build_slices(dataset: str, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks over the impressions of `frame`, in its own row order."""
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == "val")
    )
    articles = pl.read_parquet(PROCESSED / "articles.parquet").filter(
        pl.col("dataset") == dataset
    )
    train_impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == dataset) & (pl.col("split") == "train")
    )
    article_ids = articles.get_column("article_id").to_list()

    # The parquet holds one row per (impression, method); the masks are per
    # impression, so they are built against the method-filtered order and the
    # caller applies them to each method's rows in that same order.
    order = frame.filter(pl.col("method") == METHODS[0]).get_column("impression_id").to_list()
    position = {imp: i for i, imp in enumerate(order)}
    if len(position) != len(order):
        raise ValueError("impression_ids are not unique within a method")

    clicked = [None] * len(order)
    for imp, clicks in zip(
        impressions.get_column("impression_id").to_list(),
        impressions.get_column("clicked_article_ids").to_list(),
    ):
        i = position.get(imp)
        if i is not None:
            clicked[i] = clicks
    if any(c is None for c in clicked):
        raise ValueError("an impression in the report is missing from the store")

    history_len = (
        frame.filter(pl.col("method") == METHODS[0]).get_column("history_len").to_numpy()
    )
    has_query = (
        frame.filter(pl.col("method") == METHODS[0]).get_column("has_query").to_numpy()
    )

    cold = slices.cold_start_mask(history_len) & has_query
    exposure_head = slices.head_set_from_counts(slices.exposure_counts(impressions))
    train_head = slices.train_popularity_head_set(
        rerank.train_click_counts(train_impressions, article_ids), article_ids
    )

    exp_head_mask, exp_tail_mask, exp_mixed = slices.head_tail_masks(clicked, exposure_head)
    tp_head_mask, tp_tail_mask, tp_mixed = slices.head_tail_masks(clicked, train_head)

    print(f"  slice sizes (of {has_query.sum():,} impressions with a query):")
    print(f"    cold (history <= 5) {cold.sum():7,}   warm {(has_query & ~cold).sum():7,}")
    print(f"    exposure head {(exp_head_mask & has_query).sum():7,}   tail "
          f"{(exp_tail_mask & has_query).sum():7,}   mixed-click {exp_mixed:,}")
    print(f"    train-pop head {(tp_head_mask & has_query).sum():7,}   tail "
          f"{(tp_tail_mask & has_query).sum():7,}   mixed-click {tp_mixed:,}"
          f"   <- degenerate, kept to show it")

    return {
        "all": has_query,
        "cold": cold,
        "warm": has_query & ~cold,
        "head-exposure": has_query & exp_head_mask,
        "tail-exposure": has_query & exp_tail_mask,
        "head-trainpop": has_query & tp_head_mask,
        "tail-trainpop": has_query & tp_tail_mask,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    parser.add_argument("--resamples", type=int, default=boot.DEFAULT_RESAMPLES)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-recent", type=int, default=10)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        path = REPORTS / f"rerank_{dataset}_{args.split}_n{args.n_recent}.parquet"
        if not path.exists():
            print(f"missing {path}; run scripts/run_rerank_eval.py first")
            return 1
        frame = pl.read_parquet(path)
        print(f"\n=== {dataset} ({args.split}) ===")
        masks = build_slices(dataset, frame)

        per_method = {m: frame.filter(pl.col("method") == m) for m in METHODS}

        for slice_name, mask in masks.items():
            n = int(mask.sum())
            if n == 0:
                print(f"\n  {slice_name}: empty, skipped")
                continue
            # One draw per slice, shared by every method and metric in it.
            draw = boot.draw_indices(n, args.resamples, boot.DEFAULT_SEED)
            print(f"\n  {slice_name} (n = {n:,})")
            print(f"    {'method':<11} " + " ".join(f"{m:>24}" for m in METRICS))
            for method in METHODS:
                cells, record = [], {
                    "dataset": dataset,
                    "slice": slice_name,
                    "method": method,
                    "n": n,
                }
                for metric in METRICS:
                    values = per_method[method].get_column(metric).to_numpy()[mask]
                    ci = boot.bootstrap_mean(values, indices=draw)
                    cells.append(f"{ci.point:6.4f} [{ci.low:6.4f},{ci.high:6.4f}]")
                    record[metric] = round(ci.point, 4)
                    record[f"{metric}_lo"] = round(ci.low, 4)
                    record[f"{metric}_hi"] = round(ci.high, 4)
                print(f"    {method:<11} " + " ".join(f"{c:>24}" for c in cells))
                rows.append(record)

    REPORTS.mkdir(exist_ok=True)
    name = f"eval_report_{'-'.join(args.datasets)}_{args.split}.csv"
    pl.DataFrame(rows).write_csv(REPORTS / name)
    print(f"\nwrote {REPORTS / name}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
