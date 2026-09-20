"""all seven metrics for the full two-stage pipeline, sliced, with bootstrap CIs.

The pipeline graded here is the one both leaderboards grade: the platform's
supplied candidate list, ordered by the Q2 LambdaRank model, against A1's two
scorers and a random arm on the same lists (so "before and after" is one table).
The retrieved-top-K regime is reported separately by run_retrieved_regime.py
(D37), because its candidate sets - and therefore its metrics - are not
comparable to these.

  accuracy         AUC, MRR, nDCG@5, nDCG@10, per impression
  beyond-accuracy  intra-list diversity (embedding and category), novelty,
                   coverage, over each system's top-K
  slices (D26)     all / cold vs warm (history <= 5) / head vs tail exposure
  intervals        1,000-resample percentile CIs, one shared draw per slice.
                   Coverage carries its spread, not a CI (D27).

    python scripts/run_extended_eval.py [--datasets mind ebnerd] [--k 10]

Reads data/processed/features/{ds}_test.parquet and the saved model scores from
data/processed/scores/lgbm_{ds}_test.parquet, so it never retrains or rebuilds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.eval import beyond_accuracy as ba  # noqa: E402
from newsrec.eval import bootstrap as boot  # noqa: E402
from newsrec.eval import metrics as met  # noqa: E402
from newsrec.eval import rerank, slices  # noqa: E402
from newsrec.rank import gbdt  # noqa: E402
from newsrec.retrieval import semantic  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
METRICS = ["AUC", "MRR", "nDCG@5", "nDCG@10"]
SPLIT = "test"


def _saved(dataset: str, tag: str, ft: pl.DataFrame) -> np.ndarray:
    saved = pl.read_parquet(PROCESSED / "scores" / f"lgbm_{dataset}{tag}_{SPLIT}.parquet")
    if saved.height != ft.height or not (saved["article_id"] == ft["article_id"]).all():
        raise ValueError(f"lgbm_{dataset}{tag} scores are not row-aligned with the "
                         "feature table - rebuild them with run_reranker.py")
    return saved["score"].to_numpy()


def system_scores(ft: pl.DataFrame, dataset: str,
                  extra_tags: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Row-aligned scores for every system, on the SAME candidate lists.

    `extra_tags` adds further saved models as their own systems - D42's rack arm
    is `_rack`. Every system is graded on identical candidate lists, so the
    paired bootstrap that follows is comparing orderings and nothing else.
    """
    out = {
        "random": np.random.default_rng(gbdt.SEED).random(ft.height),
        "bm25 (A1)": ft["bm25"].fill_null(0.0).to_numpy(),
        "semantic (A1)": ft["cos_inf"].fill_null(0.0).to_numpy(),
        "lgbm (A2)": _saved(dataset, "", ft),
    }
    for tag in extra_tags:
        out[f"lgbm{tag} (A2)"] = _saved(dataset, tag, ft)
    return out


def top_k_rows(ft: pl.DataFrame, scores: np.ndarray, row_of: dict, k: int) -> list[np.ndarray]:
    """Each impression's top-k candidates, as row indices into the article arrays."""
    sizes = gbdt.group_sizes(ft["impression_id"])
    bounds = np.cumsum(sizes)
    art = np.fromiter((row_of[a] for a in ft["article_id"]), np.int64, ft.height)
    out, start = [], 0
    for end in bounds:
        s = scores[start:end]
        order = np.argsort(-s, kind="stable")[:k]
        out.append(art[start:end][order].astype(np.int32))
        start = end
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--extra-models", nargs="*", default=[],
                    help="further saved model tags to grade alongside, e.g. _rack (D42). "
                         "They are scored on the same candidate lists as everything else.")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args()

    for ds in args.datasets:
        ft = pl.read_parquet(PROCESSED / "features" / f"{ds}_{SPLIT}.parquet")
        imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == ds)
        target = imps.filter(pl.col("split") == SPLIT)
        hist = pl.read_parquet(PROCESSED / "history.parquet").filter(
            (pl.col("dataset") == ds) & (pl.col("split") == SPLIT))
        arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == ds)
        ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset=ds)
        row_of = {a: i for i, a in enumerate(ids)}
        cats = (pl.DataFrame({"article_id": ids})
                .join(arts.select("article_id", "category"), on="article_id", how="left")
                ["category"].fill_null("<none>").to_numpy())
        novelty = ba.item_novelty(rerank.train_click_counts(
            imps.filter(pl.col("split") == "train"), ids))

        # --- per-impression alignment: impression order of the feature table ---
        imp_ids = ft["impression_id"].rle().struct.field("value").to_list()
        order = pl.DataFrame({"impression_id": imp_ids}).join(
            target.select("impression_id", "user_id", "clicked_article_ids"),
            on="impression_id", how="left")
        hist_len = dict(zip(hist["user_id"], hist["history_article_ids"].list.len()))
        history_len = np.array([hist_len.get(u, 0) for u in order["user_id"]], np.int64)
        clicked = order["clicked_article_ids"].to_list()

        cold = slices.cold_start_mask(history_len)
        head_set = slices.head_set_from_counts(slices.exposure_counts(target))
        head, tail, n_mixed = slices.head_tail_masks(clicked, head_set)
        masks = {"all": np.ones(len(imp_ids), bool), "cold": cold, "warm": ~cold,
                 "head-exposure": head, "tail-exposure": tail}
        print(f"{ds}: {len(imp_ids):,} impressions; cold {cold.sum():,}, "
              f"head {head.sum():,}, tail {tail.sum():,}, mixed (neither) {n_mixed:,}", flush=True)

        systems = system_scores(ft, ds, tuple(args.extra_models))
        acc, lists = {}, {}
        for name, s in systems.items():
            _, per_imp, labels = gbdt.per_impression(ft, s)
            acc[name] = met.evaluate_impressions(per_imp, labels).as_dict()
            lists[name] = top_k_rows(ft, s, row_of, args.k)
        beyond = {name: ba.evaluate_lists(v, emb, cats, novelty, len(ids))
                  for name, v in lists.items()}

        rows = []
        for slice_name, mask in masks.items():
            n = int(mask.sum())
            if n == 0:
                continue
            idx = boot.draw_indices(n, boot.DEFAULT_RESAMPLES, boot.DEFAULT_SEED)
            for name in systems:
                r = {"dataset": ds, "slice": slice_name, "system": name, "n": n, "k": args.k}
                for m in METRICS:
                    iv = boot.bootstrap_mean(acc[name][m][mask], indices=idx)
                    r.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4),
                              f"{m}_hi": round(iv.high, 4)})
                for m in ("ild_embedding", "ild_category", "novelty"):
                    iv = boot.bootstrap_mean(np.asarray(beyond[name][m])[mask], indices=idx)
                    r.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4),
                              f"{m}_hi": round(iv.high, 4)})
                sel = [lists[name][i] for i in np.flatnonzero(mask)]
                cov = boot.bootstrap_coverage(sel, len(ids), indices=idx)
                r.update({"coverage": round(cov.point, 4),
                          "coverage_spread_lo": round(cov.low, 4),
                          "coverage_spread_hi": round(cov.high, 4)})
                rows.append(r)
            r = {"dataset": ds, "slice": slice_name, "system": "lgbm - semantic (A1) [paired]",
                 "n": n, "k": args.k}
            for m in METRICS:
                iv = boot.paired_bootstrap_diff(acc["lgbm (A2)"][m][mask],
                                               acc["semantic (A1)"][m][mask])
                r.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4),
                          f"{m}_hi": round(iv.high, 4)})
            rows.append(r)

        out = pl.DataFrame(rows, infer_schema_length=None)
        path = REPORTS / f"extended_eval_{ds}_{SPLIT}_k{args.k}{args.tag}.csv"
        out.write_csv(path)
        with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=220):
            print(out.filter(pl.col("slice") == "all").select(
                "system", *METRICS, "ild_category", "novelty", "coverage"))
        print(f"-> {path.relative_to(REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
