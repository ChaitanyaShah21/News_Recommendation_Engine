"""Q2, the second candidate regime (D37): retrieve top-K from the corpus, then re-rank.

Stage 1 is A1's candidate generator, unchanged: semantic retrieval (D20/D21),
query = last N=10 clicks (D12), the user's own history excluded (D15), searched
over articles already in circulation at the impression's hour (D19). K = 100.

Stage 2 is the Q2 LambdaRank model, trained on the supplied inview lists and
applied here without retraining - the distribution shift that implies is the
point of reporting this regime separately (D37).

"Before" = stage-1 order. "After" = stage-2 order over the SAME K articles.
Re-ranking a fixed set cannot change recall@K; it can change everything inside
it. Ranking metrics are defined only for impressions with at least one click
inside the K retrieved - A1 measured recall@200 at 5-9%, so that is a small
subset by construction (Landmine 8), and its size is reported beside the metrics.

    python scripts/run_retrieved_regime.py --datasets mind ebnerd
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.eval import bootstrap, metrics  # noqa: E402
from newsrec.features import assemble  # noqa: E402
from newsrec.rank import gbdt  # noqa: E402
from newsrec.retrieval import availability as avail_mod  # noqa: E402
from newsrec.retrieval import semantic, semantic_search  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
REPORTS = REPO_ROOT / "reports"
K = 100
N_RECENT_RETRIEVAL = 10  # D12 - A1's generator, not the re-ranker's N=100
METRICS = ["AUC", "MRR", "nDCG@5", "nDCG@10"]
CHUNK = 40_000


def retrieve(dataset: str, split: str, imps_all: pl.DataFrame, hist: pl.DataFrame,
             ids: list[str], emb: np.ndarray) -> pl.DataFrame:
    """The split's impressions with candidate_article_ids replaced by A1's top-K."""
    target = imps_all.filter(pl.col("split") == split)
    h = hist.filter(pl.col("split") == split)
    users = semantic_search.build_user_vectors(h, ids, emb, n_recent=N_RECENT_RETRIEVAL)
    row_of = {a: i for i, a in enumerate(ids)}
    hist_of = dict(zip(h["user_id"], h["history_article_ids"].to_list()))
    exclude = [np.asarray([row_of[a] for a in (hist_of.get(u) or []) if a in row_of], np.int32)
               for u in users.user_ids]
    bucketed, bucket_id, masks = avail_mod.build_availability(imps_all, target, ids, "1h")
    tasks = bucketed.select("user_id", "bucket_start").unique(maintain_order=True)
    q_of = {u: i for i, u in enumerate(users.user_ids)}
    t_user = tasks["user_id"].to_list()
    top = semantic_search.retrieve_bucketed(
        users, emb, np.array([q_of[u] for u in t_user], np.int64),
        np.array([bucket_id[b] for b in tasks["bucket_start"].to_list()], np.int64),
        masks, k=K, exclude_rows=exclude)
    key = {(u, b): i for i, (u, b) in enumerate(zip(t_user, tasks["bucket_start"].to_list()))}
    lists = [[ids[r] for r in top[key[(u, b)]]]
             for u, b in zip(bucketed["user_id"].to_list(), bucketed["bucket_start"].to_list())]
    return bucketed.drop("bucket_start").with_columns(
        pl.Series("candidate_article_ids", lists, dtype=pl.List(pl.Utf8)))


def run(dataset: str, split: str = "test", models=("base",)) -> pl.DataFrame:
    imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == dataset)
    hist = pl.read_parquet(PROCESSED / "history.parquet").filter(pl.col("dataset") == dataset)
    arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == dataset)
    ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset=dataset)

    t0 = time.perf_counter()
    target = retrieve(dataset, split, imps, hist, ids, emb)
    n_all = target.height
    recall = target.select(
        (pl.col("clicked_article_ids").list.set_intersection("candidate_article_ids").list.len()
         / pl.col("clicked_article_ids").list.len()).alias("r"))["r"]
    target = target.filter(pl.col("candidate_article_ids").list.len() > 0)
    print(f"{dataset}: retrieved top-{K} for {target.height:,}/{n_all:,} impressions "
          f"in {time.perf_counter() - t0:.0f} s; macro recall@{K} {recall.mean():.4f}", flush=True)

    # A1's retrieval order as a score: rank 0 best. Kept per impression, in list order.
    ctx = assemble.build_context(dataset, imps, arts, ids, emb)
    # D42c: every requested model scores the SAME retrieved candidates, so the gap
    # between them prices what a rack-relative feature costs when its reference set
    # changes. The models were trained on inview racks (MIND median 25, EB-NeRD
    # median 9) and are asked here to read percentiles over K = 100 retrieved ones.
    boosters = {}
    for tag in models:
        suffix = "_rack" if tag == "rack" else ""
        bst = lgb.Booster(model_file=str(REPO_ROOT / "data" / "models"
                                         / f"lgbm_{dataset}{suffix}.txt"))
        want = bst.feature_name()
        if want not in (assemble.FEATURES[dataset], assemble.ALL_FEATURES[dataset]):
            raise SystemExit(f"lgbm_{dataset}{suffix} uses an unsanctioned feature set")
        boosters[tag] = (bst, want)

    before, labels = [], []
    after = {t: [] for t in boosters}
    for off in range(0, target.height, CHUNK):
        part = target.slice(off, CHUNK)
        ft = assemble.build_rows(ctx, part, hist)  # strict: every candidate was in circulation
        chunk_y = None
        for tag, (bst, want) in boosters.items():
            _, sa, chunk_y = gbdt.per_impression(ft, bst.predict(
                gbdt.feature_matrix(ft, want)))
            after[tag] += sa
        labels += chunk_y
        # Retrieval order, rank 0 best. Lengths come from this chunk's own
        # per-impression split, not from a previous one.
        before += [-np.arange(len(x), dtype=np.float64) for x in chunk_y]
    b = metrics.evaluate_impressions(before, labels).as_dict()
    per = {t: metrics.evaluate_impressions(v, labels).as_dict() for t, v in after.items()}
    a = per[next(iter(boosters))]
    n_def = int((~np.isnan(a["MRR"])).sum())

    rows = []
    named = [("retrieval order (A1 semantic)", b)]
    named += [(f"lgbm{'' if t == 'base' else '+rack'} re-ranked", per[t]) for t in boosters]
    for name, vals in named:
        row = {"dataset": dataset, "system": name, "k": K, "n_impressions": n_all,
               "n_with_a_hit": n_def, f"recall@{K}": round(float(recall.mean()), 4)}
        for m in METRICS:
            iv = bootstrap.bootstrap_mean(vals[m])
            row.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4), f"{m}_hi": round(iv.high, 4)})
        rows.append(row)
    diffs = [(f"lgbm{'' if t == 'base' else '+rack'} - retrieval order", per[t], b)
             for t in boosters]
    if "base" in boosters and "rack" in boosters:
        diffs.append(("rack - base (paired, retrieved regime)", per["rack"], per["base"]))
    for name, x, y in diffs:
        row = {"dataset": dataset, "system": name, "k": K, "n_impressions": n_all,
               "n_with_a_hit": n_def, f"recall@{K}": None}
        for m in METRICS:
            iv = bootstrap.paired_bootstrap_diff(x[m], y[m])
            row.update({m: round(iv.point, 4), f"{m}_lo": round(iv.low, 4),
                        f"{m}_hi": round(iv.high, 4)})
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mind", "ebnerd"])
    ap.add_argument("--models", nargs="+", default=["base"], choices=["base", "rack"],
                    help="D42c: pass both to price what rack features cost when the "
                         "candidate set changes from inview lists to retrieved top-K.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    for ds in args.datasets:
        out = run(ds, models=tuple(args.models))
        path = REPORTS / f"retrieved_regime_{ds}_test_k{K}{args.tag}.csv"
        out.write_csv(path)
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(out.select("system", "n_with_a_hit", *METRICS))
        print(f"-> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
