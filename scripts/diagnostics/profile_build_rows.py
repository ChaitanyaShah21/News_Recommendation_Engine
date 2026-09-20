"""D44's finding: where the BATCH feature path spends a single request's time.

The answer that motivated `newsrec/serving.py`: on MIND, 52% of the time is
inside `PyLazyFrame.collect` — 66 separate Polars query collections for a table
of 100 candidates — and 23% in joins against tables of 65,238 (MIND) or
1,219,746 (EB-NeRD session) rows. None of it is the arithmetic.

Relative attribution is the reading. Absolute milliseconds belong to
`scripts/benchmark_serving.py`, which times both paths in one process.

    python scripts/diagnostics/profile_build_rows.py [dataset] [n_requests]
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402

from newsrec.features import assemble  # noqa: E402
from newsrec.retrieval import semantic  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"


def main() -> None:
    ds = sys.argv[1] if len(sys.argv) > 1 else "mind"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == ds)
    hist = pl.read_parquet(PROCESSED / "history.parquet").filter(pl.col("dataset") == ds)
    arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == ds)
    ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset=ds)
    ctx = assemble.build_context(ds, imps, arts, ids, emb)
    if ctx.sessions is not None:
        print(f"ctx.sessions rows: {ctx.sessions.height:,}")

    target = imps.filter(pl.col("split") == "test")
    hs = hist.filter(pl.col("split") == "test")
    sample = target.sample(n, seed=3)
    # Frames are sliced OUTSIDE the profiled region: a serving system is handed
    # its request, it does not filter a 186,721-row table to find it.
    reqs = [(target.filter(pl.col("impression_id") == r["impression_id"]),
             hs.filter(pl.col("user_id") == r["user_id"]))
            for r in sample.iter_rows(named=True)]

    def work():
        for one, h in reqs:
            assemble.build_rows(ctx, one, h)

    pr = cProfile.Profile()
    pr.enable()
    work()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
    print("\n".join(s.getvalue().splitlines()[:34]))


if __name__ == "__main__":
    main()
