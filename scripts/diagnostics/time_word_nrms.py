"""D43a's cost measurement: word-level vs sentence-level NRMS training speed.

D39 pre-committed to replacing its estimate with a timing before launching, and
this is that timing. Both encoders are run over the SAME impressions, so the
RATIO is the answer even on a busy machine — the 2026-09-16 error-log entry
records absolute milliseconds here moving 4.7x between dates while ratios held.

Calibration: the sentence-level figure should reproduce D41's independently
recorded ~9 min/epoch at 60,000 impressions. If it does not, distrust the
extrapolation rather than the model.

    python scripts/diagnostics/time_word_nrms.py [n_impressions] [threads]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from newsrec.rank import nrms  # noqa: E402
from newsrec.retrieval import semantic  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
D41_IMPRESSIONS, D41_EPOCHS, D41_ARMS = 60_000, 3, 4


def main() -> None:
    n_imp = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    ds = "mind"

    hist = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == ds) & (pl.col("split") == "train"))
    ft = pl.read_parquet(PROCESSED / "features" / f"{ds}_train.parquet")
    keep = ft["impression_id"].unique(maintain_order=True).sample(n_imp, seed=1)
    ft = ft.filter(pl.col("impression_id").is_in(keep.implode()))
    print(f"timing on {n_imp:,} MIND train impressions, {ft.height:,} rows, "
          f"{torch.get_num_threads()} threads", flush=True)

    z = np.load(PROCESSED / f"words_{ds}.npz", allow_pickle=True)
    s_ids, s_emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset=ds)

    rates, per_imp = {}, None
    for name, ids, mat, wemb in (("sentence", s_ids, s_emb, None),
                                 ("word-level", list(z["article_ids"]), z["tokens"], z["emb"])):
        b = nrms.prepare(ft, hist, {a: i for i, a in enumerate(ids)})
        n_samples = len(nrms.training_samples(b, np.random.default_rng(nrms.SEED)))
        per_imp = n_samples / n_imp
        torch.manual_seed(nrms.SEED)
        model = nrms.NRMS(mat.shape[1], word_embeddings=wemb)
        t0 = time.perf_counter()
        nrms.train(model, b, mat, epochs=1, log=lambda m: None)
        el = time.perf_counter() - t0
        rates[name] = n_samples / el
        print(f"{name:11s}: {n_samples:,} samples in {el:6.1f} s = {rates[name]:7,.0f} "
              f"samples/s   params {sum(p.numel() for p in model.parameters()):,}", flush=True)

    print(f"\nword-level is {rates['sentence'] / rates['word-level']:.1f}x slower per sample")
    for name, r in rates.items():
        ep = D41_IMPRESSIONS * per_imp / r
        print(f"{name:11s}: ~{ep / 60:5.1f} min/epoch at D41's 60k impressions, "
              f"~{ep * D41_EPOCHS / 3600:4.1f} h per {D41_EPOCHS}-epoch arm, "
              f"~{ep * D41_EPOCHS * D41_ARMS / 3600:5.1f} h for all {D41_ARMS} arms")
    print("NOTE: training only. Epoch selection re-scores the validation subsample after "
          "every epoch, which added ~40% to the real two-arm run (D43a's over-run).")


if __name__ == "__main__":
    main()
