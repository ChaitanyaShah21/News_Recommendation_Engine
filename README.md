# News Recommendation: Lexical & Semantic Retrieval at Scale

**Chaitanya Shah · Systems Design** · Chaitanya Shah

A news-recommendation pipeline over two datasets in two languages — **MIND**
(Microsoft News Dataset, English) and **EB-NeRD** (Ekstra Bladet News Recommendation
Dataset, Danish) — covering ingestion into one unified schema, BM25 (Best Match 25)
lexical retrieval, embedding-based semantic retrieval, a full offline evaluation harness
with bootstrap confidence intervals, an anti-gaming leakage ablation, and submissions to
both Codabench leaderboards.

**240 tests pass.** Every module was mutation-tested — deliberate bugs reintroduced one
at a time to confirm the tests actually catch them.

---

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # ~390 MB; CPU-only PyTorch, 0 CUDA packages
```

> `requirements.txt` carries `--index-url https://download.pytorch.org/whl/cpu` **above**
> the `torch` pin. Delete those two lines and a rebuild silently pulls **2,894 MB**
> instead, 2,238 MB of it NVIDIA packages that never load on a machine without a GPU.

### One-command rebuild (Q1.5)

```bash
.venv/bin/python scripts/build_pipeline.py
```

Reads raw-data locations from `configs/mind.yaml` and `configs/ebnerd.yaml`, verifies the
files are present, and rebuilds the whole feature store into `data/processed/` in **2.8 s**
(77,015 articles · 280,197 impressions · 154,714 user histories).

It does **not** download anything (D10). If raw data is missing it prints the exact
download commands and exits 1 — both the success and failure paths are tested.

---

## Reproducing every result

Run in this order. Timings are measured on the development machine (WSL2, 7.7 GB RAM,
**no GPU**).

| # | Command | Produces | Time |
|---|---|---|---|
| 1 | `scripts/build_pipeline.py` | the feature store | **2.8 s** |
| 2 | `scripts/build_embeddings.py` | `embeddings.parquet` (77,015 × 384) | **13.3 min** |
| 3 | `scripts/run_bm25_recall.py` | Q2 recall@K, both pools | not separately timed |
| 4 | `scripts/run_semantic_recall.py` | Q3 recall@K, both pools | not separately timed |
| 5 | `scripts/summarise_recall.py` | `reports/recall_summary.csv` | seconds |
| 6 | `scripts/run_rerank_eval.py` | Q4.2 AUC/MRR/nDCG, 4 scorers | **136 s** MIND, **1.2 s** EB-NeRD |
| 7 | `scripts/run_beyond_accuracy.py` | Q4.3 diversity / novelty / coverage | not separately timed |
| 8 | `scripts/run_eval_report.py` | Q4.3 slices + Q4.4 bootstrap CIs | not separately timed |
| 9 | `scripts/run_ablation.py` | Q9 serving-time ablation | not separately timed |

Bold timings are measured on this machine. The rest were never stopwatch-timed as
individual steps, and are left unclaimed rather than guessed — all of them are minutes,
not hours.

Step 2 is deliberately **not** part of `build_pipeline.py`: that script rewrites
`articles.parquet` on every run, so vectors stored there would be recomputed each rebuild
(2.8 s → ~13 min, a 280× regression) or destroyed (D22).

### Tests

```bash
.venv/bin/python -m pytest          # 240 tests
.venv/bin/python -m pytest tests/test_no_leakage.py   # the Q9 deliverable
```

---

## Codabench submissions (Q5)

Needs the large bundles — `MINDlarge_test` and `ebnerd_testset` — with `test_root` in each
config pointing at them.

```bash
.venv/bin/python scripts/build_submission_store.py                 # ~2 s
.venv/bin/python scripts/build_embeddings.py \
    --datasets mind   --articles data/processed/submission/articles_mind.parquet \
    --output data/processed/submission/embeddings_mind.parquet     # 24.4 min (measured)
.venv/bin/python scripts/build_embeddings.py \
    --datasets ebnerd --articles data/processed/submission/articles_ebnerd.parquet \
    --output data/processed/submission/embeddings_ebnerd.parquet   # ~25 min (projected)

.venv/bin/python scripts/run_submission.py --dataset mind          # 6.4 min (measured)
.venv/bin/python scripts/validate_submission.py --dataset mind

.venv/bin/python scripts/run_submission.py --dataset ebnerd        # ~35 min (projected)
.venv/bin/python scripts/validate_submission.py --dataset ebnerd
```

Upload `reports/submissions/{dataset}_semantic.zip`:

- MIND → https://www.codabench.org/competitions/13967/
- EB-NeRD → https://www.codabench.org/competitions/2469/

**Always run `validate_submission.py` before uploading.** A submission cannot be debugged
from a leaderboard — it returns one number and no diagnostics, so a wrong rank direction,
a dropped row and a shuffled order are indistinguishable from "the model is weak". The
validator re-reads the raw bundle and checks that line *i* carries impression *i*'s own id
and that every rank list is a permutation of 1..n against that impression's real candidate
count.

---

## Layout

| Path | Role |
|---|---|
| `src/newsrec/ingest_*.py` | dataset-specific readers → one unified schema (D3) |
| `src/newsrec/temporal_split.py` | train/val/test by time, never randomly |
| `src/newsrec/retrieval/` | `bm25.py`, `semantic.py`, their search modules, `availability.py` |
| `src/newsrec/eval/` | metrics, beyond-accuracy, slices, bootstrap, re-ranking, ablation |
| `src/newsrec/submission.py`, `predict.py` | Q5 leaderboard path |
| `scripts/` | thin command-line entry points, no logic |
| `configs/` | all paths; nothing hardcoded in code |
| `tests/` | 240 tests, including `test_no_leakage.py` (Q9) |
| `data/`, `reports/submissions/` | gitignored |

### Documents

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | system design + the full decision log (D1–D30), every alternative rejected and why |
| `GLOSSARY.md` | every term, plain-language then technical |
| `SCALE_NOTES.md` | measured "where this breaks at 10×" observations |

---

## Headline results

All on the validation split, macro-averaged per impression, impressions with a usable
query (D17/D18).

**Retrieval — recall@200 against a random baseline** (the baseline is what makes it
readable; without it EB-NeRD's numbers are actively misleading):

| Dataset | Pool | BM25 | Semantic | BM25 lift | Semantic lift |
|---|---|---|---|---|---|
| MIND | whole corpus | 2.05% | **2.17%** | 6.69× | **7.08×** |
| MIND | in circulation | 3.95% | **5.41%** | 4.19× | **5.74×** |
| EB-NeRD | whole corpus | 2.45% | **2.65%** | 1.44× | **1.56×** |
| EB-NeRD | in circulation | 7.27% | **8.57%** | 1.07× | **1.26×** |

**Re-ranking the platform's own candidate list** (what both leaderboards score):

| Dataset | random | popularity | BM25 | semantic |
|---|---|---|---|---|
| MIND — AUC | 0.5007 | 0.5423 | 0.5492 | **0.6338** |
| EB-NeRD — AUC | 0.4987 | 0.4647 | 0.4966 | **0.5331** |

**The anti-gaming result (Q9)** — identical algorithm, counting window moved from training
to the evaluated window:

| arm | MIND | EB-NeRD |
|---|---|---|
| popularity (train) · honest | 0.5423 | 0.4647 |
| popularity (**future**) · leaking | **0.6102** | **0.6657** |
| semantic · honest, our best system | 0.6338 | 0.5331 |

On EB-NeRD, moving one counting window is worth **+0.2010 AUC** while semantic beats random
by only 0.0344 — **one leaked feature outvalues roughly 4× all the honest modelling in this
project.** That is the argument for why leaderboard rank is a poor grading signal, in a
single number.

---

## Three findings that shaped the design

1. **Content-based retrieval is blind to time, structurally.** BM25's top-200 reproduces
   the corpus's own freshness profile (16.6% fresh) while **92.7% of actual clicks are
   fresh**. Running a completely different matching function reproduced it exactly
   (semantic: 31.9% fresh vs a 33.5% corpus baseline, against 93.5% of real EB-NeRD
   clicks) — so this is a property of content matching as such, not a BM25 flaw. Neither
   formula contains a time term, so it cannot be tuned away; it has to be handled outside
   the scoring function (D19) or not at all.

2. **Retrieval and re-ranking are different jobs.** BM25 holds a 1.21× retrieval lift on
   EB-NeRD yet **cannot re-rank it at all** (AUC 0.4966, CI [0.4917, 0.5009] — contains
   0.5). The platform's own recommender has already spent the easy signal, so everything in
   the candidate list is already plausible and lexical overlap no longer separates it.

3. **A metric measured in the space a method optimises grades it tautologically.** Semantic
   ÷ BM25 intra-list diversity is 0.544 in embedding space but **1.041 by category** on
   EB-NeRD — reported conventionally we would have published "semantic produces markedly
   less diverse lists", which is false in *direction*. The same shape appeared again in
   popularity scoring AUC 0.9737 on a slice defined by popularity. Two instances made it a
   rule rather than an anecdote.

---

## Environment

WSL2 · 7.7 GB RAM · **no GPU** · Python 3.12 · Polars 1.43 · CPU-only PyTorch 2.13.
Everything above runs on that machine; nothing requires cloud compute (D29).
