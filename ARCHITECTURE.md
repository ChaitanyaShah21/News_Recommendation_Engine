# Architecture & Decision Log

Last updated: 2026-08-21

---

## What we are building

A news-recommendation retrieval pipeline over two datasets, producing ranked candidate
articles for each impression, evaluated offline and submitted to two public leaderboards.

```
                      ┌──────────────────────────────────────────┐
   raw files          │  scripts/build_pipeline.py               │
   (TSV / Parquet) ──▶│  download → ingest → split → feature store│──▶ data/processed/
                      └──────────────────────────────────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
              ┌──────────────────┐            ┌──────────────────┐
              │ lexical (BM25)   │            │ semantic (embed) │
              │ inverted index   │            │ ANN index        │
              └────────┬─────────┘            └────────┬─────────┘
                       └──────────────┬────────────────┘
                                      ▼
                         ┌────────────────────────┐
                         │ evaluation harness     │
                         │ AUC MRR nDCG + beyond- │
                         │ accuracy + slices + CI │
                         └───────────┬────────────┘
                                     ▼
                         ┌────────────────────────┐
                         │ submission generator   │
                         │ → Codabench            │
                         └────────────────────────┘
```

---

## Layout

| Path | Role |
|---|---|
| `src/newsrec/` | The package. All real logic lives here. |
| `src/newsrec/ingest_*.py` | Dataset-specific readers → one unified schema |
| `src/newsrec/retrieval/` | `popularity.py` (baseline), `bm25.py`, `semantic.py` |
| `src/newsrec/eval/` | Metrics, beyond-accuracy, slices, bootstrap, harness |
| `scripts/` | Thin command-line entry points; no logic |
| `configs/` | `mind.yaml`, `ebnerd.yaml` — no hardcoded paths anywhere in code |
| `tests/` | Includes `test_no_leakage.py`, required by temporal leakage testing |
| `notebooks/` | `00_provided_*` are course-supplied reference. Ours only display results. |
| `data/` | Gitignored. Raw and processed data. |

---

## Decision log

Every choice, the alternatives rejected, and why. This section becomes the design note.

### D1 — Python package with thin notebooks, rather than notebook-first
**Date:** 2026-08-21 · **Decided by:** Chaitanya

**Chosen:** Real modules under `src/newsrec/`, command-line scripts, notebooks that only
import and display.

**Alternatives rejected:**
- *Notebook-first, extract scripts near the deadline.* Faster to iterate, but extraction
  at the end is where reproducibility usually breaks — and Q1 explicitly grades a
  one-command rebuild.
- *Flat scripts, no package.* Simplest to read, but gets repetitive across two datasets
  with different file formats.

**Why:** The assignment requires `python build_pipeline.py` to rebuild everything from raw
files. Small named functions are also what make the code explainable step by step.

---

### D2 — Develop locally on small bundles, use cloud only for large test sets
**Date:** 2026-08-21 · **Decided by:** Chaitanya

**Chosen:** All development and debugging on MIND-small and EB-NeRD-demo, which fit in
the local 7 GB of RAM. Push to GitHub, pull into a cloud notebook for the large runs.

**Alternatives rejected:**
- *Everything in cloud notebooks.* Simpler on resources, but hostile to a clean git
  history and a one-command pipeline.
- *Everything local.* EB-NeRD-large has 13.5 M impressions and there is no local GPU
  for computing text embeddings.

**Why:** Matches the assignment's own demo → small → large advice, and keeps a single
codebase that runs in both places.

**Open sub-decision:** which cloud platform. Deferred to Phase 5, to be decided against
measured memory numbers rather than guesses.

---

_Further decisions appended as they are made._
