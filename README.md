# NewsRec: Two-Stage News Recommendation on MIND and EB-NeRD

A retrieve-and-rank news recommender evaluated on two public datasets in two languages:
**MIND** (Microsoft News Dataset, English) and **EB-NeRD** (Ekstra Bladet, Danish). Stage 1
retrieves candidates with BM25 and multilingual sentence embeddings. Stage 2 re-ranks them with a
LightGBM LambdaRank model over behavioural features that are built leakage-free, and an NRMS
neural model serves as the comparison baseline. Everything runs on CPU on a single laptop
(WSL2, no GPU).

Detailed design decisions, rejected alternatives and every measurement with its provenance live
in [ARCHITECTURE.md](ARCHITECTURE.md) and [reports/NUMBERS.md](reports/NUMBERS.md). This README
covers the final system, the latest verified results, how to reproduce them, and the known
limitations.

---

## Results

### Ranking quality (AUC)

Leaderboard numbers are from the public Codabench competitions. Offline numbers are on the
local temporal test split (MIND 21,947 impressions, EB-NeRD 186,721), with 95% confidence
intervals from a 1,000-resample paired bootstrap.

| System | MIND leaderboard | EB-NeRD leaderboard | MIND offline test | EB-NeRD offline test |
|---|---|---|---|---|
| Semantic retrieval baseline | 0.6191 (2026-08-26) | 0.5396 (2026-08-27) | 0.6097 | 0.5457 |
| LambdaRank re-ranker | 0.6181 (2026-09-20) | 0.7397 (2026-09-20) | 0.6144 | 0.7428 |
| **LambdaRank + rack normalisation (final)** | **0.6319**, rank 63 (2026-09-26) | **0.7621**, rank 74 (2026-09-26) | **0.6367** | **0.7629** |
| NRMS + freshness + exposure (neural baseline) | — | — | 0.6144 | 0.6738 |

How to read this:
- **EB-NeRD:** the re-ranker adds about +0.20 AUC over retrieval, and the offline harness
  predicted the leaderboard level to within 0.0008.
- **MIND:** the original re-ranker was **indistinguishable from the baseline on the leaderboard**
  (−0.0010). Rack normalisation is what produced a real gain (+0.0138 on the leaderboard,
  +0.0223 [0.0195, 0.0250] offline).
- NRMS was trained on a 60k-impression subsample for 3 epochs on CPU and was not submitted. It is
  a comparison point, not the shipped model.

### Serving latency: three different measurements

The repository reports latency at three scopes. They are **not interchangeable**, and only
the last one is the full serving path. Every figure is from one WSL2 laptop, where the same
code has measured up to ~2.4× slower on a different day (NUMBERS.md §C), so absolute
milliseconds are dated and only ratios between paths timed in the same run carry over.

**1. Retrieval only** (single query, top-200, no features or scoring;
`scripts/audit_numbers.py`, 2026-09-16)

| Dataset | BM25 p50 / p99 | Semantic p50 / p99 |
|---|---|---|
| MIND (65,238 docs) | 15.0 ms / 24.0 ms | 5.6 ms / 15.5 ms |
| EB-NeRD (11,777 docs) | 1.6 ms / 12.3 ms | 4.2 ms / 14.7 ms |

An earlier run on 2026-08-26 measured MIND semantic retrieval at 1.73 ms p50. That number is
superseded and should not be quoted as current.

**2. End-to-end, batch feature path** (retrieve K=100 → build features → score, one request at a
time; `scripts/benchmark_serving.py`, shipped base model, 2026-09-19)

| Dataset | p50 | p99 | p99 < 100 ms target |
|---|---|---|---|
| MIND | 237 ms | 360 ms | **not met** |
| EB-NeRD | 452 ms | 2,369 ms | **not met** |

Feature construction takes over 90% of request time. The batch path reuses the offline Polars
code, which on MIND runs 66 lazy-frame collections per request (NUMBERS.md §K.5) and on
EB-NeRD runs 15 joins against tables of up to 1.2M rows (§K.7).

**3. End-to-end, online feature path** (same requests, features built from precomputed arrays
and nightly user profiles; `benchmark_serving.py --model-tag _rack --online`, rack model,
September 2026)

| Dataset | Warm cache p50 / p99 | Cold cache p99 (fresh VM) |
|---|---|---|
| MIND | 13.7 ms / 30.5 ms | not measured |
| EB-NeRD | 15.9 ms / 33.7–48.9 ms (2 runs) | **125.5 ms** |

The online path **meets the 100 ms p99 target only with a warm page cache.** From a cold start,
stage-1 retrieval alone reaches 121 ms at p99 on EB-NeRD while the 193 MB embedding matrix is
paged in from disk. Production serving would need to pre-fault the indexes before taking
traffic, and this repo does not implement that. The online path also serves user profiles
that can be up to a day stale.

Raw outputs: [reports/serving_benchmark_*.json](reports/) (`meets_sla` refers to the batch
path; `online_meets_sla` refers to the online path in the run recorded there).

### Other measured numbers

- **Feature store rebuild:** 1.45M impressions in 40 s, 3.3 GB peak RSS (`build_pipeline.py`).
- **Leakage check:** a deliberately planted lookahead leak inflates EB-NeRD AUC by +0.2010.
  That is the size of the error the leakage tests guard against.
- **Serving cost** at $0.04 per vCPU-hour: online path $0.00016–0.00022 per 1,000 queries,
  batch path $0.0027 (MIND) / $0.0077 (EB-NeRD).

---

## Architecture

```
            Raw datasets (MIND TSV, EB-NeRD Parquet)
                              │
                              ▼
               scripts/build_pipeline.py
          ingest → unified schema → temporal split
                              │
                              ▼
          Parquet store (articles · impressions · history)
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
   BM25 sparse index                   Multilingual MiniLM embeddings
   (scipy.sparse CSR)                  (384-d, brute-force cosine)
          └───────────────────┬───────────────────┘
                              ▼
                 Stage 1: candidate retrieval
                              │
                              ▼
               Behavioural features (strictly before T)
     recency-decayed user vectors · freshness · exposure share
     session context · rack-relative percentiles and z-scores
                              │
                              ▼
         Stage 2: LightGBM LambdaRank   (NRMS as neural baseline)
                              │
                              ▼
     Evaluation: AUC, MRR, nDCG@5/10, slices, diversity/novelty/coverage,
                 paired bootstrap 95% CIs
```

Serving ([src/newsrec/serving.py](src/newsrec/serving.py)) has two feature paths over the same
retrieval stage. The batch path reuses the offline Polars code. The online path uses
precomputed arrays and nightly user profiles.

---

## Key findings

1. **Content retrievers ignore time, and news readers don't.** 92.7% (MIND) and 93.5% (EB-NeRD)
   of clicks go to fresh articles. BM25's MIND top-200 is only 16.6% fresh, because neither
   BM25 nor cosine similarity has a time term. Multi-scale recency decay and freshness features
   address this, and freshness is the top feature by gain on EB-NeRD (38%).
2. **Retrieval and re-ranking are different problems.** BM25 has a 1.21× retrieval lift on
   EB-NeRD but scores AUC 0.4966 when re-ranking the platform's candidate list. Those lists are
   already topically filtered, so lexical overlap stops being discriminative.
3. **Temporal splits, enforced by tests.** [tests/test_no_leakage.py](tests/test_no_leakage.py)
   asserts that no feature for time T reads interactions at or after T, and that candidate pools
   never see labels. Four planted forward leaks were confirmed to fail the tests.
4. **Rack-relative normalisation.** Up to 74% of a feature's variance sat *between* candidate
   lists rather than within them. Re-expressing features as within-list percentiles and z-scores
   raised AUC by +0.0223 (MIND) and +0.0202 (EB-NeRD) offline, and removed both of MIND's
   slice regressions (cold users and head-exposure articles).
5. **Memory under a laptop budget.** A full 37,777 × 65,238 MIND user-article score matrix would
   need 9.9 GB on a 7.7 GB machine. The baseline leaderboard run (13.5M EB-NeRD impressions)
   fit by streaming impressions in slices, memory-mapping user vectors, and restricting
   the scoring matrix to the 8.3% of articles that can ever be candidates. That last change cut
   the dominant memory term 12× without changing any result. The later LambdaRank submission
   peaked at 10.6 GB RSS and relied on swap.

---

## Setup and reproduction

### Environment

Tested on Python 3.12.3 (WSL2, aarch64, no GPU). `requirements.txt` pins every version and
uses the CPU-only PyTorch index (~390 MB instead of ~2.9 GB with CUDA wheels).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: rank-bm25, used only by a benchmark
```

### Data

Datasets and trained models are **not included** in this repository. Download them separately
and place them as `configs/mind.yaml` and `configs/ebnerd.yaml` expect:

- MIND: `data/raw/mind/` (and `MINDlarge_test` for leaderboard submissions)
- EB-NeRD: `data/raw/ebnerd/ebnerd_large/` (and `ebnerd_testset` for submissions)

`scripts/build_pipeline.py` checks that the files are present. If any are missing, it prints
download instructions and exits.

### Tests

```bash
python -m pytest -q
```

Verified on 2026-10-10 in a fresh virtualenv built only from `requirements.txt` and
`requirements-dev.txt`:

| Environment | Result | Time |
|---|---|---|
| No datasets (fresh clone) | 340 passed, 21 skipped | 2.5 min |
| Feature store built (`data/processed/`) | **361 passed** | 6 min |

The 21 skipped tests are the ones that need the real feature store, including the
real-data leakage checks in `test_no_leakage.py`. They skip with "feature store not built"
until `build_pipeline.py` has been run.

### Pipeline

Run in order. Times are from the development laptop (September 2026) and vary with machine
state.

| Step | Command | Produces | Time |
|---|---|---|---|
| 1 | `scripts/build_pipeline.py` | Parquet store, temporal splits | 40 s |
| 2 | `scripts/build_embeddings.py` | article embeddings (one-off) | ~13 min |
| 3 | `scripts/run_bm25_recall.py`, `scripts/run_semantic_recall.py` | retrieval recall@K | ~5 min |
| 4 | `scripts/build_feature_tables.py` | feature tables, 22.5M rows | 3 min MIND, 10 min EB-NeRD |
| 5 | `scripts/run_reranker.py --features rack` | final LambdaRank model + test metrics | ~2 min MIND, ~8 min EB-NeRD |
| 6 | `scripts/compare_rack.py` | rack vs base, paired CIs | ~1 min per dataset |
| 7 | `scripts/run_extended_eval.py --extra-models _rack --tag _rack` | sliced metrics | ~2 min MIND, ~30 min EB-NeRD |
| 8 | `scripts/run_nrms.py --datasets mind ebnerd` | NRMS baseline and ablation | hours on CPU |
| 9 | `scripts/benchmark_serving.py --dataset mind --model-tag _rack --online` | latency, footprint, cost | ~5 min per dataset |

Run step 9 on an otherwise idle machine, and more than once. A single p99 is not a reliable
measurement on this hardware (NUMBERS.md §K.9).

---

## Limitations

- **Latency target only met under warm conditions.** See above. The batch serving path misses
  the 100 ms p99 target on both datasets, and the online path misses it from a cold start on
  EB-NeRD.
- **Single-machine measurements.** All timings come from one laptop under WSL2. Absolute numbers
  have moved ~2.4× between days on identical code.
- **Small MIND gains.** On MIND, the leaderboard delivered only 62% of the offline rack
  improvement, and the base re-ranker did not beat retrieval. The local MIND test split covers
  12 hours, while the leaderboard covers a week.
- **EB-NeRD is subsampled locally.** Training and offline evaluation use 5% of `ebnerd_large`
  users (48,666, sampled by user ID). Leaderboard scoring covers the full test set.
- **NRMS is under-trained.** Sentence-level encoder, 60k impressions, 3 epochs, CPU only. A
  word-level variant overfit validation (+0.02 on validation, no gain on test).
- **Stale profiles.** The online path's nightly user profiles trade freshness for latency.

---

## Repository layout

```
configs/            dataset paths (mind.yaml, ebnerd.yaml)
src/newsrec/
  ingest_mind.py, ingest_ebnerd.py   dataset readers → unified schema
  temporal_split.py                  time-based train/val/test split
  retrieval/                         BM25 index, semantic search, availability masks
  features/                          history, article, session, rack features; leakage quarantine
  rank/                              LambdaRank (gbdt.py), NRMS (nrms.py)
  eval/                              metrics, slices, bootstrap CIs, beyond-accuracy
  serving.py                         batch and online serving paths
  submission.py, predict.py          leaderboard submission path
scripts/            command-line entry points (no logic); diagnostics/ for ledger measurements
tests/              unit, property and leakage tests
reports/            NUMBERS.md (measurement ledger), design_note, CSV/JSON results, figures
ARCHITECTURE.md     design decisions with rejected alternatives
SCALE_NOTES.md      scaling analysis
GLOSSARY.md         terminology
```
