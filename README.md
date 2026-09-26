# NewsRec: Scalable Multilingual News Recommendation Engine

A production-grade, two-stage **retrieve-and-rank** news recommendation pipeline evaluated over **15M+ interactions** across two languages: **MIND** (Microsoft News Dataset, English) and **EB-NeRD** (Ekstra Bladet News Recommendation Dataset, Danish).

The system integrates sparse lexical retrieval, dense multilingual sentence embeddings, behavioural feature engineering with guaranteed zero temporal leakage, trained LambdaRank GBDTs, and attention-based neural recommendation models (NRMS). Designed for high throughput and memory efficiency, the entire pipeline executes on commodity CPU hardware under **8 GB RAM** without requiring GPU infrastructure.

---

## Performance Highlights

| Benchmark / Dataset | Baseline Retriever | Two-Stage LambdaRank | Neural NRMS (+ Freshness) | Test Set Status |
|---|---|---|---|---|
| **EB-NeRD (Danish)** | 0.5396 AUC | **0.7397 AUC** (+0.2001 lift) | **0.7182 AUC** (+0.0864 lift) | Verified (RecSys 2024 Challenge) |
| **MIND (English)** | 0.6037 AUC | **0.6191 AUC** (+0.0154 lift) | **0.6181 AUC** | Verified (MIND Benchmark) |
| **With Rack Normalisation** | — | **0.7621 AUC** (EB-NeRD) | **0.6319 AUC** (MIND) | Local test split & leaderboard |

### Serving & Systems Benchmarks (Measured on 7.7 GB RAM, CPU)
- **Semantic Retrieval Latency:** **1.73 ms** (p50) / **6.50 ms** (p99) per query over 65,000+ candidates.
- **Serving Cost:** **$0.0027** per 1,000 queries (MIND) / **$0.0077** per 1,000 queries (EB-NeRD).
- **Throughput:** Ingestion and feature store generation over **1.45M impressions** in **40 seconds** (3.3 GB peak memory).
- **Test Suite:** **354 tests** passing, including property-based tests enforcing zero future temporal leakage.

---

## System Architecture

```
                                    Raw Source Datasets
                         (MIND TSV / EB-NeRD Parquet, Multi-Lingual)
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    scripts/build_pipeline.py  │
                             │  Ingestion & Schema Norm      │
                             │  Strict Temporal Splitting    │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                                  Parquet Feature Store
                             (articles · impressions · history)
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
           ┌──────────────────────┐                      ┌──────────────────────┐
           │   Lexical Stage      │                      │   Semantic Stage     │
           │  BM25 Inverted Index │                      │ Dense Multilingual   │
           │  (scipy.sparse CSR)  │                      │ Sentence Embeddings  │
           └──────────┬───────────┘                      └──────────┬───────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                            ┌─────────────────────────────────┐
                            │    Behavioural Feature Store    │
                            │  • Multi-scale recency decay    │
                            │  • Session interaction context  │
                            │  • Freshness & exposure share   │
                            │  • Rack-relative normalisation  │
                            └────────────────┬────────────────┘
                                             │
                                             ▼
                            ┌─────────────────────────────────┐
                            │       Stage 2: Re-Ranking       │
                            │  • LightGBM LambdaRank (GBDT)   │
                            │  • NRMS Neural Multi-Head Model │
                            └────────────────┬────────────────┘
                                             │
                                             ▼
                            ┌─────────────────────────────────┐
                            │     Evaluation & Benchmarks     │
                            │  • AUC, MRR, nDCG@5, nDCG@10    │
                            │  • Diversity, Novelty, Coverage │
                            │  • Paired Bootstrap 95% CIs     │
                            └─────────────────────────────────┘
```

---

## Key Engineering & ML Innovations

### 1. Solving the Structural Time-Blindness of Content Matching
Content-based retrieval algorithms (BM25 and vector cosine similarity) have no time term in their scoring equations. Measurements on both corpora revealed that while **92.7% (MIND)** and **93.5% (EB-NeRD)** of real user clicks occur on newly published articles, standard content retrievers surface older topical matches (only 16.6% fresh for BM25). 
- **Solution:** We engineered multi-scale exponential recency decay features ($\tau \in \{1\text{h}, 6\text{h}, 24\text{h}, \infty\}$) and article freshness deltas that explicitly capture news decay velocity.

### 2. Retrieval vs. Re-Ranking Divergence
BM25 demonstrated a **1.21× retrieval lift** when selecting candidates from the corpus, but produced an AUC of **0.4966** (worse than random) when re-ranking platform-supplied candidate lists. Because the platform's initial candidate selection already filters for relevant topics, lexical overlap alone ceases to be a discriminative signal. Re-ranking requires behavioural features (user dwell patterns, session length, and freshness decay).

### 3. Guaranteed Zero-Lookahead Temporal Splitting
In recommendation systems, random cross-validation leaks future user engagement into past predictions. We enforce a strict temporal cut across all interactions.
- Our test harness (`tests/test_no_leakage.py`) asserts two invariant properties:
  1. **Temporal Safety:** No prediction for time $T$ accesses interaction logs at or after $T$.
  2. **Label Independence:** Candidate pools and popularity counters are computed strictly without label information (`clicked_article_ids`).
- Verified against mutation testing: deliberate lookahead leaks yielded an artificial +0.2010 AUC gain, confirming that temporal leakage invalidates evaluation integrity.

### 4. Rack-Relative Feature Normalisation
Candidate lists vary in composition across sessions. Computing z-scores and relative position metrics within the presentation rack (candidate slate) removed session-level baseline shifts and improved tail-slice AUC by **+0.0223** on MIND and **+0.0202** on EB-NeRD.

### 5. Memory-Safe Streaming under Resource Constraints
Scoring 13.5M impressions against dense vector representations would conventionally generate a 9.9 GB matrix, exceeding system RAM.
- Utilised Polars streaming scans and memory-mapped NumPy arrays (`np.memmap`) for user profile histories.
- Filtered candidate spaces to valid impression sets, reducing memory consumption by **12×** with zero accuracy loss.

---

## Quickstart

### Prerequisites & Installation
Requires Python 3.10+ (tested on Python 3.12).

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU-optimised PyTorch; zero CUDA bloat)
pip install -r requirements.txt
```

### 1. Build the Feature Store
Reads raw dataset configurations from `configs/mind.yaml` and `configs/ebnerd.yaml` and builds the feature store in `data/processed/`:

```bash
python scripts/build_pipeline.py
```

### 2. Generate Dense Embeddings
Extracts sentence embeddings for all articles using `paraphrase-multilingual-MiniLM-L12-v2`:

```bash
python scripts/build_embeddings.py
```

### 3. Train Re-Ranker and Evaluate
Trains the LambdaRank model on behavioural feature tables and evaluates against retrieval baselines with paired bootstrap confidence intervals:

```bash
# Build cached feature tables
python scripts/build_feature_tables.py

# Train LambdaRank re-ranker
python scripts/run_reranker.py

# Run comprehensive sliced evaluation
python scripts/run_extended_eval.py
```

---

## Pipeline Reproduction Steps

Every timing below was measured on a local machine (WSL2, 7.7 GB RAM, **no GPU**):

| Step | Script | Description | Runtime |
|---|---|---|---|
| **1** | `scripts/build_pipeline.py` | Ingests data and builds unified feature store | **2.8 s** |
| **2** | `scripts/build_embeddings.py` | Builds multilingual article embeddings | **13.3 min** |
| **3** | `scripts/run_bm25_recall.py` | Evaluates BM25 candidate recall@K | **235 s** |
| **4** | `scripts/run_semantic_recall.py` | Evaluates dense semantic recall@K | **43 s** |
| **5** | `scripts/build_feature_tables.py` | Builds behavioural feature tables (22.5M rows) | **3.2 min** (MIND) / **10.0 min** (EB-NeRD) |
| **6** | `scripts/run_reranker.py` | Trains LambdaRank re-ranker and computes paired CIs | **~3 min** per dataset |
| **7** | `scripts/run_nrms.py` | Trains NRMS neural model with freshness term | CPU-optimised |
| **8** | `scripts/benchmark_serving.py` | Profiles p50/p99 latency, RAM footprint, and SLA | **~5 min** |
| **9** | `scripts/run_beyond_accuracy.py` | Evaluates intra-list diversity, novelty, and coverage | **283 s** |

---

## Repository Structure

```
├── configs/               # Dataset paths and sampling configurations
│   ├── ebnerd.yaml
│   └── mind.yaml
├── reports/               # Benchmark tables, figures, and architectural ledgers
│   ├── NUMBERS.md         # Measured systems ledger with exact provenance
│   ├── design_note.md     # Technical design note and algorithmic analysis
│   └── figures/           # Leaderboard and benchmark evaluation curves
├── scripts/               # CLI entry points for pipeline execution
│   ├── benchmark_serving.py     # Latency and memory profiling
│   ├── build_embeddings.py      # Article dense vector generation
│   ├── build_feature_tables.py  # Behavioural feature assembly
│   ├── build_pipeline.py        # Ingestion and temporal feature store
│   ├── run_extended_eval.py     # Sliced evaluation across cohorts
│   ├── run_nrms.py              # Neural recommendation baseline
│   ├── run_reranker.py          # LambdaRank model training and test
│   └── validate_submission.py   # Pre-flight format and rank validator
├── src/newsrec/           # Core library package
│   ├── eval/              # Ranking metrics, slices, bootstrap CI, diversity
│   ├── features/          # Recency decay, session context, exposure share
│   ├── rank/              # LambdaRank GBDT and NRMS neural model
│   ├── retrieval/         # BM25 sparse index and vector search
│   ├── ingest_ebnerd.py   # EB-NeRD Parquet parser
│   ├── ingest_mind.py     # MIND TSV parser
│   ├── serving.py         # Low-latency online inference path
│   └── temporal_split.py  # Zero-leakage temporal partitioner
└── tests/                 # Comprehensive test suite (354 tests)
    ├── test_no_leakage.py       # Property test asserting zero future leakage
    ├── test_gbdt.py             # LambdaRank training convergence
    ├── test_nrms.py             # Neural architecture invariants
    └── test_bm25_index.py       # Sparse index consistency
```

---

## Testing & Quality Assurance

Run the test suite:

```bash
# Run all tests
python -m pytest

# Run temporal leakage verification
python -m pytest tests/test_no_leakage.py
```

All 354 tests pass deterministically. The test suite includes unit tests for mathematical accuracy, shape validation for tensor operations, and property tests guaranteeing temporal isolation.