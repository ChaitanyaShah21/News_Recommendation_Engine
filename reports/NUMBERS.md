# Numbers ledger

**Every figure we would quote aloud, with where it came from.** Created 2026-09-16, after
Quiz-1 showed that answers about our own system are cross-checked against this repository
during oral evaluation. The rule: *a number that is not in this table with a command
beside it does not get said in a viva, a design note, or a README.*

Columns: **Value** as measured · **Where it came from** (command, or file and line) ·
**When**. "Estimate" is written out as an estimate, with its reasoning, never as a
measurement.

Re-measure with the commands in the last section. If a number changes, change it here in
the same commit that changes the thing.

---

## A. The corpus and what was indexed

Measured against **A1's own feature store** (`/home/csharp/IRE/A1/data/processed`), i.e.
the system that was submitted, not A2's rebuilt store. Command: `audit_a1.py` (section G).

| Quantity | MIND | EB-NeRD (demo) | Where from |
|---|---|---|---|
| Documents indexed | 65,238 | 11,777 | `articles.parquet`, one row per article |
| Raw source bytes | 74.7 MB (two `news.tsv`) | 15.3 MB (`articles.parquet`) | `du -sb` on the raw files |
| Fields indexed | title + abstract | title + abstract | `bm25.build_index`, [bm25.py:172](src/newsrec/retrieval/bm25.py#L172) |
| Vocabulary (distinct terms) | 60,951 | 31,642 | audit, `len(index.vocab)` |
| Postings (non-zeros) | 2,361,877 | 268,891 | audit, `index.doc_term.nnz` |
| Postings per document | 36.2 | 22.8 | derived |
| Mean tokens per document | 47.2 | 25.8 | audit, `index.doc_len.mean()` |
| Total tokens | 3,081,158 | 303,890 | audit, `index.doc_len.sum()` |

**EB-NeRD documents are about half the length of MIND's** (25.8 vs 47.2 tokens) on 5.5×
fewer documents, and that difference drives the latency inversion in section C.

**Tokenisation (D11):** lowercase, Unicode word split, **no stopword removal, no
stemming**. Danish keeps its own letters — verified against the EB-NeRD title
"Rådden kørsel på blå plader" ([bm25.py:39](src/newsrec/retrieval/bm25.py#L39)).

**BM25 parameters (D13):** k₁ = 1.5, b = 0.75.

---

## B. Index size and memory tier

| Index | Size | vs raw corpus | Structure and tier | Evidence |
|---|---|---|---|---|
| BM25 inverted (MIND) | 19.4 MB | 26% | `scipy.sparse.csr_matrix`, float32, **RAM** | `sparse.csr_matrix(...)` [bm25.py:247](src/newsrec/retrieval/bm25.py#L247) |
| Embedding matrix (MIND) | 100.2 MB | 134% | dense `np.ndarray` float32 (65238, 384), **RAM** | `np.ascontiguousarray` [semantic.py:165](src/newsrec/retrieval/semantic.py#L165) |
| User vectors (MIND val, N=100) | 77 MB | — | dense float32, **RAM** | `semantic_search.build_user_vectors` |
| BM25 inverted (EB-NeRD demo) | 2.2 MB | 15% | same | same |
| Embedding matrix (EB-NeRD demo) | 18.1 MB | 118% | dense (11777, 384), **RAM** | same |
| User vectors (EB-NeRD val, N=100) | 2 MB | — | dense float32, **RAM** | same |

**The dense matrix is 5.2× the sparse index** for the same corpus, and the only index
larger than the text it came from. Sparse wins on storage decisively; it does **not** win
on speed (section C).

**Was RAM a choice or the only option?** A choice, and a reversible one. 100 MB fits
trivially in 11 GB, and D22 explicitly considered `np.memmap` so the matrix need never be
resident, rejecting it because there was nothing to buy with it at this scale. At 10× the
corpus the same decision would be re-taken on different grounds (`SCALE_NOTES.md`).

**Embedding model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
**384 dimensions**, 12 layers, 250,037-token vocabulary, CPU-only PyTorch (D20).
One multilingual model covers English (MIND) and Danish (EB-NeRD).

---

## C. Latency, throughput, and Little's law

> **Read the date before quoting any millisecond in this section.** The same code on
> byte-identical data measured **~2.4× slower on 2026-09-16 than on 2026-08-26**. The
> control was A1's own `benchmark_engineering.py`, re-run unmodified: it too reported the
> slow figures, so both sets were honestly measured and the machine changed underneath
> them (frequency/core placement on a big.LITTLE ARM laptop under WSL2). Within a single
> run the numbers are stable — two independent 300-query passes agreed to within 0.5 ms,
> and two separate runs the same day agreed to within ~5% (BM25 p50 14.99–16.03 ms), so
> differences under about a tenth are noise, not signal.
> **Quote the ratios, which held; treat the absolutes as dated.** See PROGRESS.md's
> error log, 2026-09-16.

### Per-query retrieval latency, batch size 1, top-200 (the serving number)

**2026-09-16, `scripts/audit_numbers.py`** (300 queries, two passes, warm BLAS):

| Dataset | Method | p50 | p95 | p99 | mean |
|---|---|---|---|---|---|
| MIND (65,238 docs) | BM25 sparse | 15.51 / 14.99 ms | 21.14 | 24.04 | 15.92 |
| MIND | semantic dense | **5.95 / 5.63 ms** | 11.89 | 15.48 | 6.37 |
| EB-NeRD (11,777 docs) | BM25 sparse | **1.63 / 1.60 ms** | 3.71 | 12.34 | 2.06 |
| EB-NeRD | semantic dense | 4.54 / 4.18 ms | 9.54 | 14.74 | 5.14 |

**2026-08-26, `scripts/benchmark_engineering.py`,** MIND only, same code: BM25 p50
**10.10 ms** (p99 20.21), semantic p50 **1.73 ms** (p99 6.50).

**The finding that survives both dates, and it inverts between datasets.** On MIND the
dense brute-force scan is **2.6× faster** than the sparse index; on EB-NeRD the sparse
index is **2.8× faster** than the dense scan. Sparse means *less arithmetic*, not less
time: a dense (n × 384) float32 product is sequential, SIMD-friendly work handed to BLAS,
while a sparse product does irregular scattered gathers. Which wins depends on corpus
size — EB-NeRD's 11,777 documents make the sparse walk small enough that its irregularity
stops mattering, while the dense side still carries a full 18 MB matrix multiply. **So
"semantic is faster than BM25" is a statement about MIND's scale, not about the methods.**

### Re-ranking a supplied candidate list (the leaderboard-shaped operation)

| Dataset | Throughput | Per impression | Mean candidates |
|---|---|---|---|
| MIND | 3,064 impressions/s | 0.33 ms | 37.0 |
| EB-NeRD | 67,395 impressions/s | 0.015 ms | 12.0 |

Measured end-to-end over the whole val split (`rerank.score_semantic`, batched). The
per-impression *scoring* cost alone is 0.01 ms p50 (`benchmark_engineering.py`), so most
of MIND's 0.33 ms is candidate gathering, not arithmetic.

**Re-ranking is ~50–1,000× cheaper than whole-corpus retrieval.** That is the quantitative
form of the retrieval-vs-re-ranking distinction: it is why the leaderboards' 13.5M
impressions are tractable while retrieving over the whole corpus for each would not be.

### Little's law consistency check, N = 1

Little's law: in-flight = throughput × response time. With **one** request in flight
(N=1), a single worker cannot exceed **X = 1/R**:

| Dataset | Method | R (mean) | Ceiling 1/R | Measured, batch 32 | Ratio |
|---|---|---|---|---|---|
| MIND | BM25 | 15.92 ms | 62.8 q/s | 225.4 q/s | ×3.59 |
| MIND | semantic | 6.37 ms | 157.0 q/s | 347.2 q/s | ×2.21 |
| EB-NeRD | BM25 | 2.06 ms | 485.6 q/s | 1,224.0 q/s | ×2.52 |
| EB-NeRD | semantic | 5.14 ms | 194.7 q/s | 1,436.2 q/s | ×7.38 |

**The measured throughput exceeds 1/R, and that does not break Little's law — it prices
the batching.** Batching stops the system being N=1: a batch of 32 puts 32 requests in
flight, so the law reads X = N/R with N=32, and the ceiling rises accordingly. The ratio
column is therefore the **measured speed-up from batching**, and it is bought with
latency: an individual query inside a batch waits for its 31 neighbours. This is exactly
the offline-versus-serving trade-off — offline evaluation should batch, a live
single-user request must not.

**Throughput versus batch size** (MIND semantic, 2026-09-16): 142 / 360 / 420 queries per
second at batch 1 / 32 / 256 — monotone. On 2026-08-26 the same measurement peaked at
batch 32 (620 / 1,269 / 727) and regressed at 256. The regression did not reproduce; the
correction is recorded in `SCALE_NOTES.md`. **Batching helps by single-digit multiples,
and the location of the optimum is machine-dependent.**

### Index construction throughput

| Dataset | BM25 build | Rate | Embedding inference |
|---|---|---|---|
| MIND | 2.72 s | 23,989 docs/s | ~87–115 articles/s (CPU, one-off) |
| EB-NeRD demo | 0.33 s | 35,998 docs/s | same |

Embedding is ~200× slower per document than indexing, is CPU-bound with no GPU, and
scales with **articles** rather than impressions — which is why A2 reused the existing
125,541 vectors rather than recomputing them (section I).

### The rejected alternative, measured (D14)

`rank_bm25`, the library D14 rejected by argument: p50 **5,704 ms** per query against our
15.23 ms — **375×** — and a projected **79.2 hours** for one MIND val run against our
12.7 minutes. It builds 2.6× faster because it defers all work to query time. The
defensible form of the argument is not "the library is slow" but **"it moves work to the
wrong side of a boundary crossed 50,000 times"**.

---

## D. Retrieval quality (A1, val, has-query slice)

Ground truth: the **click labels in the held-out val split** — `clicked_article_ids`,
carved from the dev/validation week by D7/D8, never used to build queries or candidate
pools (Q9's two conditions, `tests/test_no_leakage.py`).

recall@K, macro-averaged per impression, `reports/recall_summary.csv`:

| Dataset | Pool | Method | recall@50 | recall@100 | recall@200 | Random @200 | Lift |
|---|---|---|---|---|---|---|---|
| MIND | available | semantic | 0.0187 | 0.0325 | **0.0541** | 0.0094 | 5.7× |
| MIND | available | BM25 (raw tf) | 0.0184 | 0.0270 | 0.0395 | 0.0094 | 4.2× |
| MIND | whole corpus | semantic | 0.0066 | 0.0123 | 0.0217 | 0.0031 | 7.1× |
| EB-NeRD | available | semantic | 0.0216 | 0.0435 | **0.0857** | 0.0681 | 1.26× |
| EB-NeRD | available | BM25 (raw tf) | 0.0206 | 0.0400 | 0.0727 | 0.0681 | 1.07× |
| EB-NeRD | whole corpus | semantic | 0.0063 | 0.0128 | 0.0265 | 0.0170 | 1.56× |

n = 49,649 MIND impressions (75,962 clicks), 17,749 EB-NeRD (17,856 clicks).

**Absolute recall is low by construction** and the lift column is the honest reading: on
MIND, retrieval beats chance 4–8×; on EB-NeRD's available pool it barely beats it at all.
This is the measurement A2 exists to act on.

---

## E. Ranking metrics (A1, val, 1,000-resample bootstrap 95% CIs)

`reports/eval_report_{mind,ebnerd}_val.csv`. Formulas in `GLOSSARY.md`; **AUC** via the
rank identity, not the O(P·N) pair loop; **nDCG** with capped IDCG; ties broken
pessimistically (D23).

**MIND, all val impressions (n = 49,649):**

| Method | AUC [95% CI] | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| random | 0.5007 [0.4982, 0.5034] | 0.2489 | 0.2264 | 0.2906 |
| popularity (train) | 0.5423 [0.5400, 0.5446] | 0.2541 | 0.2278 | 0.2892 |
| BM25 | 0.5492 [0.5463, 0.5518] | 0.3006 | 0.2760 | 0.3377 |
| **semantic** | **0.6338 [0.6312, 0.6364]** | 0.3475 | 0.3316 | 0.3912 |

**EB-NeRD, all val impressions (n = 17,749):**

| Method | AUC [95% CI] | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| random | 0.4987 [0.4935, 0.5038] | 0.3126 | 0.3443 | 0.4295 |
| popularity (train) | 0.4647 [0.4629, 0.4666] | 0.1436 | 0.0939 | 0.2382 |
| BM25 | 0.4966 [0.4917, 0.5009] | 0.3128 | 0.3418 | 0.4297 |
| **semantic** | **0.5331 [0.5286, 0.5379]** | 0.3373 | 0.3730 | 0.4532 |

**Two claims these intervals license.** BM25's EB-NeRD interval **contains 0.5**, so it
cannot be said to beat chance there. Popularity's interval lies **entirely below 0.5** —
worse than random, and not noise.

### Slices (D26)

| Slice | MIND semantic AUC | EB-NeRD semantic AUC | n (MIND / EB-NeRD) |
|---|---|---|---|
| cold (history ≤ 5) | 0.6160 | 0.5217 | 7,522 / 55 |
| warm | 0.6370 | 0.5332 | 42,127 / 17,694 |
| head-exposure | 0.6397 | 0.5269 | 32,060 / 9,735 |
| tail-exposure | 0.6222 | 0.5406 | 10,617 / 7,994 |

**The crossover worth naming:** on MIND, popularity beats BM25 on the **cold** slice
(0.5542 vs 0.5296) while losing overall — a single averaged number hides it.
**The tautology worth naming:** popularity scores AUC 0.9737 (MIND) / 0.9407 (EB-NeRD) on
the head-trainpop slice, because that slice is *defined by* the quantity the method ranks
on. Reported as a warning, not a result.

---

## F. Q9 anti-gaming: the price of a feature that is not available at serving time

`reports/ablation_mind-ebnerd_val.csv`. Adding a **future-window** popularity feature:

| Dataset | Safe (semantic) | + FUTURE popularity | Delta AUC |
|---|---|---|---|
| MIND | 0.6338 | 0.6572 | **+0.0234** |
| EB-NeRD | 0.5331 | 0.5872 | **+0.0541** |

Future popularity *alone* scores 0.6657 on EB-NeRD — beating every honest method we have.
That is the anti-gaming argument in one number: the leaderboard cannot tell the
difference, so the allowlist has to.

---

## G. Beyond-accuracy (K=10, D25)

`reports/beyond_accuracy_mind-ebnerd_val_k10.csv`. Catalogue: 65,238 MIND / 11,777 EB-NeRD.

| Dataset | Method | ILD (embedding) | Novelty | Coverage |
|---|---|---|---|---|
| MIND | BM25 retrieval | 0.7169 | 17.72 | 0.5102 |
| MIND | semantic retrieval | 0.5618 | 17.72 | 0.5321 |
| MIND | popularity retrieval | 0.9003 | 6.86 | **0.00015** |
| EB-NeRD | semantic retrieval | 0.4072 | 14.78 | 0.1444 |

Popularity reaches **0.015% catalogue coverage** — it serves the same handful of articles
to everyone. Coverage is reported **without a resample CI** (D27): the bootstrap estimate
is biased for a set-size statistic, so the pivotal bounds are shown separately.

---

## H. Leaderboard (Codabench), the external check

| Dataset | Submission | AUC | Rank | Offline val AUC | Gap |
|---|---|---|---|---|---|
| MIND | first (N=10) | 0.6037 | 62 / 90 | 0.6338 | −0.030 |
| MIND | final (N=100, D31) | **0.6191** | **54 / 90** | 0.6489 | −0.030 |
| EB-NeRD | 904082 (N=100) | **0.5396** | **147 / 247** | 0.5413 | −0.0017 |

**The calibration claim, and it is the defensible one.** The offline harness predicted
MIND's N=10→N=100 *delta* at +0.0151 and the leaderboard delivered **+0.0154**, while the
absolute levels sat ~0.030 apart. On EB-NeRD the *absolute level* transferred to within
0.0017. So: deltas transfer on MIND, levels transfer on EB-NeRD, and the harness was
right about the dataset where it says the signal is weakest.

---

## I. A2 so far

| Quantity | Value | Where from |
|---|---|---|
| EB-NeRD users sampled (D33) | 48,666 of 974,791 (5%) | `user_id % 100 < 5` |
| EB-NeRD impressions in store | 597,348 train / 435,677 val / 186,721 test | `build_pipeline.py`, verified post-build |
| EB-NeRD history rows | 39,260 / 39,420 / 39,420 | same |
| MIND impressions (unchanged) | 156,965 / 51,205 / 21,947 | same |
| Articles in store | 65,238 MIND + 125,541 EB-NeRD | same |
| Store rebuild | 40 s, 3.3 GB peak RSS | `/usr/bin/time` on `build_pipeline.py` |
| Embeddings assembled (not recomputed) | 7 s vs ~18 min | `assemble_embeddings.py` |
| Tests | 304 passing (7 min: the Q9 tests rebuild real feature tables) | `pytest -q` |
| EB-NeRD history clicks at or after their impression | **0** (train, val and test) | `test_no_real_ebnerd_history_click_postdates_its_impression` |
| Age of the newest history click at impression time | median 5.0 days (p10 1.2, p90 7.7) | D35 measurement, one impression per user, 16.5M pairings |
| EB-NeRD candidates shown before their `published_time` | 1,810 of 14,088,920 (0.013%), 11 articles; median 21 min, max 43 h | D34c measurement |
| Exposure windows (D34b) | 1 h and 24 h, strictly before T, as a share | `features/article.py` |
| EB-NeRD session IDs shared by >1 user | 1,032; hence keyed by (user, session) | A2.3 measurement |
| EB-NeRD session length, keyed properly | median 1 impression (p99 8, max 32); longest 29 min | A2.3 measurement |
| Feature count (allowlist) | MIND 10, EB-NeRD 14 | `assemble.FEATURES` |
| Feature-table smoke run, 200 val impressions | MIND 7,658 rows, 4.0% clicked; EB-NeRD 2,127 rows, 9.4% clicked | A2.4 smoke run |
| Feature-table fixed build cost | ~35 s MIND, ~75 s EB-NeRD (BM25 index + 3 decayed-vector builds) | A2.4 smoke run, 2026-09-18 |
| Decay half-lives (D35) | EB-NeRD 1 d / 7 d / ∞; MIND 3 / 10 / ∞ positions | `features/history.py`, `HALF_LIVES` |
| Estimated val AUC CI half-width at this n | ±0.001 **(estimate)** | A1's ±0.005 on 17,749, scaled 1/√n |


### I.2 Phase A3: the trained re-ranker (2026-09-19, local test split, 1,000-resample CIs)

| Quantity | MIND | EB-NeRD | Where from |
|---|---|---|---|
| Feature-table rows (train / val / test) | 5,843,444 / 1,895,867 / 845,131 | 6,623,716 / 5,212,769 / 2,252,435 | `build_feature_tables.py` |
| Feature build time, all three splits | 3 min 11 s, 4.5 GB peak | 10 min 0 s, 7.6 GB peak | `/usr/bin/time` on the same |
| Trees chosen by early stopping | 18 (val nDCG@10 flat from ~20 to 300 trees) | see model file | `run_reranker.py` |
| AUC, A1 semantic ("before") | 0.6097 [0.6057, 0.6133] | 0.5457 [0.5443, 0.5472] | `reports/rerank_*_test.csv` |
| AUC, LambdaRank ("after") | 0.6144 [0.6104, 0.6185] | **0.7428 [0.7415, 0.7441]** | same |
| Paired AUC gain over A1 semantic | +0.0048 [0.0012, 0.0080] | +0.1971 [0.1953, 0.1989] | same, `paired_bootstrap_diff` |
| Paired MRR gain over A1 semantic | −0.0009 (CI spans 0) | +0.1780 [0.1761, …] | same |
| Top-3 features by gain | cos_inf 27%, freshness 20%, exposure_1h 16% | freshness 38%, exposure_1h 21%, exposure_24h 13% | `reports/rerank_importance_*.csv` |
| MIND exposures lost to reused impression ids (bug, fixed) | 11,896 of 8,584,442 (0.14%) | 0 (ids unique) | regression check, 2026-09-19 |
| Leaderboard candidate rows | 93,115,001 | 205,925,868 | `load_submission_behaviors`, streamed |
| Leaderboard context | — | 170,014,788 exposures over 14,556,456 impressions, built in 137 s | `run_submission_lgbm.py` log |
| Leaderboard featurise + score rate | — | 2.7 min per 215k-impression user group; 10.6 GB peak RSS | smoke run, 1 of 64 groups |
| `is_beyond_accuracy` rows | — | 200,000, all impression id 0, one second, one identical 250-article list | D38 measurement |
| `hours_since_last_click`, p10 / p50 / p90 | — | local test 132 / 153 / 182 h vs leaderboard 24 / 98 / 167 h | feature-sample comparison, D38 |

---

## J. How to re-measure

```bash
# Feature store (A2), then its embeddings
.venv/bin/python scripts/build_pipeline.py
.venv/bin/python scripts/assemble_embeddings.py

# Every quantity in sections A-C: corpus, index sizes, tiers, latency, Little's law
.venv/bin/python scripts/audit_numbers.py                        # this repo's store
.venv/bin/python scripts/audit_numbers.py \
    --processed /home/csharp/IRE/A1/data/processed \
    --raw /home/csharp/IRE/A1/data/raw --datasets mind ebnerd    # A1 as submitted

# Engineering: index build, per-query latency, batching, rejected alternative
.venv/bin/python scripts/benchmark_engineering.py --dataset mind

# Retrieval quality
.venv/bin/python scripts/run_bm25_recall.py --datasets mind ebnerd --split val
.venv/bin/python scripts/run_semantic_recall.py --datasets mind ebnerd --split val
.venv/bin/python scripts/summarise_recall.py

# Ranking metrics, slices, bootstrap CIs; Q9 ablation; beyond-accuracy
.venv/bin/python scripts/run_eval_report.py --datasets mind ebnerd --split val
.venv/bin/python scripts/run_ablation.py --datasets mind ebnerd --split val
.venv/bin/python scripts/run_beyond_accuracy.py --datasets mind ebnerd --split val --k 10

# Tests, including the Q9 leakage assertions
.venv/bin/python -m pytest -q
```

**Measurement hygiene, learned the hard way on 2026-09-16:** latency is machine- and
day-dependent. Timings go in this ledger with the date they were taken, and a number that
disagrees with a previous measurement gets explained before either is quoted.
