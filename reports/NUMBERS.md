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


### I.3 Phase A3 continued: Q4 serving, Q5 sliced, Q9 priced (2026-09-19/20, local test split)

**Q4, per single request, K = 100, measured with nothing else running** (`benchmark_serving.py`):

| Quantity | MIND | EB-NeRD |
|---|---|---|
| Resident footprint | 283 MB | 475 MB |
| Stage 1 retrieve, p50 / p99 | 9.0 / 23.7 ms | 9.6 / 22.2 ms |
| Stage 2 features, p50 / p99 | 223.0 / 348.0 ms | 439.0 / 2,352 ms |
| Stage 3 score, p50 / p99 | 3.2 / 9.0 ms | 4.3 / 6.9 ms |
| Total, p50 / p99 | 237 / **360 ms** | 452 / **2,369 ms** |
| Per-user share of stage 2 | 13.5% | 16.8% |
| Throughput per core / Little's law bound | 4.2 q/s / 1/0.237 s = 4.2 | 1.5 q/s |
| Cost per 1,000 queries @ $0.04/vCPU-h | $0.0027 | $0.0077 |
| Cores for 1,000 QPS | 239 | 689 |
| p99 < 100 ms | **not met** | **not met** |
| Batch rate, same code | ~11,500 rows/s | ~22,000 rows/s (leaderboard run) |

**Q5, sliced AUC (paired difference against A1's semantic scorer):**

| Slice | MIND n | MIND diff | EB-NeRD n | EB-NeRD diff |
|---|---|---|---|---|
| all | 21,947 | +0.0048 [0.0012, 0.0080] | 186,721 | +0.1971 [0.1953, 0.1989] |
| cold (≤5 history) | 3,904 | **−0.0113** | 680 | +0.1810 |
| warm | 18,043 | +0.0083 | 186,041 | +0.1971 |
| head-exposure | 13,108 | **−0.0127 [−0.0176, −0.0079]** | 98,719 | +0.2179 |
| tail-exposure | 5,917 | +0.0353 | 87,759 | +0.1741 |
| mixed (in neither head nor tail) | 2,922 | — | 243 | — |

Beyond-accuracy at K=10 (semantic → LambdaRank): category diversity 0.8166 → 0.7352 (MIND),
0.7993 → 0.8029 (EB-NeRD); novelty 17.34 → 17.69 and 18.87 → 19.06; coverage
0.0283 → 0.0318 and 0.0260 → 0.0247. Coverage has no CI by D27.

**Q9, the price of a serving-unavailable feature:**

| | honest (ships) | + unavailable | paired difference |
|---|---|---|---|
| MIND AUC | 0.6144 | 0.5994 | **−0.0150 [−0.0181, −0.0118]** (window truncated; unmeasurable) |
| EB-NeRD AUC | 0.7428 | 0.7590 | **+0.0162 [0.0153, 0.0172]** |

MIND's forward window: 37–138 h of future in training, **5.7–11.4 h at test**, mean of the
feature 0.075 → 0.241. EB-NeRD's leaky gain is carried by `total_pageviews` (17%) and
`future_exposure_share_24h` (16%); `read_time` and `scroll_percentage` are rack-constant
and therefore inert under a ranking objective.


### I.4 Phase A4: Q3, NRMS reproduced then beaten (2026-09-20, local test split)

Four arms per dataset, identical but for the candidate time inputs, each trained on the
same seeded 60,000 impressions for 3 epochs (D41), epoch chosen by val AUC on 30,000
sampled val impressions, evaluated on the **whole** test split. Paired bootstrap, 1,000
resamples.

| Arm | MIND AUC | paired Δ vs baseline | EB-NeRD AUC | paired Δ vs baseline |
|---|---|---|---|---|
| nrms (baseline) | 0.6146 [0.6109, 0.6184] | — | 0.5874 [0.5860, 0.5887] | — |
| + freshness | 0.6159 | +0.0013 [0.0004, 0.0022] | 0.7014 | +0.1141 [0.1126, 0.1155] |
| + exposure | 0.6122 | −0.0024 [−0.0032, −0.0014] | 0.6244 | +0.0370 [0.0359, 0.0380] |
| **+ both (pre-registered)** | 0.6144 | −0.0002 [−0.0011, 0.0008] | **0.6738** | **+0.0864 [0.0852, 0.0877]** |
| MRR, + both | 0.3201 | +0.0053 [0.0038, 0.0067] | 0.4361 | +0.0628 [0.0613, 0.0642] |

**Robustness check, 2026-09-20 (MIND only, 6 epochs, `nrms_ablation_mind_test_6epochs.csv`):**
baseline 0.6180, + freshness 0.6192 (+0.0012 [0.0003, 0.0022]), + exposure 0.6160
(−0.0020 [−0.0030, −0.0010]), + both 0.6180 (−0.0001 [−0.0011, 0.0009], spans zero; MRR
+0.0040 [0.0025, 0.0054]). Validation trace: 0.5967 / 0.6039 / 0.6107 / 0.6152 / 0.6180 /
0.6187 — last gain +0.0007. Every arm gains ~+0.0035 and no conclusion changes.

Validation trace (baseline): MIND 0.5967 → 0.6039 → 0.6107 (still rising at the cap);
EB-NeRD 0.5818 → 0.5848 → 0.5823 (epoch 2 selected). Training cost after D41's subsample:
~9 min per MIND epoch, ~6 min per EB-NeRD epoch, ~30 min per arm including scoring.


### I.5 Codabench, the external check (2026-09-20)

| | A1 submitted system | A2 (LambdaRank) | offline test AUC | offline vs leaderboard |
|---|---|---|---|---|
| MIND (comp. 13967) | 0.6191 | **0.6181**, rank 69 (sub 934561) | 0.6144 | 0.0037 apart |
| EB-NeRD (comp. 2469) | 0.5396 | **0.7397**, rank 77 (sub 934124) | 0.7428 | **0.0031 apart** |

**EB-NeRD: +0.2001 over A1**, and the offline harness predicted the level to within 0.0031.
**MIND: −0.0010 against A1**, i.e. the two systems are indistinguishable there, although
the offline harness predicted a +0.0048 gain. The offline *level* transferred on both
datasets (≤0.004); the offline *delta* transferred only on EB-NeRD. Consistent with the
slice analysis, which already showed MIND's aggregate gain hiding a −0.0127 loss on
head-exposure impressions.

### I.6 Codabench after D42, the rack features (2026-09-26)

Resubmitted with `lgbm_{ds}_rack.txt` (the D42a arm). Screenshots:
`reports/figures/{mind,ebnerd}_rack.png`. The EB-NeRD score came from a **self-hosted
worker** (Chaitanya's Codespace, the organisers' queue credentials, upstream's worker code)
because the competition's own workers are retired - see D32.

| | before (I.5) | **after D42** | leaderboard delta | offline test AUC | offline delta | offline vs leaderboard |
|---|---|---|---|---|---|---|
| MIND (comp. 13967) | 0.6181, rank 69 | **0.6319, rank 63** (sub 945225) | **+0.0138** | 0.6367 | +0.0223 | 0.0048 apart |
| EB-NeRD (comp. 2469) | 0.7397, rank 77 | **0.7621, rank 74** (sub 945302) | **+0.0224** | 0.7629 | +0.0202 | **0.0008 apart** |

Other columns as displayed (assumed MRR, nDCG@5, nDCG@10 in the leaderboards' order):
MIND 0.3075 / 0.3322 / 0.391 against offline 0.3411 / 0.324 / 0.384; EB-NeRD 0.548 / 0.6181 /
0.6487 against offline 0.5453 / 0.6137 / 0.646.

**Reading.** The pattern of I.5 repeats, now with a second data point per dataset.
EB-NeRD transfers both **level** (0.0008) and **delta** (+0.0224 vs +0.0202). MIND gains
clearly but delivers only **62% of the offline delta** (+0.0138 of +0.0223), and its level is
off by 0.0048 - the same direction and size as I.5's miss. The ~0.640 forecast made before
submitting was too high for MIND and right for EB-NeRD (~0.760). A plausible cause, not
tested: MIND's local test split is 21,947 impressions from one day, the leaderboard is 2.37M
over a week, so the local estimate of any MIND delta carries more week-to-week drift than its
bootstrap CI (which only resamples impressions within that day) can show.

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

---

## K. Extension week (2026-09-20 → 26), branch `a2-improvements`

`main` is untouched and still carries every figure in sections A–J. Nothing below
replaces a number above; it is a second, later measurement of a changed system.

### K.1 D42 diagnosis: between-rack variance share (validation feature tables)

Share of each feature's total variance sitting *between* impressions rather than within
them — the part a single global tree threshold cannot resolve. Source:
`diag_rack.py`, on `data/processed/features/{ds}_val.parquet`.

| feature | MIND | EB-NeRD |
|---|---|---|
| `bm25` | **0.738** | **0.522** |
| `exposure_share_1h` | 0.356 | 0.425 |
| `exposure_share_24h` | 0.208 | 0.235 |
| `cat_share_short/medium/inf` | 0.159 / 0.166 / 0.170 | 0.182 / 0.175 / 0.173 |
| `cos_short/medium/inf` | 0.088 / 0.113 / 0.130 | 0.176 / 0.121 / 0.118 |
| `freshness_hours` | 0.039 | 0.116 |
| `hours_since_last_click`, `session_position`, `minutes_since_session_start`, `device_type` | n/a (MIND lacks them) | **1.000** (constant in 100.0% of racks) |

Rack sizes: MIND min 2, p50 22, p90 91, max 295, mean 37.0 (51,205 val impressions);
EB-NeRD min 5, p50 9, p90 23, max 97, mean 12.0 (435,677).

### K.2 D42a arm selection (validation; test chose nothing)

| | trees | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| MIND base | **18** | 0.6545 | 0.3658 | 0.3489 | 0.4108 |
| MIND +pct | 209 | **0.6614** | 0.3684 | 0.3521 | 0.4155 |
| MIND +z | 120 | 0.6579 | 0.3682 | 0.3509 | 0.4150 |
| MIND +both *(selected)* | 42 | 0.6573 | **0.3720** | **0.3545** | **0.4177** |
| EB-NeRD base | 322 | 0.7295 | 0.5089 | 0.5790 | 0.6131 |
| EB-NeRD +pct | 376 | 0.7500 | 0.5314 | 0.6001 | 0.6329 |
| EB-NeRD +z | 443 | 0.7495 | 0.5276 | 0.5971 | 0.6300 |
| EB-NeRD +both *(selected)* | **656** | **0.7566** | **0.5369** | **0.6060** | **0.6387** |

Selection rule, fixed before reading the table: validation nDCG@10 (LambdaRank's own
objective and the early-stopping criterion), applied identically to both datasets.
**+both wins 7 of 8.** The dissent: MIND AUC, where +pct leads by 0.0041 — and AUC is the
leaderboard metric. Recorded, not smoothed (D42a).

**Tree counts are the finding, not a footnote.** MIND's base arm early-stops at 18 trees;
the same algorithm on the same rows, given the same information expressed rack-relatively,
runs to 209. EB-NeRD goes 322 → 656.

### K.3 Feature-table rebuild, verified as additive

All six tables rebuilt with D42's columns. Row counts unchanged (MIND 5,843,444 /
1,895,867 / 845,131; EB-NeRD 6,623,716 / 5,212,769 / 2,252,435), **every base column
byte-identical**, exactly +20 columns each. Source: `verify_rebuild.py` against
`data/processed/features_pre_rack/`. This is the check that caught the reused-MIND-id bug
on 2026-09-19, so it is run rather than assumed.

### K.4 D43a word-level NRMS cost (measured, quiet machine, 8 threads)

| encoder | samples/s | parameters | min/epoch @ 60k impressions | h per 3-epoch arm |
|---|---|---|---|---|
| sentence-level (D39, shipped) | 150 | 413,328 | 9.9 | 0.5 |
| word-level (D43) | 11 | 10,995,692 | 138.7 | 6.9 |

**14.0x slower per sample.** Calibration: the sentence-level figure reproduces D41's
independently recorded ~9 min/epoch, so the extrapolation is trustworthy.
MIND title vocabulary 34,297 tokens, GloVe 6B.300d covers 31,863 (**92.9%**); title
lengths mean 11.2, p50 11, p90 15, p99 21, max 50, so `TITLE_LEN = 20` keeps **98.9%**
of titles whole.

### K.5 D44 serving profile (cProfile, 20 real MIND requests, K = 100)

| where the time goes | share |
|---|---|
| `PyLazyFrame.collect` — **66 collections per request** | 52% |
| joins (65,238-row article tables, to decorate 100 candidates) | 23% |
| per-user work (`build_decayed_user_vectors`, `build_queries`) | 13% |
| `replace_strict` over a 65,238-entry category map | 12% |

Taken under CPU contention, so the *attribution* is the reading; absolute latency is
re-measured on a quiet machine (K.6).

### K.6 D42a rack arm on the test split (paired against the shipped model)

Both models scored the same impressions and the same candidate lists, so the difference is
paired (Q3.4's method). Source: `reports/rack_vs_base_{ds}_test.csv`, via
`scripts/compare_rack.py`. 1,000 resamples. MIND n = 21,947; EB-NeRD n = 186,721.

| | MIND base | MIND +rack | paired Δ | EB-NeRD base | EB-NeRD +rack | paired Δ |
|---|---|---|---|---|---|---|
| AUC | 0.6144 | **0.6367** | **+0.0223 [0.0195, 0.0250]** | 0.7428 | **0.7629** | **+0.0202 [0.0194, 0.0210]** |
| MRR | 0.3191 | 0.3411 | +0.0220 [0.0193, 0.0247] | 0.5219 | 0.5453 | +0.0234 [0.0222, 0.0245] |
| nDCG@5 | 0.3000 | 0.3240 | +0.0240 [0.0215, 0.0265] | 0.5912 | 0.6137 | +0.0225 [0.0215, 0.0235] |
| nDCG@10 | 0.3636 | 0.3840 | +0.0204 [0.0182, 0.0226] | 0.6250 | 0.6460 | +0.0210 [0.0202, 0.0219] |

**Every interval excludes zero on every metric on both datasets.** Trees chosen by early
stopping: MIND 23 (base 18), EB-NeRD 490 (base 322).

**For scale:** A2's entire gain over A1's semantic scorer on MIND was +0.0048 AUC. This
change adds **+0.0223** — roughly five times that — from normalising features the model
already had, not from new information.

**A run-to-run caveat, recorded rather than hidden.** The validation sweep and the shipped
run order the rack columns differently (`rack_feature_names()` interleaves `_pct`/`_z` per
feature; the sweep grouped all `_pct` then all `_z`). With `feature_fraction = 0.9` the
column order changes which features each tree may sample, so MIND's chosen tree count moved
from 42 to 23 between the two runs on identical data. The feature *set* — which is what was
selected — is the same, and the test result above is the one that ships.

### K.7 D44 serving path, measured on a quiet machine (2026-09-20, `_rack` model)

Both paths timed **in the same process over the same 291 requests**, K = 100, single
request at a time. The ratio is the quotable figure; absolute milliseconds carry their
date, per the 2026-09-16 error-log entry.

**MIND (30 features, 291 requests), latency in ms:**

| stage | batch p50 | batch p99 | online p50 | online p99 |
|---|---|---|---|---|
| retrieve (shared) | 9.37 | 21.99 | 9.37 | 21.99 |
| **features** | **252.16** | **495.29** | **3.18** | **5.06** |
| score | 3.91 | 6.93 | 0.91 | 2.30 |
| **total** | **266.30** | **509.67** | **13.65** | **30.53** |

- **Speedup: 19.5x at p50, 16.7x at p99.** The feature stage alone is 79x.
- **`p99 < 100 ms` SLA: batch NOT met (509.67), online MET (30.53).** Q4's example SLA was
  unreachable before and is reachable now.
- Throughput 3.6 → **68.4 q/s per core**; cores for 1,000 QPS **278 → 15**; cost per 1,000
  queries **$0.003078 → $0.000162**.
- Resident footprint unchanged at 283.1 MB (the arrays are views of the same state).

**The nightly profile job, and the batching asymmetry it rests on.** 50,000 MIND user
profiles build in **18.7 s = 0.37 ms/user**, against **~30 ms/user** measured for the same
work done one request at a time (Q4, section I.3). An **81x** gap, because the expensive
parts are matrix products that vectorise. `SCALE_NOTES.md` predicted this asymmetry before
it was measured; this is the measurement. The corpus-tier index flattening costs 0.1 s.

**What it buys and what it costs, stated together:** the per-user tier is work *moved*, not
removed. A nightly job means a user whose history changed today is served yesterday's
profile — a freshness-for-latency trade, on a dataset where 92.7% of clicks go to fresh
articles.

**EB-NeRD (34 features, 300 requests), and a measurement that had to be re-run.**

The first EB-NeRD run reported a batch p99 of 22,397.72 ms, implying a 665x p99 speedup.
That was not reported. It was re-run unchanged as a control on a quiet machine, per the
2026-09-16 error-log entry:

| | run 1 | run 2 (control) |
|---|---|---|
| batch `features` p50 | 454.42 | 520.17 |
| batch `features` p95 | 7,054.38 | 10,134.99 |
| batch `features` p99 | 22,397.72 | 14,467.94 |
| batch `features` max | 27,286.46 | 30,063.12 |
| **online `features` p50 / p99** | **3.69 / 8.27** | **3.62 / 6.70** |
| **online `total` p50 / p99** | **15.83 / 33.67** | **15.89 / 48.93** |

**The tail is real and reproduces; its magnitude does not.** So no p99 speedup ratio is
quoted for EB-NeRD — the denominator moves by 1.5x between runs on identical code, which
would let us report "665x" or "296x" by choosing a run. What is quoted:

- **p50 speedup, stable across both runs: 33.9x** (MIND 19.5x).
- **The online path's own p99: 48.93 ms** (MIND 30.53 ms). **Under the 100 ms SLA in both
  runs, on both datasets**, which is the claim that matters for Q4.3.
- Throughput 0.5 → **57.2 q/s per core**; cores for 1,000 QPS **1,842 → 18**; cost per
  1,000 queries $0.018647 → **$0.000193**. Footprint 476.4 MB. 39,420 EB-NeRD profiles
  build in 50.8 s (**1.29 ms/user**).

**The tail is NOT caused by D42.** Measured in isolation, `add_rack_features` on a
100-candidate rack is p50 11.09 ms, p99 23.43 ms, max 30.87 ms over 200 calls — steady, and
three orders of magnitude below the tail. The mechanism is the batch builder's joins:
cProfile over 30 real EB-NeRD requests puts **64% of the feature path in `DataFrame.join`**
(15 joins per request, 31 ms each on average), several against `ctx.sessions` at
**1,219,746 rows** and the article/first-seen tables at 125,541. Those joins are exactly
what the online path replaces with array indexing, which is why its distribution is tight.

### K.8 D42a sliced AUC (test split) — the diagnosis confirming itself

| MIND slice | n | semantic (A1) | lgbm base | lgbm+rack | base − sem | **rack − sem** | rack − base |
|---|---|---|---|---|---|---|---|
| all | 21,947 | 0.6097 | 0.6144 | 0.6367 | +0.0047 | **+0.0270** | +0.0223 |
| cold | 3,904 | 0.5642 | 0.5529 | 0.5827 | **−0.0113** | **+0.0185** | **+0.0298** |
| warm | 18,043 | 0.6195 | 0.6277 | 0.6484 | +0.0082 | +0.0289 | +0.0207 |
| head-exposure | 13,108 | 0.5955 | 0.5828 | 0.6117 | **−0.0127** | **+0.0162** | **+0.0289** |
| tail-exposure | 5,917 | 0.6319 | 0.6671 | 0.6797 | +0.0352 | +0.0478 | +0.0126 |

| EB-NeRD slice | n | semantic (A1) | lgbm base | lgbm+rack | rack − base |
|---|---|---|---|---|---|
| all | 186,721 | 0.5457 | 0.7428 | 0.7629 | +0.0201 |
| cold | 680 | 0.5538 | 0.7348 | 0.7517 | +0.0169 |
| warm | 186,041 | 0.5457 | 0.7428 | 0.7630 | +0.0202 |
| head-exposure | 98,719 | 0.5565 | 0.7745 | 0.7876 | +0.0131 |
| tail-exposure | 87,759 | 0.5334 | 0.7076 | 0.7357 | +0.0281 |

**Both of MIND's slice regressions are eliminated**, and the size of the repair tracks the
diagnosis: head-exposure **+0.0289** and cold **+0.0298** are the two largest rack-over-base
gains, against **+0.0126** on tail-exposure, where the base model was already winning.
§10 of the design note predicted that absolute exposure "is learned in the inverse direction
and damages head-exposure impressions" **before this fix existed**; the fix repairs
head-exposure hardest. That is a prediction that held, not a result hunted for afterwards.

**Cold start was listed in §10 as a separate problem needing "a separate path".** It did not
need one. The mechanism is visible in the feature semantics: a cold user's `cos_*`,
`cat_share_*` and `bm25` are all null, so the model has only freshness and exposure to work
with — and those are exactly the columns rack normalisation repairs.

**Beyond-accuracy at K = 10 (base → rack):** MIND category diversity 0.7352 → 0.7426,
novelty 17.6906 → 17.7043, coverage 0.0318 → 0.0311. EB-NeRD 0.8029 → 0.8039,
19.0593 → 19.0271, coverage 0.0247 → 0.0256. No meaningful beyond-accuracy cost.

### K.9 EB-NeRD serving, three runs — and the correction they force

Run 3 was taken minutes after a `wsl --shutdown`, i.e. on a **cold page cache** with swap
reset to zero (`pswpin`/`pswpout` both 0). Runs 1 and 2 were on a VM up for ~1 day with
~1.4 GB of swap in use and ~14M pages cumulatively paged out.

| p99, ms | run 1 (warm) | run 2 (warm, control) | **run 3 (cold, fresh VM)** |
|---|---|---|---|
| batch `features` | 22,397.72 | 14,467.94 | **2,742.75** |
| batch `total` | 22,419.01 | 14,481.16 | **2,756.90** |
| `retrieve` (stage 1, shared) | — | 24.38 | **121.49** |
| **`online_features`** | **8.27** | **6.70** | **6.76** |
| `online_total` | 33.67 | 48.93 | **125.47** |

**Two opposite artefacts, both real:**

1. **The extreme batch tail was memory-state amplification.** On a fresh VM the batch p99
   is 2,742.75 ms, which reconciles with the **2,352 ms recorded on 2026-09-19**. The batch
   path still has a genuine tail (p99 is 7.3x its p50 of 375 ms), but not a 30-second one.
2. **A cold page cache makes stage 1 five times worse at p99** (24.38 → 121.49 ms).
   `pgmajfault` = 10,920 since boot: retrieval touches the 125,541 x 384 float32 embedding
   matrix (193 MB) and the availability masks, and on a cold cache those are disk reads.
   Startup context build moved the other way, 26.0 → 8.7 s, from the same cause in reverse.

**What this forces us to stop claiming, and what survives.**

- **Not robust:** "the two-stage pipeline meets a p99 < 100 ms SLA". It does warm (33.67,
  48.93) and does not cold (125.47). The failure is entirely in **stage 1, which D44 did
  not touch and both paths share**.
- **Robust across every run and both memory conditions:** D44 removes the *feature stage*
  as the bottleneck. `online_features` p99 is **6.70–8.27 ms** in all three runs, against a
  batch feature stage of **2,742–22,398 ms**. The p50 speedup is likewise stable
  (27.6x–33.9x on EB-NeRD; 19.5x on MIND).
- **The actionable consequence, now measured rather than asserted:** a serving process must
  **pre-fault the embedding matrix and indexes at start-up before accepting traffic**. The
  cost of not doing so is 121.49 ms at p99 on stage 1 alone — more than the entire SLA
  budget, from a stage whose warm p99 is 24 ms.

**Method note.** Three runs were taken because the first produced a number too good to
report (a 665x p99 ratio). Had only one been taken — in any of the three states — the
conclusion drawn would have been wrong in a different direction each time. This is the
2026-09-16 error-log entry generalising: on this machine, a latency measured once is not a
measurement.

### K.10 D43 word-level NRMS — completed 2026-09-21, 20.2 h

Two arms on MIND, word-level news encoder (GloVe 300d, self-attention over title tokens,
additive pooling), 60,000 train impressions, 3 epochs, epoch chosen by AUC on a seeded
30,000-impression validation sample, **evaluated on the whole 21,947-impression test split**.
1,000-resample paired bootstrap. `reports/nrms_ablation_mind_test_word.csv`.

**Validation traces (these SELECTED the epoch; they are not results):**

| | ep 1 | ep 2 | ep 3 | chosen |
|---|---|---|---|---|
| sentence-level baseline (D41) | 0.5967 | 0.6039 | 0.6107 | ep 3 |
| word-level baseline | **0.6305** | 0.6269 | 0.6273 | ep 1 |
| word-level + fresh + exposure | 0.6229 | 0.6141 | 0.6133 | ep 1 |

**Test split — the reportable numbers:**

| system | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| word-level `nrms` | 0.6129 [0.6091, 0.6167] | 0.3110 | 0.2931 | 0.3567 |
| word-level `nrms+fresh+exposure` | 0.6161 [0.6122, 0.6201] | 0.3140 | 0.2980 | 0.3596 |
| **paired difference** | **+0.0032 [0.0018, 0.0045]** | +0.0031 [0.0011, 0.0050] | +0.0049 [0.0030, 0.0067] | +0.0029 [0.0012, 0.0045] |

**Finding 1 — the word-level encoder overfits; D39's substitution cost nothing.**

| MIND baseline | validation | test |
|---|---|---|
| sentence-level, 3 epochs | 0.6107 | 0.6146 [0.6109, 0.6184] |
| sentence-level, 6 epochs | 0.6187 | 0.6180 [0.6143, 0.6221] |
| word-level | **0.6305** | **0.6129 [0.6091, 0.6167]** |

On validation the word-level encoder leads by **+0.0198**. On test it is **indistinguishable
from the 3-epoch sentence-level model** (intervals overlap) and **below** the 6-epoch one.
11.0M parameters against 413k: it fits the validation window hard and does not transfer to
the later test window. Both word-level arms peak at **epoch 1** and decline, while the
sentence-level baseline was still rising at epoch 3 — and the 2026-09-20 epoch defect
(identical negatives every epoch) makes that overfitting more likely than it should be.

**This is D7/D36 working as designed.** Reporting the validation number would have claimed a
+0.02 improvement that does not exist. The split that chose nothing is what caught it.

**Finding 2 — D40's time term changes sign on MIND under a faithful encoder.**

| MIND, `+fresh+exposure − nrms` | paired ΔAUC | excludes zero? |
|---|---|---|
| sentence-level, 3 epochs | −0.0002 [−0.0011, 0.0008] | no |
| sentence-level, 6 epochs | −0.0001 [−0.0011, 0.0009] | no |
| **word-level** | **+0.0032 [0.0018, 0.0045]** | **yes** |

The MIND null was partly an artefact of the substituted news encoder. It is small — EB-NeRD's
is +0.0864 — but it is now separated from zero on all four metrics.

**A caveat that applies to both findings.** The word-level and sentence-level numbers come
from separate runs, so the *cross-encoder* comparison is unpaired; only the within-encoder
arm differences are paired. And validation misled about the direction of BOTH comparisons —
on validation the time term appeared to *hurt* by 0.0076, on test it helps by 0.0032. On
MIND, with high-capacity models and a temporal split, validation AUC is a poor predictor of
test AUC, and that is itself a reportable observation.

**Cost:** 20.2 h wall clock (estimated 16.6 h). Per-epoch time drifted upward, 2.2 h → 3.6 h;
test scoring took ~1 h per arm because `score()` makes one `model.news` call per impression.

### K.11 D42c — the retrieved regime inflates rack features; the number is not reportable as a gain

`run_retrieved_regime.py --models base rack` scores both models over A1's semantic top-100
(MIND test, macro recall@100 = 2.62%, so only **798 of 21,947** impressions have a clicked
article among the 100 and carry a defined metric).

| system | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| retrieval order (A1 semantic) | 0.5668 | 0.0695 | 0.0419 | 0.0699 |
| lgbm re-ranked (base) | 0.7858 | 0.2603 | 0.2480 | 0.2895 |
| lgbm + rack re-ranked | **0.9222** | 0.4484 | 0.4851 | 0.5313 |
| rack − base (paired) | **+0.1364** | +0.1881 | +0.2372 | +0.2419 |

**None of that is evidence the rack features are good, and it must not be quoted as such.**
AUC 0.9222 against 0.6367 for the same model on the inview lists is the giveaway. The cause,
measured rather than guessed:

| MIND, mean `exposure_share_1h` | clicked | unclicked | ratio |
|---|---|---|---|
| inview lists (the training distribution) | 0.428 | 0.321 | **1.33x** |
| retrieved top-100 | 0.355 | **0.00316** | **112x** |

Mean `freshness_hours`: inview 21.2 (clicked) vs 27.9 (not); retrieved **23.0 vs 82.8**.
Mean `exposure_share_1h_pct` in the retrieved regime: **0.983 clicked vs 0.500 unclicked** —
the clicked article is essentially always the most-exposed item in its rack.

**Why.** A retrieved rack holds one article the platform was actively promoting — it was in
the user's inview list, which is how it came to be clicked — plus 99 semantic neighbours
drawn from the in-circulation pool, most of which are barely being shown. "Which of these is
being promoted right now" separates them almost perfectly, and exposure answers exactly that.
The regime inflates the **base** model too (0.7858 vs 0.6367 on inview); rack normalisation
inflates it further because a within-rack percentile turns a 112x gap into "top of rack".

**This reverses D42c's stated prediction.** D42c expected rack features to *degrade* when the
reference set changed from a 9-25 candidate inview list to 100 retrieved ones. They do the
opposite. The prediction was wrong in direction and the reason is more interesting than the
prediction: a relative feature does not merely lose meaning when its reference set changes —
it can acquire a spurious one, if the new set is composed differently with respect to that
feature.

**It is also D37's warning, quantified.** D37 said the retrieved regime's negatives are
"articles the user never saw, not articles the user saw and skipped", and declined to train a
second model on them for that reason. The 112x exposure ratio is the size of that gap.

**What to report:** the inview numbers (K.6), which is what both leaderboards score. The
retrieved regime belongs in the design note as a **cautionary measurement** — evidence for why
the supplied candidate list is the honest evaluation — not as a headline.
