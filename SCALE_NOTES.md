# Where This Breaks at 10×

System scale analysis asks where the pipeline breaks at ten times the scale.
Observations are captured here **as they happen**, not reconstructed at the end.

Format: what we observed → what it implies at 10× → what we would do about it.

---

### The inverted index itself scales fine; the score matrix does not
**Observed (2026-08-24, Phase 2):** building the BM25 index over the real corpora took
**1.70 s for MIND** (65,238 docs, 60,951 terms, 2.36 M non-zeros) and **0.19 s for
EB-NeRD** (11,777 docs, 31,642 terms, 269 K non-zeros). Peak process memory 0.51 GB
against 7 GB available. The sparse representation is **19.2 MB** where the dense
equivalent would be **15.9 GB** — a 830× saving, at 0.0594% density.

**What it implies at 10×:** the index is not the problem. Document count and vocabulary
both grow sub-linearly in cost here (vocabulary growth follows Heaps' law, so 10× the
documents gives well under 10× the terms), and 10× MIND is ~190 MB of CSR — still
trivial. Build time would be ~17 s, dominated by the pure-Python tokenise-and-count
loop, not by scipy.

**The actual wall is on the query side**, and it is not sparse. Scoring produces a dense
`n_queries × n_docs` score matrix: MIND's val alone is 37,777 unique users × 65,238
documents × 4 bytes ≈ **9.9 GB**, already above this machine's RAM before any 10×.
EB-NeRD-large's 13.5 M impressions would be far worse.

**What we would do about it:** batch the query side — process ~500 users at a time, take
top-K per row, discard the scores. Memory then depends on batch size, not on user count.
This is a bound we hit at *demo* scale, not at 10×, so the batching is not premature.

**Update (2026-08-25), now measured rather than projected:** batching implemented at
`batch_size=256`; peak memory stayed flat and MIND val retrieval ran in **198–218 s** for
37,777 users × 65,238 articles. Confirmed by test that batch sizes 1, 7 and 1000 produce
byte-identical results.

**But retrieval time is now the binding constraint, not memory.** ~200 s per run at
MIND-small scale, and the cost scales with (queries × candidate pool). MINDlarge_test has
**2.37 M impressions** against MIND-small val's 51,205 — roughly **46×** — and a larger
article corpus besides, projecting to **2.5–3 hours per configuration** single-threaded.
Four configurations (2 pools × 2 query variants) would be a full day.

**What we would do about it at 10×:** (a) the `max_df` knob from D11 — a query containing
*the* forces the product to walk a 50,233-entry posting list to earn an IDF of 0.26;
(b) drop to one query variant now that D16's ablation shows raw-vs-binary is near-noise;
(c) parallelise across batches, which is embarrassingly parallel and currently
single-threaded; (d) for Codabench, note the leaderboard task is *re-ranking* a supplied
candidate list (~38 articles for MIND), not whole-corpus retrieval — so Q5 does not pay
this cost at all. Only Q2/Q3's recall@K does.

---

### Time-bucketed retrieval multiplies work by bucket count, not by impressions
**Observed (2026-08-25, Phase 2, D19):** restricting candidates to articles in
circulation makes the candidate pool time-dependent, so the "score once per unique user"
saving no longer applies — the same user must be re-ranked in each time bucket they
appear in. At 1-hour buckets: EB-NeRD went from 1,562 user-retrievals to **8,777 tasks**
(5.6×, though still only 6 s), MIND from 50,000 to **47,998 tasks** (roughly break-even,
because MIND's val window is only 13 hours long).

**What it implies at 10×:** the multiplier is (buckets a user appears in), which grows
with the *span* of the evaluation window, not its row count. EB-NeRD-large's validation
window is far longer than the demo's 5 days, so this factor grows there while MIND's
stays near 1. A coarser bucket (6 h, 1 day) trades accuracy of the availability cutoff
for a smaller task count, and the cost of that trade is measurable — worth quantifying
before assuming hourly is necessary.

---
### Exact brute-force search is fine now and is the first thing to break at 10×
**Observed (2026-08-25, Phase 3, D21):** Q3.2 permits brute force "for small scale", and
at our scale it is the better choice — exact, so it cannot cost recall, and the whole
MIND score computation is roughly 10^12 multiply-adds, well under a minute of optimised
linear algebra. The binding constraint is memory, not arithmetic: a full
37,777 users x 65,238 articles float32 score matrix is **9.9 GB against 7 GB of RAM**, so
it must be batched. This is the *same* constraint D14 hit for BM25, arrived at from a
completely different direction (dense vectors rather than a sparse product).

**What it implies at 10x:** the score matrix is `users x articles`, so it grows with the
**product** of both, while the embedding matrix itself grows only with `articles x 384`
(65,238 articles is ~100 MB - trivial, and it stays trivial). Batching hides this until
the per-batch slice itself stops fitting, at which point the fix is no longer a smaller
batch but a real ANN (Approximate Nearest Neighbour) index, which trades exactness for a
sublinear number of comparisons. That is the crossover point where FAISS/ScaNN stop being
optional. EB-NeRD-large's corpus is 125,541 articles (10.7x the demo's 11,777), which
moves the matrix but not yet the model.

**Second, less obvious axis:** unlike BM25, embedding *inference* is a one-off cost that
scales with **articles**, not with impressions or users - and it is CPU-bound here with no
GPU. That cost is measured separately; the point for Q6 is that lexical and semantic
retrieval have differently-shaped scaling curves, so "which is cheaper at 10x" has a
different answer than "which is cheaper now".

---


## The re-ranking path is list-of-arrays shaped, and that is what breaks first at 10^7

**Observed (2026-08-25, Phase 4 → 5 handoff, measured not estimated).**
`rerank.build_candidate_set` on MIND val: **51,205 impressions / 1,895,867 candidates in
0.9 s**, +0.09 GB — 18.5 µs per impression. Extrapolated to the test bundles:

| | impressions | build time | per-object overhead, **one** list |
|---|---|---|---|
| MIND val (measured) | 51,205 | 0.9 s | 0.006 GB |
| MIND test | 2,370,727 | ~0.7 min | ~0.27 GB |
| **EB-NeRD test** | **13,536,710** | **~4.2 min** | **~1.52 GB** |

**Compute is not the problem — object overhead is.** A NumPy array carries ~112 bytes of
Python/array-header overhead regardless of its contents, and the re-ranking path holds
*three* parallel lists of one small array per impression (`candidate_rows`, `labels`, and
each method's scores). At 13.5 M impressions that is **~4.5 GB of pure overhead** before a
single article id is stored, against 7 GB of RAM locally and ~13 GB on a free cloud
notebook. The actual data — 162 M candidate ids at int32 — is only 0.65 GB.

**Why it was invisible until now:** at 10^4 impressions the overhead is 6 MB and the shape
is the *right* one, because impressions have genuinely ragged candidate counts (MIND val:
2 to 295) and a ragged list is the honest representation. The design is not wrong; it is
correct at the scale it was built for and wrong two orders of magnitude up. That is
exactly the shape of answer Q6 asks for.

**What Phase 5 must therefore do:** process the test split in **chunks of impressions**,
writing predictions incrementally, never materialising the whole split. The flat-array
alternative (`np.concatenate` of all candidates plus an offsets array — the same layout
`bootstrap_coverage` already uses for its ragged gather) removes the overhead entirely and
is the right fix if chunking proves awkward, but chunking is smaller and does not require
rewriting the scorers.

**Second Phase 5 cost, also measured rather than assumed:** article embedding is CPU-bound
at **96 articles/s** (Phase 3, D22). MIND test ships 120,961 news articles and EB-NeRD
large 125,541 — **~21 and ~22 minutes of CPU inference each**, ~43 min total, one-off.
That is a direct input to the cloud-platform choice: a GPU turns it into a couple of
minutes, and it is the only part of the pipeline that would benefit from one.

---

---

## Phase 5 — what the real test bundles did and did not break

**Measured 2026-08-25 on MINDlarge_test (2,370,727 impressions) and ebnerd_testset
(13,536,710 impressions), locally, on 7.7 GB of RAM with ~2.5 GB actually free.**

### The 13.5 M-row scare was a materialisation problem, not a size problem

Computing the distinct candidate set across **all 13,536,710** EB-NeRD test impressions —
roughly 135 million list elements — took **1 second at 1.03 GB peak** using
`pl.scan_parquet(...).collect(engine="streaming")`. The same question asked eagerly would
have needed several GB.

This is the concrete form of the note already recorded from Phase 4: the re-ranking path
breaks on *object overhead*, not compute. Polars' streaming engine never builds the
intermediate. Any code we write that calls `.to_list()` or holds one small NumPy array per
impression does, and that is where the 4.5 GB goes.

### The scale factors that actually bit

| Quantity | Dev scale | Test scale | Factor |
|---|---|---|---|
| MIND users needing a history read | 33,617 | 702,005 | 21× |
| EB-NeRD users needing a user vector | 4,714 | 807,677 | 171× |
| EB-NeRD history entries, untruncated | ~430 K | **~67 M** | 156× |
| Articles to embed | 77,015 | 246,502 | 3.2× |

**The one that forced a code change is the third.** `build_user_vectors` calls
`.to_list()` on `history_article_ids`. At 67 million ids that is ~67 million Python
strings — several GB, on a machine with ~2.5 GB free — for data of which **all but the
last 10 per user is discarded immediately**. `load_submission_history` therefore truncates
inside the reader, cutting it to at most 8 million. The function was correct at demo
scale and correct at test scale; it was only *affordable* at demo scale. That is the
purest example in this project of behaviour that changes at 170× while the logic stays
scale-independent.

### Where a GPU would and would not have helped

Only the embedding step: 246,502 articles at ~96 articles/s is ~43 min of CPU, and a free
T4 does it in ~3. Everything else — the streaming scans above, sparse BM25 products,
cosine scoring, writing predictions — is unaffected by a GPU. This asymmetry is what
decided D29, and it is worth stating in the design note because "use free-tier GPUs" is
the spec's own advice and it is only correct for one of the six steps.

**Observed, and worth a sentence of its own:** the embedding run's throughput fell from
the 96 articles/s measured in Phase 3 to **~50** when development work ran concurrently on
the same cores — a ~1.9× slowdown from nothing but contention. Single-machine benchmarks
that ignore what else is running on the box are optimistic by roughly a factor of two.

### A finding, not just a constraint

EB-NeRD's 13.5 M test impressions draw on only **10,451 distinct candidate articles** out
of a 125,541-article corpus (8.3%). MIND's 2.37 M impressions draw on 30,043 of 120,961
(24.8%). The scoring work is therefore bounded by the *candidate pool*, not the corpus:
we must embed the corpus once, but the score matrix that matters is tiny. This is the
same structural fact Phase 2 found from the other direction — EB-NeRD's in-circulation
set is small — arriving here as a compute budget rather than a recall ceiling.

### Phase 5 outturn: what the projections got wrong

Both large-scale estimates in this file were wrong in the same direction, and the reasons
differ:

| Step | Projected | Actual | Why |
|---|---|---|---|
| Embed both test corpora | ~43 min | **43.0 min** (24.4 + 18.6) | correct — but a mid-run flag claimed ~80 min, because throughput was sampled under CPU contention (~50 articles/s loaded vs 113–115 unloaded) |
| MIND submission | — | **6.4 min** @ 7,140 impressions/s | — |
| EB-NeRD submission | ~35 min | **6.7 min** @ 42,683 impressions/s | extrapolated from MIND's rate; EB-NeRD's racks are shorter (median 9 vs 25) and its scoring matrix is 10,451 columns vs 30,043 |

**The transferable lesson is that throughput does not transfer between datasets even for
identical code.** EB-NeRD has 5.7× more impressions than MIND yet finished in the same
wall-clock time, because per-impression cost is driven by rack size and candidate-matrix
width, not by row count. A capacity estimate that scales linearly in rows is wrong by 6×
here — in the safe direction this time, which is exactly why it would not have been
noticed had it gone the other way.

**And a measurement-hygiene note worth more than either number:** a benchmark taken while
other work runs on the same cores is not a benchmark of the code. The ~1.9× slowdown
observed under contention was enough to turn a correct 43-minute estimate into an
80-minute alarm.

---

## Engineering benchmarks (2026-08-26): latency, and the alternatives we rejected

Run after the course email made the grading criterion explicit — *"tool/db/index
choices, their impact on engineering metrics, how they compare to alternatives …
optimizations improving latency/throughputs"*. Two claims in the decision log were
arguments rather than measurements; this converts them. `scripts/benchmark_engineering.py`.

### Index construction

| | MIND (65,238 docs) | EB-NeRD (11,777 docs) |
|---|---|---|
| BM25 inverted index | **1.61 s**, 60,951 terms, 2,361,877 non-zeros, **19 MB** | 0.23 s |
| Embedding matrix load (Parquet → contiguous) | **0.85 s**, (65238, 384), **100 MB** | — |
| BM25 queries, all users | 0.64 s (50,000 users) | — |
| User vectors, all users | 0.88 s, 77 MB | — |
| Peak RSS | 1.05 GB | 0.58 GB |

**The dense embedding matrix is 5.2× the size of the sparse BM25 index** (100 MB vs 19 MB)
for the same corpus. Sparse wins decisively on storage. It does not win on speed — below.

### Per-query retrieval latency, top-200, batch size 1 (the serving number)

| method | p50 | p95 | p99 |
|---|---|---|---|
| BM25 sparse (D14, ours) | 10.10 ms | 13.47 ms | 20.21 ms |
| semantic brute-force (D21, ours) | **1.73 ms** | 4.51 ms | 6.50 ms |

**Semantic retrieval is 5.8× faster than BM25 per query, which inverts the usual
intuition.** A dense (65238 × 384) float32 product is 100 MB of sequential,
SIMD-friendly, cache-predictable work handed to BLAS; a sparse query vector against a CSR
matrix does irregular scattered gathers. *Sparse means less arithmetic, not less time.*
It also retires any worry that D21's "brute force" was a compromise: at this scale it is
the faster option as well as the exact one.

**Re-ranking is ~1,000× cheaper than retrieval**: 0.01 ms per impression on MIND
(35.9 candidates mean), 0.00 ms on EB-NeRD (11.8). This is precisely why the leaderboard
task is tractable over 13.5 M impressions while whole-corpus retrieval over the same set
would not be — and it is the quantitative form of the retrieval-vs-re-ranking distinction.

### What batching actually bought (D14/D21), quantified

| batch size | semantic throughput |
|---|---|
| 1 | 620 queries/s |
| **32** | **1,269 queries/s** |
| 256 | 727 queries/s ← *regression* |

**Correction to how D14/D21 described batching.** It was framed as an optimisation; it is
really a *memory necessity* with a modest throughput effect — ~2x at the optimum on the
day it was measured.

> **Re-measured 2026-09-16 and the shape changed: 142 / 360 / 420 queries per second at
> batch 1 / 32 / 256, i.e. monotone increasing, with no regression at 256.** The original
> run's peak-then-regress profile (620 / 1,269 / 727) did not reproduce, and neither did
> its absolute level: the same script on byte-identical data was ~2.4x slower overall
> today. Absolute throughput and the location of the optimum are therefore
> **machine-state-dependent** (frequency and core placement on a big.LITTLE ARM laptop),
> and the cache argument below is a plausible mechanism for the original observation
> rather than a demonstrated one. What survives both runs: batching helps, and by
> single-digit multiples, not orders of magnitude.

The original reasoning, kept for the record: performance degraded past ~32 because a
(256 × 65,238) float32 score block is 67 MB and no longer fits cache. The real justification stands unchanged — a full
37,777 × 65,238 matrix is 9.9 GB against 7 GB of RAM — but "batching makes it faster" was
never measured and is only true up to a batch of about 32.

### The rejected alternative, measured: `rank_bm25` (D14)

| | build | per query (p50) | full val run |
|---|---|---|---|
| `rank_bm25` | **0.61 s** | 2,183.84 ms | **30.3 hours** |
| ours (scipy.sparse) | 1.61 s | **10.10 ms** | **8.4 min** |

**216× per query on MIND; 692× on EB-NeRD** (693.34 ms vs ~1 ms over 11,777 docs).

**D14's stated justification was overstated and its conclusion was right.** It claimed
37,777 queries × 65,238 documents "would not finish". Measured, it finishes — in about a
day. Recorded as a correction rather than quietly restated, because an argument that
overshoots is a defect in the reasoning even when the decision it supports is correct.

**The genuine trade-off is build-time versus query-time work, and it runs the opposite way
to the headline.** `rank_bm25` *builds 2.6× faster* than our index (0.61 s vs 1.61 s)
because it merely stores tokenised documents and defers everything to query time. We pay
1.00 s more up front to construct a real document–term matrix, and recover it after

    1.00 s extra build / 2.174 s saved per query = **0.46 queries**

i.e. our index has paid for itself before the first query has finished. That is the
defensible form of the argument: not "the library is slow", but "the library moves the
work to the wrong side of a boundary crossed 50,000 times".

---

# Assignment 2

## Phase A1 — ingesting `ebnerd_large` (2026-09-15)

**The sample's size is nearly free at ingestion; the scan is not.**
- *Ad hoc script, filter applied after the read:* a 5% and a 15% user sample both took
  ~50 s at ~5 GB peak. The time is dominated by reading all 24.6M behaviour rows plus
  2.4 GB of history, not by how many rows are kept.
- *Production path, filter inside the Parquet scan:* the whole store (MIND included)
  rebuilt in **40 s at 3.3 GB peak**. That is 1.7 GB less, because Polars skips row
  groups rather than materialising and discarding them.

**Where it breaks at 10×.**
- **Storage:** `history.parquet` is the heaviest raw file (1.24 GB train, 1.13 GB
  validation, median 92 / 80 clicks per user). Its list columns decompress far beyond
  their on-disk size.
- **Full `ebnerd_large`, unsampled (20× the rows):** would put history alone well past
  11 GB RAM. The current `.collect()`-to-DataFrame return type in `ingest_ebnerd` is the
  first thing to break, and would need to become a streaming or per-user-chunk write.
- **A 50% sample** would likely still fit.

**Size is paid downstream, and the next bottleneck is already visible.**
- **Training rows:** a re-ranker over retrieved candidates has one row per (impression,
  candidate). 597,348 train impressions × K=100 is ~60M rows, which exceeds RAM before a
  single feature is computed. Hence D36 subsamples training impressions separately.
- **Bootstrap:** its cost grows linearly with evaluated impressions (435,677 val now,
  against 17,749 in A1), ~25× per resample.

**Reuse beat recompute by 150×.** Assembling store embeddings from the Phase-5 testset
vectors took 7 s, against ~18 min to re-embed 125,541 articles on CPU. It only works
because the two article files were *checked* frame-equal first.

## Phase A3 — measured at leaderboard scale (2026-09-19)

**The scale that actually had to run: 205,925,868 candidate rows (EB-NeRD) and 93,115,001
(MIND).** Both leaderboard files were produced locally on the 11 GB machine.

| Quantity | EB-NeRD | MIND |
|---|---|---|
| Impressions / candidate rows | 13,536,710 / 205.9M | 2,370,727 / 93.1M |
| Wall clock, featurise + score + write | **155 min** | **26 min** |
| Peak RSS | 10.6 GB (of 11 GB + 16 GB swap) | lower, same code |
| Context built once | 137 s — 170,014,788 exposures over 14,556,456 impressions | — |

**Three things that decided whether this ran at all.**

1. **Never materialise the exploded candidate list as strings.** 206M (impression,
   article) pairs as Polars strings is ~10 GB; the same information as one sorted int64
   key per pair is 1.4 GB. `ExposureIndex` stores the key array, and `searchsorted` over
   it answers "how many impressions in [T−w, T) showed this article" in 40 ms per 59k
   rows.
2. **A deep slice of a lazy frame is not free once a row index is attached.** Slicing the
   raw leaderboard frame at offset 5M cost **297 s for one 5,000-row chunk**, because the
   row index forces the scan to walk everything before it. Writing the prepared frame to
   Parquet first and slicing *that* made the same operation instant. This is the same
   class of error as A1's: an operation assumed cheap because a *similar* one was measured
   cheap.
3. **Partition by user, not by file order.** Per-user work (three decayed profiles, the
   category shares, the BM25 query) costs ~3.4 ms and was being redone in every chunk a
   user appeared in. Grouping impressions by a hash of the user id pays it once: 2.7 min
   per 215k-impression group, and 64 groups covered the file.

**Where this breaks at 10×** (135M impressions, ~2 G candidate rows):
- The exposure key array is the first wall: 2 G int64 keys = **16 GB**, so it stops fitting
  and must become either per-shard (time-bucketed) indexes or a counting structure keyed by
  (article, hour) rather than per-impression.
- The per-user cost is linear and would reach ~8 M users × 3.4 ms ≈ **7.5 hours** of
  profile building alone. The fix is precomputation: user profiles are a function of
  history, not of the impression, so they belong in a nightly batch job, which is exactly
  how a production system would do it.
- Wall clock at 10× is ~26 h on this machine, which is past any submission deadline; the
  work is embarrassingly parallel by user group, so it is a horizontal-scaling problem
  rather than an algorithmic one.

**A slow measurement that was a machine-state artefact, not a code fact.** Profiling one
leaderboard chunk showed `searchsorted` taking 0.7 s per call over the 170M-key array;
re-measuring the same call on a quiet machine took **10 ms**. The difference was memory
pressure (the run was at 10.6 GB with swap active), not the array. Recorded because A1's
error log already contains one wrong conclusion drawn from a contended measurement.

---

## The serving path: the latency was never the arithmetic (D44, 2026-09-20)

Q4 reported the shipped pipeline at p50 237 ms (MIND) / 452 ms (EB-NeRD) per request
against an example SLA of p99 < 100 ms, with the feature stage taking 223 / 439 ms of it,
and the design note answered with a projection rather than a build. Profiling it says the
projection was aimed at the wrong thing.

**cProfile over 20 real single-user MIND requests, K = 100:**

| where the time goes | share |
|---|---|
| `PyLazyFrame.collect` — **66 separate Polars query collections per request** | 52% |
| joins (against 65,238-row article tables, to decorate 100 candidates) | 23% |
| `build_decayed_user_vectors` + `build_queries` (the per-user work) | 13% |
| `replace_strict` over a 65,238-entry category map | 12% |

Only the third line is work. The rest is a **batch pipeline being asked to serve one
request**: `assemble.build_rows` is correctly shaped for the 206 million candidate rows of
a leaderboard run, where a few milliseconds of fixed per-query cost vanish into minutes of
real work. Served one impression at a time it pays that fixed cost 66 times over on tables
of 100 rows.

**This is the general lesson, and it is worth more than the number.** Batch throughput and
serving latency are not the same optimisation, and a system tuned for one is usually
mis-shaped for the other. The same code that made the leaderboard run finish overnight is
what puts a 223 ms floor under a single request, and no amount of tuning inside that shape
would have moved it — the fix is a different shape, not a faster one.

**What D44 does about it.** Three tiers, and naming them is the design:

1. **Corpus-lifetime** — embeddings, the BM25 index, the exposure index, earliest-evidence
   times, categories. Already amortised; now additionally flattened into arrays indexed by
   article row, so a lookup is `a[rows]` rather than a join.
2. **Per-user, precomputable** — the three decayed profile vectors, three category-share
   tables and the BM25 query. Measured at 13.5% (MIND) / 16.8% (EB-NeRD) of the feature
   stage. These depend only on click history, never on the request.
3. **Per-request** — gather, six dot products, two `searchsorted` pairs, the rack
   normalisation. Pure NumPy.

**What it does not fix, stated rather than buried.** Tier 2 is work *moved*, not work
removed. A nightly profile job means a user whose history changed today is served from
yesterday's profile until it reruns — a freshness-for-latency trade a real system has to
make on purpose. On a dataset where 92.7% / 93.5% of clicks go to fresh articles, staleness
is not a free lunch, and the honest framing is that the request path gets faster while the
profile gets older.

**At 10×,** tier 1 is the wall again and for the same reason as the batch path: the
exposure key array is the first thing that stops fitting. Tiers 2 and 3 are linear in users
and candidates respectively and shard cleanly by user, so they scale horizontally. Tier 1
does not, because every request may touch any article.
