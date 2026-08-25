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

