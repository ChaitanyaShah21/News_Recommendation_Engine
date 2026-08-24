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

---
