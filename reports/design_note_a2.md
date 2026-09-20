# Learning from Click-Logs: a Two-Stage Retrieve-then-Rank System for MIND and EB-NeRD

**Chaitanya Shah · Systems Design** · Chaitanya Shah
Done **solo**, which the specification permits for this team assignment.
Code: `src/newsrec/` · decision log: `ARCHITECTURE.md` (A1's D1–D32, A2's D33–D40) ·
every quoted figure traces to `reports/NUMBERS.md`.

---

## 1. What was built, and the one idea behind it

A1 ended on a measured gap: **BM25 and embedding retrieval contain no time term** — a
structural fact about both formulas — while **92.7% (MIND) / 93.5% (EB-NeRD) of real clicks
are on fresh articles**. Three independent routes reached it, the sharpest being that
train-window popularity predicts EB-NeRD clicks in the *wrong* direction (AUC 0.4647).

A2 acts on it. The A1 scorers are not replaced; **they become features** of a trained
re-ranker that also sees time:

```
 A1, unchanged                          A2, new
 BM25 index ┐                    ┌ features/ : recency-decayed history (3 half-lives),
 embeddings ├── scores as ──────►│             category affinity, freshness, windowed
 availability filter             │             exposure share, session context
            │                    └────────────────────┬───────────────────────────────
            └── top-K (K=100) ──────────────────────► rank/gbdt.py  LambdaRank (LightGBM)
                                                      rank/nrms.py  NRMS baseline (Q3)
                                                                    │
                                        metrics · slices · paired bootstrap (A1's, extended)
```

Because `cos_inf` (infinite half-life) reproduces A1's semantic score exactly and `bm25`
is A1's lexical score, **"before re-ranking" is one column of "after"** — the Q2
comparison is two orderings of the same candidate list, not two unrelated systems.

**Two candidate sets, deliberately kept apart (D37).** Both leaderboards ask us to
re-rank a *supplied* list (`article_ids_inview` / `impressions`); Q2 also asks us to
*retrieve* top-K from the whole corpus. These are different operations on different
candidate sets, and conflating them is the easiest way to report a meaningless number.
Every table below says which it is.

## 2. Q1 — features, and the boundary that makes them honest

| Feature | MIND | EB-NeRD |
|---|---|---|
| `cos_short / cos_medium / cos_inf` — decayed user profile · candidate (D35) | ✓ (3/10/∞ positions) | ✓ (1 d / 7 d / ∞) |
| `cat_share_short / medium / inf` — recency-weighted category affinity | ✓ | ✓ |
| `bm25` — A1's lexical score, N=100 (D31) | ✓ | ✓ |
| `freshness_hours` — T − min(published_time, first_seen) (D34c) | ✓ (first_seen only) | ✓ |
| `exposure_share_1h / _24h` — label-free share, strictly before T (D34b) | ✓ | ✓ |
| `hours_since_last_click` (D35) | — | ✓ |
| `session_position`, `minutes_since_session_start`, `device_type` (D34a) | — | ✓ |

**The feature set is an allowlist, never a denylist**, and the builder *asserts* its output
columns equal keys + label + that list. This is not tidiness: `total_pageviews` and
`total_inviews` already sit in our `articles.parquet` and are whole-dataset aggregates
containing the future, so any builder that loops over article columns picks them up and
nothing errors.

**Boundary enforcement (Q1.4, Q9.2).** `tests/test_no_leakage.py` asserts two *properties*
on real validation data for both datasets, rather than a list of per-feature checks:
*future deletion* (delete every impression after τ; features at or before τ must be
byte-identical) and *label blindness* (shuffle the clicks; every feature must be
identical). A new feature is covered automatically. The suite is mutation-verified: four
realistic forward leaks were planted in production code and **all four were caught** —
but only after extending the check to MIND, because on EB-NeRD `published_time` almost
always wins the `min`, so the `first_seen` path is never exercised there.

**Three landmines, each measured rather than assumed.** EB-NeRD's catalogue runs a month
past the test week (2,148 articles published after the train week ends), so **no
catalogue-wide statistic is ever a feature**. Its `session_id` is shared by multiple users
(1,032 ids), so sessions are keyed **(user, session)** — keyed by id alone the longest
"session" is 14 days, keyed properly 29 minutes. And 1,810 candidate rows are shown
*before* their `published_time`, which is why freshness takes the earliest evidence of
existence.

## 3. Q2 — the trained re-ranker, before and after

**Model (D36).** LightGBM with the `lambdarank` objective: it is scored only on ordering
each impression's clicked candidates above its unclicked ones, weighted by how much a swap
would move nDCG. Trained on the train split, tree count chosen by validation nDCG@10,
**every number below from the local test split**, which chose nothing. Hyperparameters were
fixed before any result was seen. LightGBM learns impression boundaries from group *sizes*
alone, so a shuffle trains happily and ranks garbage; `group_sizes` derives them from the
ids and refuses non-contiguous input.

### 3.1 Supplied inview lists — what the leaderboards grade

| System | MIND AUC | MIND nDCG@10 | EB-NeRD AUC | EB-NeRD nDCG@10 |
|---|---|---|---|---|
| random | 0.4982 | 0.2756 | 0.5007 | 0.4288 |
| bm25 (A1) | 0.5641 | 0.3265 | 0.5041 | 0.4305 |
| semantic (A1) — *before* | 0.6097 | 0.3588 | 0.5457 | 0.4585 |
| **LambdaRank (A2)** — *after* | **0.6144** | **0.3636** | **0.7428** | **0.6250** |
| paired gain over *before* | +0.0048 **[0.0012, 0.0080]** | +0.0048 | +0.1971 **[0.1953, 0.1989]** | +0.1665 |

Both AUC gains exclude zero. **The asymmetry is the finding, not a disappointment.**
Importances say why: on EB-NeRD freshness (38%) and exposure share (34%) carry 73% of the
gain; on MIND `cos_inf` leads (27%) and the model saturates after 18 trees — validation
nDCG@10 is flat from ~20 to 300 trees while training climbs 0.396 → 0.457. Single-feature
AUCs on MIND put freshness at 0.534 and exposure at 0.533 against similarity's 0.610,
because **MIND ships no publication times and no history timestamps**: its freshness is
inferred from first appearance in a sampled log and its recency can only use list
position. The time signal helps where the data records time, which is also why the two
datasets' decay features are not comparable to each other.

### 3.2 Retrieved top-K (K = 100) — our own candidate generation

| | MIND | EB-NeRD |
|---|---|---|
| recall@100, in-circulation pool | 2.6% | 2.5% |
| impressions with ≥1 click retrieved | 798 of 21,289 | 4,704 of 186,721 |
| MRR, retrieval order → re-ranked | 0.070 → **0.260** | 0.052 → **0.567** |
| AUC, retrieval order → re-ranked | 0.567 → 0.786 | 0.508 → 0.968 |

Two honest readings. First, **recall@K is the ceiling**: re-ranking cannot rescue the
97.5% of impressions whose clicked article never entered the top 100, which is why both
leaderboards score a supplied list instead. Second, **EB-NeRD's 0.968 is not a better
system than §3.1's 0.743**: the 99 negatives here were drawn from a 125,541-article corpus
by text similarity, so most were not in circulation that day, and freshness separates them
almost trivially. Easy negatives inflate AUC. The supplied-list number is the one to
believe.

## 4. Q3 — baseline reproduced, then beaten

**Baseline (D39): NRMS** (Wu et al., 2019) — multi-head self-attention over the user's
clicked articles, then additive attention to pool them, scored by dot product against the
candidate, trained with one click against four same-impression negatives. **One documented
deviation:** the news encoder is a learned projection of our existing 384-dim multilingual
sentence embeddings instead of word-level attention over title tokens. That is what makes
it trainable on a GPU-less laptop; the user encoder, objective and scoring are the paper's.

**The one principled change (D40): a time term.** NRMS has no time input at all. A small
MLP over (log freshness, exposure share 1 h, exposure share 24 h) is added to the click
score, zero-initialised so training *starts* at exactly the baseline. The change was
chosen from A1's measurements **before** any NRMS result existed — A1's Finding 5 is that
selecting a claim after seeing its result is a tautology in disguise.

**Ablation (Q3.3) and significance (Q3.4).** Four arms, identical in everything but the
time inputs, each trained on the same seeded 60,000 impressions (D41) and evaluated on the
whole test split. Paired bootstrap, 1,000 resamples, differences against the baseline.

| Arm | MIND AUC | paired Δ | EB-NeRD AUC | paired Δ |
|---|---|---|---|---|
| nrms (reproduced baseline) | 0.6146 | — | 0.5874 | — |
| + freshness | 0.6159 | +0.0013 [0.0004, 0.0022] | 0.7014 | +0.1141 [0.1126, 0.1155] |
| + exposure share | 0.6122 | **−0.0024** [−0.0032, −0.0014] | 0.6244 | +0.0370 [0.0359, 0.0380] |
| **+ both — the pre-registered change** | 0.6144 | −0.0002 [−0.0011, 0.0008] | **0.6738** | **+0.0864 [0.0852, 0.0877]** |

**On EB-NeRD the claim holds with room to spare: +0.0864 AUC, CI clear of zero**, and
+0.0628 MRR. **On MIND it does not**: the pre-registered arm is indistinguishable from the
baseline on AUC (−0.0002, interval straddling zero) though it does lift MRR
(+0.0053 [0.0038, 0.0067]).

What the ablation buys over a single "improved" number: it **localises** the gain
(freshness carries it; exposure alone adds +0.037 and *subtracts* when combined, the two
being correlated — a fresh article is also a heavily-pushed one); it **reproduces the Q2
ordering in a different model class** (LambdaRank importances also rank freshness first on
EB-NeRD and find exposure inverse on MIND), which is the kind of agreement that makes a
finding believable; and it **says where the change does not transfer, and why** — MIND has
no publication times and no history timestamps, so there is less time signal to add.

**Two limits we state rather than let a reader discover.** The baseline is trained on
60,000 impressions for 3 epochs and its validation AUC was **still rising at the last
epoch**, so these are *undertrained* NRMS numbers — the ablation is valid because every arm
has the same budget, but the absolute levels understate a fully-trained NRMS. And
`+ freshness` beats the pre-registered `+ both` on both datasets; we report that as an
ablation observation rather than promoting it to the headline after the fact, because
choosing the winner after seeing the results is precisely the error A1's Finding 5
records.

## 5. Q4 — serving: memory, p99, and cost per 1,000 queries

Measured per **single user request** (K = 100), single process, **with nothing else running**
— NRMS training was suspended for the measurement, because A1's error log already contains
a conclusion that had to be withdrawn after being taken on a contended machine.

| | MIND | EB-NeRD |
|---|---|---|
| resident footprint, total | **283 MB** | **475 MB** |
| — largest items | embeddings 100 MB · user vectors 77 MB · exposure index 71 MB · BM25 19 MB | embeddings 193 MB · exposure index 123 MB · user vectors 61 MB · session table 44 MB |
| start-up (context build) | 5.2 s | 137 s at leaderboard scale |
| stage 1, retrieve top-100 | p50 **9.0 ms**, p99 23.7 ms | p50 **9.6 ms**, p99 22.2 ms |
| stage 2, build features | p50 **223.0 ms**, p99 348.0 ms | p50 **439.0 ms**, p99 2,352 ms |
| stage 3, score 100 candidates | p50 **3.2 ms**, p99 9.0 ms | p50 **4.3 ms**, p99 6.9 ms |
| **total** | p50 237 ms, **p99 360 ms** | p50 452 ms, **p99 2,369 ms** |
| throughput per core | 4.2 q/s | 1.5 q/s |
| **cost per 1,000 queries** (at $0.04 / vCPU-hour) | **$0.0027** | **$0.0077** |
| cores for 1,000 QPS | 239 | 689 |
| **p99 < 100 ms SLA** | **not met** | **not met** |

**Little's law check.** With one request in flight X ≤ 1/R: MIND's R = 0.237 s gives
X ≤ 4.2 q/s, exactly the measured per-core throughput — so the only way past it is to cut R
or add concurrency.

**We fail the SLA for an architectural reason, not an algorithmic one.** Stage 1 (9 ms) and
stage 3 (3–4 ms) are inside budget; **feature assembly is 93–97% of the request**, because
we call a *batch* builder once per request:

- Only **13.5% (MIND) / 16.8% (EB-NeRD)** of that stage is per-user profile work — three
  decayed profiles, three category-share tables, the BM25 query — measured separately. So
  "precompute user profiles nightly" is necessary but **not** sufficient; it buys ~30 ms of
  223 ms.
- The remainder is fixed per-call cost: DataFrame joins against whole-dataset tables (a
  1.2M-row session table, 125k articles, a 65k-entry category map) to answer a 100-row
  question. The same code processes **~11,500 rows/s in bulk** (2.25M rows in 195 s) and
  **~450 rows/s one request at a time** — a ~26× penalty that is all overhead.

**What would meet p99 < 100 ms** (a projection, not a measurement): precompute user
profiles into a key-value store (−30 ms), replace per-request joins with array lookups
keyed by article row (the targets are static within an hour), and cache article features by
(article, hour) — the structure §8 also wants at 10×. Stages 1 and 3 sum to ~13 ms, so a
~20 ms lookup path puts p99 near 40 ms and cost near $0.0003 per 1,000. **We did not build
it**, and saying so beats quoting a number we did not measure. In batch the same system is
cheap and fast: 206M rows scored in 155 min, ~22,000 rows/s. It is a strong batch scorer
and a weak online one.

## 6. Q5 — extended evaluation, sliced

All seven metrics, over the supplied-inview pipeline on the local test split, with
1,000-resample percentile CIs. Slices are D26's: cold = history ≤ 5 articles; head/tail by
exposure, where an impression is head only if *every* clicked article is in the head set
(2,922 MIND / 243 EB-NeRD impressions are mixed and counted in neither, rather than
silently assigned).

| | MIND: semantic (A1) → LambdaRank | EB-NeRD: semantic (A1) → LambdaRank |
|---|---|---|
| all | 0.6097 → 0.6144, paired **+0.0048 [0.0012, 0.0080]** | 0.5457 → 0.7428, **+0.1971 [0.1953, 0.1989]** |
| cold (n = 3,904 / 680) | 0.5642 → 0.5529, **−0.0113** | 0.5538 → 0.7348, **+0.1810** |
| warm (n = 18,043 / 186,041) | 0.6195 → 0.6277, **+0.0083** | 0.5457 → 0.7428, **+0.1971** |
| head-exposure (n = 13,108 / 98,719) | 0.5955 → 0.5828, **−0.0127 [−0.0176, −0.0079]** | 0.5565 → 0.7745, **+0.2179** |
| tail-exposure (n = 5,917 / 87,759) | 0.6319 → 0.6671, **+0.0353** | 0.5334 → 0.7076, **+0.1741** |
| diversity (category, K=10) | 0.8166 → 0.7352 | 0.7993 → 0.8029 |
| novelty (bits) | 17.34 → 17.69 | 18.87 → 19.06 |
| coverage | 0.0283 → 0.0318 | 0.0260 → 0.0247 |

Three readings we would defend:

1. **On EB-NeRD the gain is not an artefact of one population** — it holds in every slice,
   largest on head-exposure (+0.218) and smallest on tail (+0.174). A model leaning on
   popularity should be expected to do *better* where articles are widely shown, and it
   does, which is a consistency check rather than a surprise.
2. **On MIND the aggregate +0.0048 hides two sign flips, which is exactly what slicing is
   for.** The model is *worse* on cold users (−0.0113) and *worse* on head-exposure
   impressions (−0.0127 [−0.0176, −0.0079]), and better on warm (+0.0083) and much better
   on tail-exposure (+0.0353). The mechanism is measurable rather than guessed: on MIND a
   candidate's own exposure share predicts clicks in the **inverse** direction within its
   own rack (AUC 0.476 taken positively, 0.524 inverted), so the model learns to
   down-weight heavily-shown articles. That helps where the clicked article is obscure and
   hurts where it is the popular one. A cold user additionally has no decayed profile at
   all, so nothing compensates. Together these argue for a cold-start fallback and for
   rack-relative rather than absolute popularity features — the improvement we would make
   next, and the one §4's GBDT property predicts.
3. **The re-ranker is not free on diversity.** On MIND it trades category diversity
   (0.817 → 0.735) for novelty and coverage; on EB-NeRD it costs coverage (0.0260 → 0.0247)
   while raising novelty. Coverage carries no CI (D27): the bootstrap is biased downward for
   a union statistic, so the raw spread sits in the CSV under its own column names.

## 7. Q9 — anti-gaming: leakage test, and the price of a serving-time feature

**The test (Q9.2).** Two properties, not a checklist: future deletion and label
blindness (§2), mutation-verified against four planted leaks. A quarantined future feature
*fails* the same check, which is how we know the check can see a leak at all.

**The ablation (Q9.1).** The shipped model is trained twice: on the allowlist, and on the
allowlist plus everything in `features/unavailable.py` (`read_time`,
`scroll_percentage`, `total_pageviews`, `future_exposure_share_24h`).

| MIND (test, 21,947) | AUC | MRR |
|---|---|---|
| honest — **this is what ships** | 0.6144 [0.6104, 0.6185] | 0.3191 |
| + serving-unavailable | 0.5994 [0.5952, 0.6033] | 0.3014 |
| paired difference | **−0.0150 [−0.0181, −0.0118]** | −0.0177 |

| EB-NeRD (test, 186,721) | AUC | MRR |
|---|---|---|
| honest — **this is what ships, and what both leaderboard files came from** | 0.7428 [0.7415, 0.7441] | 0.5219 |
| + serving-unavailable | 0.7590 [0.7578, 0.7602] | 0.5487 |
| paired difference | **+0.0162 [0.0153, 0.0172]** | +0.0269 |

**The price of cheating on EB-NeRD is +0.016 AUC: real, but under a tenth of what the
honest behavioural features bought (+0.197).** The two columns that carry it are
`total_pageviews` (17% of the leaky model's gain) and `future_exposure_share_24h` (16%).

**`read_time` and `scroll_percentage` — the fields the schema most invites you to misuse —
are inert here, for a structural reason.** EB-NeRD records them **per impression**, so they
are constant across a rack, and a ranking objective only compares candidates *within* a
rack; neither reaches the leaky model's top five. **The dangerous leak is the one that
varies per candidate**, which is why `total_pageviews` — a per-article whole-dataset
aggregate — is the biggest offender here.

**Cheating made MIND worse, and the reason is mechanical, not moral.** MIND ships no dwell
or pageview fields, so the only quarantined column with data is the forward exposure
window — and MIND's log ends 2019-11-15 23:58, so that window is 37–138 h wide in training
but only 5.7–11.4 h at test time, inflating the feature's mean 3.2× (0.075 → 0.241). The
leaky model leaned on it (118 trees against 18; 15.7% of its gain) and was then handed a
differently-distributed input. **So MIND cannot price the leak**; EB-NeRD, where all four
columns exist and three are not window-dependent, is where that price is measurable.

## 8. Where it breaks at 10×

The honest basis for this section is that we **already ran the largest thing we have**:
205,925,868 candidate rows (EB-NeRD) and 93,115,001 (MIND) were featurised and scored
locally, in **155 min** and **26 min**, at a 10.6 GB peak on an 11 GB machine. So what
follows extrapolates from a measured run rather than from an estimate.

**Three things that made the current scale run.** (1) *Never materialise the exploded
candidate list as strings*: 206M (impression, article) pairs cost ~10 GB as strings and
1.4 GB as one sorted `int64` key per pair. (2) *A deep slice of a lazy frame is not free
once a row index is attached*: one 5,000-row chunk at offset 5M cost **297 s**, fixed by
writing the prepared frame to Parquet first. (3) *Partition by user, not by file order*:
per-user work (~3.4 ms) was being repeated in every chunk a user appeared in; hashing users
into 64 groups pays it once, 2.7 min per 215k-impression group.

**At 10× (135M impressions, ~2 G candidate rows), in the order things break:**

- **The exposure index, first.** 2 G `int64` keys is **16 GB** — it stops fitting. It must
  become either time-bucketed shards or a count keyed by (article, hour) instead of by
  impression, which is a bounded-size structure: 125k articles × 168 hours is 21M cells,
  not 2 G.
- **Per-user feature building, second.** It is linear and precomputed nowhere: ~8 M users ×
  3.4 ms ≈ **7.5 h** of profile building alone. A production system would move it to a
  nightly batch, because a user profile is a function of history, not of the request.
- **Wall clock, third.** ~26 h on this machine, which is past any deadline — but the work
  is embarrassingly parallel by user group, so it is a horizontal-scaling problem rather
  than an algorithmic one. Nothing in the pipeline requires a global shuffle.
- **What does *not* break:** the model (322 trees, 1.6 MB), the embeddings (190 MB), and
  BM25 (a fixed sparse matrix). Stage 1 and stage 2 both stay within a single machine's
  memory at 10×; it is the *log-derived* features that grow with traffic.

**One caveat about numbers taken under pressure.** Profiling one leaderboard chunk showed
`searchsorted` at 0.7 s per call; the same call on a quiet machine takes **10 ms**. The
difference was memory pressure, not the array. A1's error log already contains a
conclusion that had to be withdrawn for exactly this reason, which is why the latency
figures in §5 were taken with nothing else running.

## 9. The external check: both leaderboards

| | A1's submitted system | A2 (this system) | our offline test AUC |
|---|---|---|---|
| MIND | 0.6191 | **0.6181** (rank 69) | 0.6144 |
| EB-NeRD | 0.5396 | **0.7397** (rank 77) | 0.7428 |

**EB-NeRD gains +0.2001 AUC over A1, and our offline harness predicted that level to within
0.0031** — on 13.5M unseen impressions, scored by the organisers' own program. That is the
strongest evidence we have that the offline numbers in this note mean something.

**On MIND the two systems are indistinguishable (−0.0010), although offline we measured
+0.0048.** We take that at face value rather than explaining it away: the MIND gain was
marginal to begin with, its confidence interval [0.0012, 0.0080] was narrow but small, and
§6 already showed the aggregate concealing a **−0.0127** loss on head-exposure
impressions — which is most of the leaderboard. The offline *level* transferred on both
datasets (≤0.004); the offline *delta* transferred only where the signal was large.

## 10. Reproducing, and what is ours

infrastructure, not pipeline: the organisers' scoring program, hidden reference data and
metrics are used unmodified.
