# Learning from Click-Logs: a Two-Stage Retrieve-then-Rank System for MIND and EB-NeRD

**Chaitanya Shah · Systems Design** · Chaitanya Shah
Done **solo**, which the specification permits for this team assignment.
Code: `src/newsrec/` · decision log: `ARCHITECTURE.md` (A1's D1–D32, A2's D33–D40) ·
every quoted figure traces to `reports/NUMBERS.md`.

---

## 1. What was built, and the one idea behind it

A1 ended on a measured gap rather than a result: **BM25 and embedding retrieval contain no
time term** — a structural fact about the two formulas, not a tuning failure — while
**92.7% (MIND) / 93.5% (EB-NeRD) of real clicks are on fresh articles**. Three independent
routes reached that conclusion, the sharpest being that train-window popularity predicts
EB-NeRD clicks in the *wrong* direction (AUC 0.4647), because 86.9% of validation
candidates were never clicked during training at all.

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

**Three landmines the data set for us, each measured rather than assumed.** EB-NeRD's
article catalogue runs to 2023-07-11, a month past the test week (2,148 articles are
published after the train week ends), so **no catalogue-wide statistic is ever a
feature**. EB-NeRD's `session_id` is shared by more than one user (1,032 ids), so sessions
are keyed **(user, session)**: keyed by id alone the longest "session" is 14 days, keyed
properly it is 29 minutes. And 1,810 candidate rows (11 articles) are shown *before* their
`published_time`, which is why freshness takes the earliest evidence of existence.

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
Feature importances say why: on EB-NeRD, freshness (38%) and exposure share (21% + 13%)
carry 73% of the model's gain; on MIND, `cos_inf` leads at 27% and freshness at 20%, and
the model saturates after 18 trees — validation nDCG@10 is flat from ~20 to 300 trees
while training nDCG@10 climbs 0.396 → 0.457. Single-feature AUCs on MIND put freshness at
0.534 and exposure at 0.533 against similarity's 0.610. **MIND ships no publication times
and no history timestamps**, so its freshness must be inferred from first appearance in a
sampled log, and its recency can only use list position. The time signal helps where the
data records time and cannot help where it does not; that is a property of the datasets
that the design had to absorb, and it is why the two datasets' decay features are not
comparable to each other.

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

<!-- TBD: NRMS ablation table (4 arms, paired CI) from reports/nrms_ablation_*_test.csv -->

## 5. Q4 — serving and scale

<!-- TBD: footprint / p50-p99 per stage / cost per 1000 queries, from reports/serving_benchmark_*.json -->

## 6. Q5 — extended evaluation, sliced

<!-- TBD: table from reports/extended_eval_*_test_k10.csv, both datasets -->

## 7. Q9 — anti-gaming: the price of a feature that is not available at serving time

<!-- TBD: reports/q9_serving_ablation_*_test.csv -->

## 8. Where it breaks at 10×

<!-- TBD: SCALE_NOTES.md + the leaderboard run's measured numbers -->

## 9. Reproducing, and what is ours

infrastructure, not pipeline: the organisers' scoring program, hidden reference data and
metrics are used unmodified.
