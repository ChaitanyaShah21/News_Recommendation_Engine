# Architecture & Decision Log

Last updated: 2026-08-25

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

### D3 — Unified schema: three tables (articles, impressions, history), no separate users table
**Date:** 2026-08-21 · **Decided by:** Chaitanya

**Chosen:** `articles`, `impressions`, `history` — one row per user in `history`, built by
collapsing MIND's per-impression-repeated history string down to one row per user
(confirmed safe: MIND's history is fixed per user for the whole logging window, not
updated impression-to-impression — verified against `notebooks/00_provided_mind_analysis.ipynb`).
Demographic columns (gender/age/postcode/subscriber) stay as mostly-null columns on
`impressions` rather than a separate `users` table.

**Alternatives rejected:**
- *Four tables, with demographics split into a separate `users` table.* More normalized,
  but Q4's required slices (cold-start vs. warm, head vs. tail) don't need demographics,
  and EB-NeRD's demographic fields are 97% null anyway — the extra join would mostly sit
  unused.
- *Two tables (`articles` + flat `impressions`), history duplicated inline per row.*
  MIND's native shape, but contradicts Q1.4's "small, reusable store" wording and
  discards the separation EB-NeRD's `history.parquet` already has.

**Why:** Matches the larger dataset's (EB-NeRD) native layout, keeps the store small,
and both datasets' impressions/history data map onto it without inventing structure
neither dataset actually has. [[unified-schema]] in `GLOSSARY.md`.

**Open sub-decision, deferred:** where article embeddings (EB-NeRD's provided vectors,
MIND's TransE entity embeddings) live — as columns on `articles` or a separate store.
Deferred to Phase 3 (Q3 semantic retrieval), since nothing before then touches embeddings.

**Column-level schema (concrete):**

`articles` — dataset, article_id (str, dataset-prefixed e.g. `"mind:N55528"` so the two
corpora never collide if ever concatenated), title, abstract (MIND's `abstract` /
EB-NeRD's `subtitle`), body (null for MIND), category, subcategory (null for MIND),
published_time (null for MIND), sentiment_score, sentiment_label, total_pageviews
(null for MIND, 87% null in EB-NeRD anyway), premium, article_type (null for MIND),
entities_raw (raw JSON string — MIND's `title_entities`/`abstract_entities` or EB-NeRD's
`ner_clusters`/`entity_groups`, kept unparsed until a phase actually needs entities).

`impressions` — dataset, impression_id (str, prefixed), user_id (str, prefixed),
timestamp, candidate_article_ids (list[str]), clicked_article_ids (list[str], empty for
unlabeled test rows), read_time, scroll_percentage, device_type, session_id (all four
null for MIND), split (added during the Q1.3 temporal split, not at raw ingest).

`history` — dataset, user_id (str, prefixed), history_article_ids (list[str]),
history_timestamps (list[datetime], null for MIND — MIND's history has no per-item
timestamp, only EB-NeRD's does).

**Judgment calls made without a separate decision point** (routine engineering defaults,
not results-changing — flagged per R6's own trivia exception, reversible if wrong):
dataset-prefixing article/user/impression IDs as strings even though EB-NeRD's are
native ints, and leaving entity annotations as unparsed raw JSON until a phase needs
them (YAGNI — You Aren't Gonna Need It, a principle against building things before
something actually requires them).

---

### D4 — venv + requirements.txt for dependency management
**Date:** 2026-08-21 · **Decided by:** Chaitanya

**Chosen:** Standard library `venv`, dependencies pinned in `requirements.txt`.

**Alternatives rejected:**
- *Poetry + pyproject.toml.* Better lockfile guarantees, but adds a tool prerequisite
  (installing Poetry itself) before Q1.5's one-command rebuild can even start.
- *conda/environment.yml.* Better for non-Python binary/CUDA dependencies, but there's
  no GPU locally and nothing here needs conda's package resolution.

**Why:** Simplest path to Q1.5's "one command rebuild" — works anywhere Python does,
no extra tooling. `.venv/` is gitignored; `requirements.txt` is the only committed
artefact. Pinned exact versions (`polars==1.43.2`, `pyarrow==25.0.1`) — polars matches
the version already used in the provided notebooks, confirmed by installing it and
checking `pl.__version__`.

---

### D5 — Polars over pandas for all DataFrame work
**Date:** 2026-08-22 · **Decided by:** inherited from constraints, confirmed with Chaitanya

**Chosen:** Polars everywhere — no pandas dependency anywhere in `src/newsrec/`.

**Alternatives rejected:**
- *pandas.* The default choice for most people's first exposure to DataFrames, but
  single-threaded and eager-only. EB-NeRD-large's behaviors file has 12M+ rows on our
  7 GB RAM, no-GPU machine (`system design contract` environment facts) — pandas would need careful
  manual chunking to avoid exhausting memory, where Polars' `.lazy()` / `scan_parquet`
  does it by default.

**Why:** Not really an open fork — the assignment spec itself says "ensure your
prediction pipeline is memory-efficient (use Polars, PyArrow, or batch processing)," and
both provided notebooks already use Polars exclusively. Logged here for completeness
since it wasn't written down explicitly until asked about directly.

---

### D6 — Drop EB-NeRD's context `article_id` field from the unified schema
**Date:** 2026-08-22 · **Decided by:** Chaitanya

**Chosen:** EB-NeRD's `behaviors.parquet` has a scalar `article_id` column, distinct
from `article_ids_inview`/`article_ids_clicked`, recording which article's page the
reader was already on when a recommendation module was shown (null for front-page
impressions — confirmed by its ~97.5% correlation with `scroll_percentage` also being
non-null, and that it's the *clicked* article only 0.5% of the time). MIND has no
equivalent concept. Decided **not** to add a column for it.

**Alternatives rejected:**
- *Add a nullable `context_article_id` column to `impressions`.* Preserves the
  information for a possible future slice (on-article-page vs. front-page impressions),
  but nothing in Q1–Q9 needs it, and it would be null for 100% of MIND rows and ~70% of
  EB-NeRD rows.

**Why:** Nothing in the assignment's required retrieval, evaluation, or slicing logic
uses "which page was this recommendation shown on." Keeps the schema at D3's decided
size. Found while building `ingest_ebnerd.py` — not part of the original D3 column list
because this distinction wasn't discovered until inspecting real EB-NeRD data directly.

---

### D7 — Carve the test partition from dev/validation's tail, not train's
**Date:** 2026-08-22 · **Decided by:** Chaitanya

**Chosen:** Neither dataset provides three labeled partitions at demo/small scale
(verified: MIND-small is train Nov 9–14 + dev Nov 15 only; EB-NeRD-demo is train
May 18–25 + validation May 25–Jun 1 only — both large-bundle test sets exist but are
unlabeled, Codabench-submission-only). Train stays fully intact; dev/validation gets
split into val + test.

**Alternatives rejected:**
- *Carve test from train's tail instead.* Leakage-safety is identical either way — the
  choice doesn't affect correctness, only which partition shrinks. Shrinks train (the
  larger, more valuable-for-indexing partition) and requires re-splitting the file we'd
  otherwise leave untouched.
- *Skip a local test split entirely*, using validation for both tuning and final
  reporting. Doesn't satisfy Q1.3's literal "train/val/test" requirement, and repeatedly
  looking at the same validation numbers while tuning is a soft form of leakage through
  human decisions, not the model's.

**Why:** Maximizes train's data (BM25 index / retrieval quality depends on it directly),
touches only one file per dataset.

### D8 — 70/30 row-count split for the val/test cutoff, uniform across datasets
**Date:** 2026-08-22 · **Decided by:** Chaitanya

**Chosen:** Sort the dev/validation partition by timestamp; the earliest 70% of rows
become val, the latest 30% become test — same rule for both datasets.

**Alternatives rejected:**
- *50/50 split.* Leaves validation smaller relative to test than is typical — validation
  gets used repeatedly during tuning, test only once, so it usually deserves the larger
  share.
- *Day-count based per dataset* (e.g. "last 2 of 7 days" for EB-NeRD, matching the
  assignment's literal example phrasing). Doesn't apply evenly: MIND's dev is a single
  day, so it would still fall back to a percentile split anyway — two methods to justify
  instead of one.

**Why:** One rule, adapts automatically regardless of whether the partition spans 1 day
or 7. Verified against real data: MIND lands at exactly 70.0% val, EB-NeRD at 70.0% val,
and the core invariant (`train_max < val_min < val_max < test_min`) holds strictly for
both — this same check is the seed of the Q9 leakage test.

**Related finding, not a separate decision:** EB-NeRD's `train/history.parquet` and
`validation/history.parquet` genuinely differ per user (verified: user `95507`'s history
length is 370 in train's file, 326 in validation's — not the same data re-served). Since
val and test are both carved from the *same* validation file, both reuse that file's
history table unchanged (`temporal_split.add_history_split`) — the only snapshot
available at that granularity, and reusing it doesn't leak anything since it predates
the entire validation window, val and test both included.

---

### D9 — Feature store: 3 combined files, not per-dataset or per-split
**Date:** 2026-08-22 · **Decided by:** Chaitanya

**Chosen:** `data/processed/{articles,impressions,history}.parquet` — both datasets
combined per table, filtered downstream by the existing `dataset` column;
impressions/history already carry the `split` column from Q1.3.

**Alternatives rejected:**
- *6 files, one per dataset per table.* Keeps datasets fully separate on disk, but Q2/Q3
  retrieval is built per-dataset either way (a `.filter()` on a combined file is cheap),
  so this buys nothing beyond more paths to manage.
- *Partitioned by split too* (train/val/test as separate files). A real efficiency win
  at EB-NeRD-large's 12M+ row scale, but premature now — everything fits comfortably in
  memory at demo scale, and `pl.scan_parquet` + `.filter()` gets most of the same benefit
  without extra files. Revisit in Phase 5 if the large bundle needs it.

**Why:** Matches what Q1.3 already empirically verified concatenates cleanly. Simplest,
matches Q1.4's "small store" wording.

---

### D10 — No auto-download in the one-command rebuild; check-and-guide instead
**Date:** 2026-08-22 · **Decided by:** Chaitanya

**Chosen:** `scripts/build_pipeline.py` does not download anything. It reads
`configs/mind.yaml`/`configs/ebnerd.yaml` for raw-data locations, checks the expected
files exist, and — if not — prints exact manual download instructions and exits (code 1)
rather than crashing partway through ingestion with a confusing traceback.

**Alternatives rejected:**
- *Auto-download EB-NeRD (ungated), check-and-guide for MIND (gated).* Automates what
  can genuinely be automated. Rejected for simplicity — partial automation (one dataset
  auto-downloads, one doesn't) is arguably more confusing to explain than "download is a
  separate manual step for both."
- *Fully automate both via a required HF token.* Most complete, but needs a
  `huggingface_hub` dependency and auth handling that goes beyond `scripts/`'s "thin
  entry point, no logic" design (the original Phase 0 layout decision) — that logic
  would need its own home in `src/newsrec/`.

**Why:** Simplicity, and MIND's gating means full automation was never achievable for
both datasets uniformly anyway. The check-and-guide behavior still means a missing-data
run fails with an actionable message, not a bare `FileNotFoundError` three function calls
deep inside `ingest_mind.py`.

---

## Phase 2 — Q2, BM25 lexical retrieval

Measured facts these five decisions were made against (read off `data/processed/`, not
assumed):

| | MIND | EB-NeRD |
|---|---|---|
| Articles | 65,238 | 11,777 |
| title+abstract length, median / mean / p95 / max | 36 / 45.8 / 89 / 489 | 25 / 25.4 / 43 / 95 |
| Abstract missing or blank | 5.2% | 6.8% |
| History length, median / p95 / max | 15 / 85 / 444 | 91 / 529 / 1,000 |
| Cold-start users (val) | 1,407 / 50,000 | 0 |
| Val impressions / unique users | 51,205 / 37,777 | 17,749 / 1,437 |
| Ground-truth clicks reachable in corpus | 100% | 100% |
| Ground-truth clicks already in user's history | 0.19% | 0.47% |

---

### D11 — Tokenisation: lowercase + Unicode word split, no stopwords, no stemming
**Date:** 2026-08-24 · **Decided by:** Chaitanya

**Chosen:** One tokeniser for both datasets — lowercase, then split on anything that
isn't a Unicode letter or digit. No stopword list, no stemmer. A `max_df` cutoff (drop
terms appearing in more than X% of documents) exists as a config knob, unused by default.

**Alternatives rejected:**
- *Per-language stopword removal* (English for MIND, Danish for EB-NeRD). Barely changes
  rankings — inverse document frequency already reduces *the* to ≈0.1 against *inflation*'s
  ≈8 — so its real benefit is shrinking the largest posting lists, i.e. speed. Costs two
  language-specific code paths and a divergence between the datasets that would have to be
  justified in Q3.5's cross-dataset comparison. `max_df` gets the same speed benefit
  without a word list.
- *Stopwords + stemming* (Porter for English, Snowball for Danish). Genuinely helps a
  morphologically richer language like Danish, but then the two datasets are processed by
  measurably different algorithms, which weakens every cross-dataset claim in the design
  note.

**Why:** One algorithm across both datasets keeps Q3.5's comparison honest, and IDF
already performs most of what stopword removal would.

**Adversarial requirement recorded up front (R10):** the obvious regex `[a-z0-9]+`
silently destroys Danish — `"Rådden kørsel på blå plader"` becomes
`r dden k rsel p bl plader`, five corrupted tokens with no error raised. The pattern must
be Unicode-aware, and a test asserts this exact string tokenises correctly.

---

### D12 — Query = titles of the last N clicked articles, N = 10
**Date:** 2026-08-24 · **Decided by:** Chaitanya

**Chosen:** Build each user's BM25 query by concatenating the **titles** of their **10
most recent** clicked articles. N is a config parameter, not a literal.

**Alternatives rejected:**
- *Entire history.* Uses all evidence, but EB-NeRD's median history is 91 articles
  (max 1,000) — a ~900-token query averaging weeks of interests. BM25 contains no time
  term of any kind, so topic drift cannot be corrected by tuning; it has to be handled in
  query construction or not at all. News relevance decays in hours, which makes this the
  worst domain for it.
- *Last N titles **and** abstracts.* More signal per clicked article, but abstracts are
  ~4× longer than titles, so the drift problem returns at N=10 instead of N=91.

**Why:** Recency is the only defence available against drift, since the algorithm has no
notion of time. N=10 covers most of a MIND user's history (median 15) while cutting
EB-NeRD's long tail hard. If Phase 2 stays inside its 4h budget, sweep N ∈ {5, 10, 20,
all} on val — nearly free, since the index is built once and only the query changes.

**Assumption flagged, not hidden:** "last N" needs chronological order. EB-NeRD supplies
`history_timestamps` so its history can be sorted properly. **MIND does not** — it gives a
bare space-separated list, documented as click-ordered but unverifiable from the data
itself. For MIND, "last 10" means "last 10 in file order, trusting the documentation."

---

### D13 — `k₁` = 1.5, `b` = 0.75 for headline numbers; tune `b` only if time allows
**Date:** 2026-08-24 · **Decided by:** Chaitanya

**Chosen:** Standard defaults for the reported results. If Phase 2 is inside budget,
sweep `b ∈ {0.3, 0.5, 0.75, 1.0}` on val — 4 runs, `k₁` left fixed.

**Alternatives rejected:**
- *Full grid* `k₁ ∈ {0.9,1.2,1.5,2.0}` × `b ∈ {0.3,0.5,0.75,1.0}` — 16 runs. Better
  coverage, but 4× the runtime for the parameter that moves least.

**Why:** Our documents are short (median 25–36 tokens) whereas `b = 0.75` was tuned on
TREC news articles of several hundred words, so length normalisation is the parameter
most likely to be wrong at our document size. `k₁` is far more stable across corpora. If
only one parameter can be tuned, tune `b`.

---

### D14 — Implement the inverted index ourselves on `scipy.sparse`, not via a library
**Date:** 2026-08-24 · **Decided by:** Chaitanya

**Chosen:** Our own implementation. The document-term sparse matrix **is** the inverted
index — in compressed-sparse-column form it stores, per term, the list of documents
containing it. Scoring a batch of queries is one sparse matrix product.

**Alternatives rejected:**
- *`rank_bm25`.* Three lines to use, but it loops over every document per query in pure
  Python — not actually an inverted index.
  **Measured 2026-08-26, and this justification was overstated — see `SCALE_NOTES.md`.**
  The original claim was that 37,777 queries × 65,238 documents "would not finish". It
  does: **2,183.84 ms per query (p50) against our 10.10 ms — 216× — i.e. 30.3 hours
  against 8.4 minutes.** Prohibitive, but finite. The correct argument is that
  `rank_bm25` *builds 2.6× faster* (0.61 s vs 1.61 s) by deferring all work to query
  time, and our extra 1.00 s of build cost is repaid after **0.46 queries** — it moves
  work to the wrong side of a boundary crossed 50,000 times. Decision unchanged;
  reasoning corrected.
- *`bm25s`.* Fast and genuinely sparse, but Q2.1 grades *"build an inverted index"*, and a
  library call neither demonstrates that nor leaves anything defensible in a viva.

**Why:** It's the difference between having built the thing and having called it, on a
sub-requirement that names the data structure explicitly.

**Costs accepted:** one new dependency (`scipy`), and the query side must be batched —
37,777 queries × 65,238 documents as one dense score matrix is ~10 GB against 7 GB of
RAM. That batching constraint is itself a `SCALE_NOTES.md` entry for Q6's "where it
breaks at 10×".

---

### D15 — Exclude articles the user has already read from their retrieved candidates
**Date:** 2026-08-24 · **Decided by:** Chaitanya

**Chosen:** Remove the user's own history articles from their candidate set before
taking top-K.

**Alternatives rejected:**
- *Keep them and report recall as-is.* Simpler and needs no justification, but the query
  is built **from those articles' own titles**, so they match themselves near-perfectly
  and occupy the top of every result list — spending top-K slots on articles the user
  demonstrably already read.

**Why:** Cost measured before deciding, not assumed: only **0.19% (MIND) / 0.47%
(EB-NeRD)** of val ground-truth clicks are articles already in that user's history. So
exclusion removes at most half a percent of achievable recall while freeing slots that
would otherwise be near-guaranteed self-matches. The exact ceiling this imposes is
reported alongside the recall numbers rather than quietly absorbed.

---

### D16 — Query-term repetition counts linearly (raw query term frequency)
**Date:** 2026-08-24 · **Decided by:** Chaitanya

A fork that only surfaced while building the document side, so it was raised mid-step
rather than picked quietly (R6). The stored weights handle `f(t,D)` — repetition inside
the *document*. But a query built from 10 concatenated titles can also repeat a term: if
*tariff* appears in 8 of the user's last 10 clicked titles, how much should that count?

**Chosen:** raw query term frequency — 8 occurrences contribute 8×. The standard textbook
formulation of BM25 as a sum over query term *occurrences*. Plus a **binary-query
ablation** on val, since the index is built once and only the query vector changes, which
makes the comparison nearly free.

**Alternatives rejected:**
- *Binary query terms* (each distinct term counts once). Immune to any single word
  dominating, but discards the strongest signal available — a topic the user returned to
  repeatedly this week. Kept as the ablation rather than the default.
- *Saturated query frequency with a third parameter `k₃`* — `qtf·(k₃+1)/(k₃+qtf)`,
  Robertson's full BM25, designed for exactly this long-query case. Most principled, but
  our queries are only ~10 titles so the runaway-dominance case it protects against
  cannot really arise, and it costs a third hyperparameter to justify with no budget to
  tune it.

**Why:** the risk raw counts carry is one word swamping the query; with 10 titles the
counts top out around 8–10 and IDF already flattens common words (measured: `the` has
IDF 0.26 against a corpus maximum of 10.68 — a 41× spread). The ablation converts
"repetition matters" from an assertion into a measured number for the design note.

---

### D17 — Cold-start users: excluded from the headline, reported alongside
**Date:** 2026-08-25 · **Decided by:** Chaitanya

A user with empty history produces an empty query, so BM25 scores every article 0 and
there is no ranking to take a top-K from. Affects **1,556 MIND val impressions (3.0%),
2,440 ground-truth clicks (3.1%)** — and **zero EB-NeRD impressions**, so however this is
handled lands asymmetrically across the two datasets and feeds Q3.5's comparison.

**Chosen:** headline recall@K over impressions where a query exists, with the
all-impressions number (query-less users counted as misses) printed on the same line.

**Alternatives rejected:**
- *Include them scoring 0 recall as the only number.* Honest end-to-end, but penalises
  BM25 for something it structurally cannot do and hard-caps MIND's recall at 96.9%.
  Retained as the second number rather than discarded.
- *Give them a popularity fallback.* What real systems do, but it blends two retrieval
  systems into one number reported as "BM25 recall@K" — the conflation Q9 exists to
  catch. The popularity baseline is needed for Q4 anyway, so cold-start can be revisited
  then with it already built.

---

### D18 — Macro (per-impression) averaging for recall@K, micro reported alongside
**Date:** 2026-08-25 · **Decided by:** Chaitanya

**Chosen:** mean of per-impression recalls. Micro (pooled hits / pooled clicks) printed
as a second column.

**Alternatives rejected:**
- *Micro as the headline.* Weights a 21-click impression 21× a single-click one. Matters
  on MIND (29% of val impressions carry >1 click, max 21) and is nearly a no-op on
  EB-NeRD (99.5% single-click).

**Why:** unit consistency. Q4's metrics are per impression, its bootstrap resamples
impressions, and both leaderboards score per impression — so Q2, Q3 and Q4 numbers can be
laid side by side without a footnote saying one of them counts something different.

---

### D19 — Also report retrieval restricted to articles already in circulation
**Date:** 2026-08-25 · **Decided by:** Chaitanya

**Chosen:** keep whole-corpus BM25 as the Q2 headline **and** add a second run where, for
each impression at time T, candidates are restricted to articles whose first appearance
in the impression log is strictly before T. Buckets of 1 hour.

**Alternatives rejected:**
- *Whole corpus only.* Fully satisfies Q2's four sub-requirements, and costs nothing
  more. Loses the strongest observation available for Q6.
- *Make the restricted variant the headline.* Q2.3 says "retrieve top-K candidate
  articles using BM25 scoring"; the unrestricted run is the literal answer, and nothing
  should rest on an extension.

**Why:** the spec asks for it three times — *"rapid news decay makes temporal splitting
and freshness handling instructive"*, Q1.4's *"user features (click history, recency)"*,
and the behavioural axis's *"recency/decay"*. Q6 wants observations from experiments.

**BM25 itself is unmodified** — same formula, same `k₁`/`b`, same index, same queries.
Only the candidate pool changes. Adding a recency term *into* the scoring function and
still calling the result BM25 would have been the actual violation.

**Leak-safety, which is the real risk rather than permission:** the filter asks only
`first_seen < T` — strictly before, so an article first appearing in *this* impression is
excluded (hence the measured ceiling of 92.8% EB-NeRD / 99.8% MIND); it is computed from
`candidate_article_ids` (what was *shown*), never from clicks; and "was this already in
circulation" is knowable at serving time. `tests/test_no_leakage.py` must assert the
strict inequality so a later "fix" to `<=` cannot silently import the future.

**Cost this creates for Phase 3:** Q3.5 compares lexical vs semantic. That comparison is
only meaningful if both retrieve from the *same* pool, so embeddings must be run under
both pools too. Charged to Phase 3's budget.

**Measured outcome — the reason this was worth doing.** Restricting the pool raises
absolute recall and raises random-chance recall at the same time:

| Dataset | Pool | recall@200 | random@200 | lift |
|---|---|---|---|---|
| MIND | whole corpus | 2.05% | 0.31% | **6.7×** |
| MIND | available | 3.95% | 0.94% | **4.2×** |
| EB-NeRD | whole corpus | 2.45% | 1.70% | **1.4×** |
| EB-NeRD | available | 7.27% | 6.81% | **1.07×** |

EB-NeRD's 2.45% → 7.27% looks like a 3× win and is almost entirely the pool shrinking
from 11,777 to ~2,963. Reported without the baseline column it would have been a
materially misleading result. Within a fresh Danish news cycle, BM25's lexical similarity
to reading history is worth about 7% over choosing at random; MIND retains a real 4.2×
because its available pool is ~21,000 articles, leaving far more for content matching to
discriminate between.

---

**Judgment call made without a decision point** (R6 trivia exception — changes runtime,
not results): the query depends only on the user's history, which is fixed per user
within a split, so scoring runs **once per unique user** and joins back to impressions
rather than once per impression. Identical numbers; 26% less work on MIND and 12× less on
EB-NeRD (1,437 users behind 17,749 impressions). This does **not** hold for D19's
availability runs — what is available changes with time even though the query does not —
so those score once per (user, hour-bucket) task instead.

---

## Phase 3 — Q3, semantic retrieval (embeddings)

### D20 — Compute our own embeddings with one multilingual model, via `sentence-transformers` on CPU-only PyTorch
**Date:** 2026-08-25 · **Decided by:** Chaitanya

**Chosen:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — 384
dimensions, 12 layers, 250,037-token vocabulary, `model_type: "bert"` — run through
`sentence-transformers==6.0.0` on `torch==2.13.0+cpu`. **The same model for both
datasets.**

**Alternatives rejected:**
- *Load provided embeddings instead of computing.* Retired on 2026-08-25 as a
  **false premise** carried since Phase 0. Neither dataset ships loadable article-text
  vectors: MIND ships `entity_embedding.vec` (26,904 TransE *knowledge-graph entity*
  vectors keyed by Wikidata IDs, not articles), and EB-NeRD's `articles.parquet` has no
  embedding column at all — its provided vectors are separate downloads the spec itself
  marks *"(optional)"*. Mixing routes across datasets would make Q3.5's cross-dataset
  comparison uninterpretable, the same objection that killed per-language stemming in D11.
- *`paraphrase-multilingual-mpnet-base-v2`* — `model_type: "xlm-roberta"`, 768-dim.
  Matches Q3's second named family literally and is likely somewhat stronger, but doubles
  vector storage and score-matrix cost and is roughly 3× slower on CPU. Kept as a
  **one-string swap** if the MiniLM result looks weak; not paid for up front.
- *`distiluse-base-multilingual-cased-v1`* — the smallest, fastest-looking multilingual
  option on the sentence-transformers menu. **Rejected on a verified fact: its language
  list is `ar, zh, nl, en, fr, de, it, ko, pl, pt, ru, es, tr` — 13 languages, no `da`.**
  It would have embedded Danish without erroring, returning well-formed 512-dim vectors of
  confident nonsense. See "silent degradation" below.
- *ONNX Runtime with the int8-quantised export, no PyTorch* (~150 MB total vs ~860 MB).
  Genuinely attractive and was the initial recommendation, on the belief that PyTorch was
  an unaffordable download. **Measurement killed that argument** (see below). What
  remained was ~40 lines of hand-written tokenise → mask → mean-pool → normalise code
  whose most likely bug — omitting the attention mask, so padding tokens are averaged in —
  is silent and hits *short* documents hardest, which is exactly our corpus. Not a risk
  worth taking two days from the deadline.
- *TF-IDF + truncated SVD, no neural model at all.* Zero install. Deviates from Q3's
  "BERT/XLM-RoBERTa" wording and learns no cross-lingual meaning. Emergency parachute only.

**Why:** one model across both datasets keeps Q3.5's comparison a comparison of *methods*
rather than of *setups* — the D11 principle carried forward. `model_type: "bert"` means
the chosen model satisfies Q3's wording literally, not by analogy.

**On using a library here, when D14 refused one for BM25.** Not an inconsistency. Q2.1
grades *"build an inverted index"* — the data structure **is** the deliverable, so
`rank_bm25` would have answered nothing. Q3 names pretrained models and libraries as the
expected route (*"compute your own using BERT/XLM-RoBERTa"*, *"e.g., FAISS, ScaNN, or
brute-force"*). Of Q3's five sub-requirements the library supplies only the text→vector
step; the user representation (Q3.3), D15's exclusion, both candidate pools, recall@K and
Q3.5's comparison all remain ours.

**Decision input that had to be measured, not assumed.** The route was initially chosen
against a remembered figure of ~60 KB/s download speed (extrapolated from `scipy` taking
ten minutes days earlier). Measured on the day: **PyPI 6.1 MB/s, HuggingFace 2.8 MB/s** —
about 100× faster. That single number inverted the recommendation from ONNX to
sentence-transformers, because it removed the only argument ONNX was winning on. Recorded
because it is a clean instance of a decision resting on a stale measurement that nobody
had re-checked.

**Measured install cost (2026-08-25):**

| Route | Wheels | Model | Total |
|---|---|---|---|
| Naive `pip install sentence-transformers` | **2,894 MB** (54 pkgs) | ~470 MB | ~3.4 GB |
| **Chosen:** torch from the CPU-only index first | ~390 MB (torch 155 MB) | ~470 MB | **~860 MB** |
| ONNX Runtime, int8 | 31 MB | 118 MB | ~150 MB |

**2,238 MB of the naive route's 2,894 MB are 18 `nvidia-*`/`cuda-*` packages** —
`nvidia-cublas` alone is 543 MB — downloaded and installed on a machine with no GPU and
never loaded. `requirements.txt` therefore carries `--index-url
https://download.pytorch.org/whl/cpu` **above** the `torch==2.13.0+cpu` pin, with
`--extra-index-url https://pypi.org/simple` for everything else. Verified by a
from-scratch resolution: 46 packages, **0 CUDA**, torch served by PyTorch's CDN and the
other 45 by PyPI. Delete those two lines and Q1.5's one-command rebuild silently pulls
2.9 GB on the next machine.

**Costs accepted:** `.venv` grows to 1.6 GB (against 901 GB free — not a constraint);
inference is CPU-only, so embedding all 77,015 articles is a measured cost still to be
established, not an assumption.

---

### D21 — Exact brute-force nearest-neighbour search, not an approximate index
**Date:** 2026-08-25 · **Decision:** Selected based on empirical benchmark evaluation.
rather than silently taken

Q3.2 asks for an ANN (Approximate Nearest Neighbour) index and explicitly permits
*"FAISS, ScaNN, **or brute-force for small scale**"*. We take brute force: one dense
matrix product `(users × 384) @ (384 × articles)`, batched, then top-K per row.

**Why:** it is *exact*, so unlike an approximate index it cannot cost us recall — which
matters because recall@K is the number Q3.4 reports and Q3.5 compares. At our scale the
cost is under a minute of linear algebra — **measured 2026-08-26 at 1.73 ms per query (p50), i.e. 65 s for all 37,777 val users, and 5.8× faster per query than the sparse BM25 index**. The single real constraint is memory: a
37,777 × 65,238 float32 score matrix is **9.9 GB against 7 GB of RAM**, so it must be
batched — the *identical* constraint D14 already forced on BM25, so `bm25_search.py`'s
batching pattern is reused rather than reinvented.

**Rejected:** a real FAISS index. Defensible and closer to the sub-requirement's first
example, but costs an extra dependency and install time for a recall *loss*. The
speed-versus-recall trade-off it represents is recorded in `SCALE_NOTES.md` as a Q6
"where it breaks at 10×" observation instead — which is where that discussion is actually
graded.

---

### D22 — Embeddings live in their own `embeddings.parquet`, one vector per row beside its article_id
**Date:** 2026-08-25 · **Decided by:** Chaitanya

This is the sub-decision D3 deferred in Phase 1 ("where article embeddings live — as
columns on `articles` or a separate store"), finally answerable now that we know what the
vectors actually are.

**Chosen:** a fourth file, `data/processed/embeddings.parquet`, with columns
`(dataset, article_id, embedding: list[f32][384])` — one row per article, **each vector
in the same row as its own id**.

**Alternatives rejected:**
- *A list column on `articles.parquet`.* Keeps D9's three-file store exactly as designed.
  Rejected on a concrete cost: `build_pipeline.py` **rewrites** `articles.parquet` on
  every run, so vectors there would either be recomputed each rebuild (**2.81 s → ~13
  min, a 280× regression on the Q1.5 requirement we already built and tested**) or
  silently destroyed.
- *`embeddings.npy` + a separate id index.* The format retrieval actually wants — a native
  contiguous float32 matrix, memory-mappable so 110 MB need never be resident. Rejected
  because two artifacts can drift: one article added to the store without re-embedding
  misaligns every vector after it, and the only defence is an id-order assertion someone
  has to remember to write and keep. Chosen format makes that failure **impossible by
  construction** rather than caught by vigilance.

**Why:** alignment safety over raw load speed — and the cost of that choice turned out to
be negligible when measured rather than estimated. The feared conversion overhead
(Parquet list column → contiguous matrix, via `explode()` + `reshape()`) is **0.6 seconds
for all 77,015 vectors**, against the 9.9 GB score matrix we already batch. Keeping the
~13-minute embedding step out of the 2.81 s rebuild is worth far more.

**Stated defaults, not forks** (taken under the (b) pacing agreement, recorded so they are
not silent):
- **Document text = title + abstract**, mirroring BM25's document side (D11/Q2.1), and
  **user vector = last 10 clicked titles**, mirroring D12 — so Q3.5 varies the algorithm
  and nothing else.
- **float32, not float16.** Saves 59 MB against 901 GB free; NumPy's float16 matmul
  typically upcasts internally so it is not even faster, and precision in a 384-term dot
  product is ranking resolution.
- **Truncation at 128 subword tokens accepted, not worked around.** Measured: **7.7% of
  MIND articles exceed the limit but only 1.37% of MIND's total tokens are lost, and
  0.00% of EB-NeRD's** (max 121 tokens, not one of 11,777 clipped). Danish expands more
  per word (1.68 vs 1.55 subword tokens per whitespace word) but its articles are short
  enough that it never binds. A real dataset asymmetry for the Q3.5 write-up, an order of
  magnitude too small to justify chunk-and-average.

**Verified on the real artifact, not just designed:** 77,015 vectors built in 13.3 min at
96 articles/s, 110 MB on disk, loading in 0.6 s as a C-contiguous float32 `(77015, 384)`
matrix; norms min 1.000000 / max 1.000000; and the id order **matches
`articles.parquet` exactly**. Nearest-neighbour inspection is topically coherent in both
languages (a Democratic-primary article retrieves five more Trump-vs-Democrats polling
stories; a Danish ice-hockey story retrieves Danish athlete-career stories).

**Observation banked for Q3.5:** EB-NeRD's nearest-neighbour cosines run materially lower
than MIND's (0.48–0.52 vs 0.62–0.69). Danish tabloid headlines are short and deliberately
cryptic — `'LIVE: Nej, nej, nej'` carries almost no topical signal — which is the
*semantic* echo of the same dataset asymmetry BM25 exposed lexically.

**Every article is its own nearest neighbour at cosine exactly 1.000** (verified). Since
the mean-pooled user vector is provably the point of maximum average similarity to the
user's own history, D15's history exclusion binds **more** tightly here than it did for
BM25, and must be applied *before* top-K rather than after.

---

## Q3.5 — lexical versus semantic: the measured comparison

Not a decision; the answer Q3.5 asks for. Feeds the design note directly.

Both methods were run through the *same* harness under the *same* constraints — D12's
N = 10, D15's history exclusion, D17's cold-start policy, D18's macro headline, D19's two
candidate pools — so the only thing varying is the matching function. `run_semantic_recall.py`
is deliberately the same shape as `run_bm25_recall.py`, and `availability.py` was extracted
so both call one implementation rather than two copies that could drift.

**recall@200, macro, has-query slice, val:**

| Dataset | Pool | BM25 | Semantic | Δ | BM25 lift | Semantic lift |
|---|---|---|---|---|---|---|
| MIND | whole corpus | 2.05% | **2.17%** | +6% | 6.69× | **7.08×** |
| MIND | available | 3.95% | **5.41%** | **+37%** | 4.19× | **5.74×** |
| EB-NeRD | whole corpus | 2.45% | **2.65%** | +8% | 1.44× | **1.56×** |
| EB-NeRD | available | 7.27% | **8.57%** | +18% | 1.07× | **1.26×** |

**Finding 1 — semantic wins at depth, and BM25's advantage decays with K.**

| | K=50 | K=100 | K=200 |
|---|---|---|---|
| MIND available, BM25 | 7.81× | 5.73× | 4.19× |
| MIND available, semantic | 7.93× | 6.89× | **5.74×** |
| EB-NeRD available, BM25 | 1.21× | 1.18× | **1.07×** |
| EB-NeRD available, semantic | 1.27× | 1.28× | **1.26×** |

On EB-NeRD's in-circulation pool BM25 is essentially at chance by K=200 (1.07×) while
semantic holds 1.26×. Mechanism: BM25 ranks only articles sharing a term with the query
and runs out of signal below the top few dozen; cosine similarity ranks the *entire*
corpus meaningfully, so its tail degrades more slowly. This is the argument for using the
two as complementary candidate generators rather than choosing between them.

**Finding 2 — the one slice where lexical wins.** EB-NeRD, whole corpus, K=50: BM25 0.77%
(raw) / 0.81% (binary) against semantic's 0.63%. Shallow retrieval over a large pool of
short Danish headlines favours exact term overlap. It is the only one of twelve
configurations where BM25 leads, and Q3.5 asks specifically "on which slices?".

**Finding 3 — content-based retrieval is blind to time, and this is not a BM25 property.**
Freshness (first appearance in the impression log on/after the val window start;
`published_time` is null for 100% of MIND so first-seen is the proxy):

| | corpus | semantic top-200 | actual clicks |
|---|---|---|---|
| MIND | 10.9% | 12.3% | 13.7% |
| EB-NeRD | 33.5% | **31.9%** | **93.5%** |

Semantic's top-200 reproduces the corpus's own freshness profile while 93.5% of real
EB-NeRD clicks are fresh — the same shape Q2 found for BM25. Neither term overlap nor
cosine similarity contains a time term, so **neither can be tuned toward freshness**; it
must be handled outside the scoring function (D19's pool restriction) or not at all.
Phase 2 could only claim this about BM25; running a completely different matching function
and getting the same profile generalises it to content-based retrieval as such.

MIND's clicks are only mildly fresh-skewed (13.7% against a 10.9% baseline), which is why
pool restriction helps it far less than it helps EB-NeRD — and is a real dataset
difference to report rather than a wrinkle to smooth over.

**Finding 4 — mean pooling's characteristic failure, observed not hypothesised.** MIND
user `mind:U13132` read three political stories and one about a Starbucks gingerbread
latte. Semantic retrieval returned **five Popeyes chicken-sandwich articles** and nothing
political. One click in ten hijacked the whole recommendation.

The mechanism is sharper than "the mean lands between the topics". MIND's fast-food
articles form a *dense* cluster of near-duplicates; political articles are spread out.
Nearest-neighbour search is won by dense regions, so a mean sitting closer to politics
overall still finds more neighbours inside the tight food cluster. **BM25 does not have
this failure** — a bag of words from ten titles still matches political vocabulary
strongly. This is the concrete cost of representing a user as a single point, and the
argument for either multi-vector user representations or interest clustering, neither of
which is in scope here.

**Finding 5 — the dataset asymmetry is a property of the data, not the method.** MIND lifts
run 4–8× and EB-NeRD's 1.1–1.6× for *both* methods. EB-NeRD's available pool is small
(~2,478–3,385 articles per hour) and its clicks are overwhelmingly fresh, so there is
little room for content matching to discriminate. MIND's available pool is ~21,000, leaving
far more for either method to work with.

---

_Further decisions appended as they are made._

---

## Phase 4 — Q4, offline evaluation harness (+ Q9 folded in)

### Forced consequence, not a decision: Q4's metrics grade a *re-ranking*, not a retrieval
**Date:** 2026-08-25

Recorded here because it looks like a fork and is not one. AUC, MRR and nDCG each need a
per-item clicked/not-clicked label. Whole-corpus retrieval (Q2/Q3) produces top-200 lists
against 65,238 articles of which only the clicked handful are labelled — the other ~65,000
were never *shown*, so they are unlabelled, not negative. Treating them as negatives would
invent ~65,000 facts per impression that the log never recorded.

So the harness scores the platform's own supplied candidate list — MIND's `impressions`
field, EB-NeRD's `article_ids_inview` — re-ranked by the same two scoring functions Q2 and
Q3 already built. Q4.5's *"run your evaluation harness on both BM25 and embedding-based
retrieval results"* means the same scorers, a different candidate set. This is also exactly
what both Codabench leaderboards consume, so Phase 4 feeds Phase 5 directly rather than
being a detour. It is the `system design contract` §6 retrieval-vs-re-ranking trap, landing here.

**Measured facts the harness was designed against** (val split, real store):

| | MIND | EB-NeRD |
|---|---|---|
| Val impressions | 51,205 | 17,749 |
| Candidates per impression — min / median / p95 / max | 2 / 22 / 119 / 295 | 5 / 9 / 30 / 90 |
| Impressions with >1 click | 29.3% | 0.5% |
| Impressions with 0 clicks **or** all-clicked | 0.0% | 0.0% |

Two consequences carried into the code:
- **AUC is always defined on our val data** (no impression is all-positive or
  all-negative), but that is a property of this split, not of the code — so the degenerate
  case is handled explicitly rather than relied upon. MIND has 2,744 val impressions with
  only 2 candidates, where AUC is exactly 0 or 1.
- **nDCG@10 is not a top-k metric on EB-NeRD.** Median rack 9 < cutoff 10, so it collapses
  into nDCG@all — a full-list ordering measure, close to what AUC already reports. On MIND
  (median 22, p95 119) the cutoff genuinely bites. Same metric, doing real work on one
  dataset and near-duplicate work on the other. Reported with the reason named, not
  smoothed over.

---

### D23 — Break ranking ties pessimistically, and report the optimistic bound alongside
**Date:** 2026-08-25 · **Decided by:** Chaitanya

AUC defines its own tie rule (a tied pair scores 0.5). **MRR and nDCG do not** — they need
a total order, so a tie must be broken by something outside the score. BM25 scores an
article exactly 0 when it shares no term with the query, so ties are real, not theoretical.

**Chosen:** clicked candidates are ordered **last** within their tie group, making every
MRR and nDCG figure a **lower bound** no tie-luck can inflate. The optimistic bound
(clicked first) is computed alongside, so the gap between the two *measures* how much tie
handling matters instead of asserting it doesn't.

**Alternatives rejected:**
- *Stable sort (leave ties alone).* Simplest and fully deterministic. Rejected because
  "leave them alone" is not neutral — `np.argsort` is stable, so it silently means "rank
  ties by position in the raw candidate list". That is safe only if raw order carries no
  click signal, which we verified rather than assumed (below) — but verified about *these
  two val splits*, and nothing in the code would notice a test bundle behaving differently.
- *Seeded random tiebreak.* Robust to any hidden ordering signal and reproducible, but
  the reported number shifts with the seed, and a single number still cannot show its own
  sensitivity.

**Why:** only the two-bound version can produce the sentence the design note actually
needs — *"tie handling moves the metric by less than X, measured, so no conclusion rests
on it."* Cost is roughly five lines.

**Measured before deciding** (BM25 scores over real val candidate lists, 3,000 sampled
impressions per dataset):

| | MIND | EB-NeRD |
|---|---|---|
| Candidates scoring exactly 0 | 2.4% | 4.0% |
| Largest tie group, as a fraction of the rack | 10.0% | 12.7% |
| Impressions where **every** candidate scores 0 | 0.10% | 0.00% |
| Impressions where a clicked article is inside the zero-tie | 2.70% | 4.00% |

Smaller than expected — a 10-title query carries enough vocabulary that most same-cycle
candidates share something. But 0.10% of MIND impressions have their MRR and nDCG decided
*entirely* by the tiebreak.

**The check that made this worth stopping for.** Because stable sorting silently ranks
ties by raw candidate order, we tested whether that order leaks click information:

```
                mean normalised position of a clicked item    clicked-item-first
  MIND     :    0.5017        (0.5 = uniform)                  9.86%  vs 10.18% expected
  EB-NeRD  :    0.4961                                        11.74%  vs 11.70% expected
```

Both platforms pre-shuffle `article_ids_inview` / `impressions`; raw order carries no
click signal. Good news, but *verified* good news about the val splits specifically — which
is precisely why the tie policy is explicit in code rather than inherited from a sort's
stability.

**Ties are exact float equality.** That catches the structurally important group (score
exactly 0.0 from an empty sparse dot product) and not near-ties differing by float
rounding. Known and bounded, with a test pinning the behaviour rather than leaving it to
be discovered.

---

### D24 — Evaluate two baselines alongside BM25 and semantic
**Date:** 2026-08-25 · **Decided by:** Chaitanya

Q4 requires no baseline. AUC needs none — 0.5 is random by construction. But **MRR and
nDCG have no natural zero point**, so "nDCG@5 = 0.28" is uninterpretable on its own. Phase
2 already paid for that lesson: EB-NeRD's recall@200 looked like a 3× win until the
random-baseline column showed the pool had merely shrunk.

**Chosen:** two extra scorers through the same harness — **random** (seeded per impression)
and **popularity** (rank by train-window click count, the standard non-personalised
baseline).

**Alternatives rejected:**
- *Random only.* Cheapest, and gives the floor. Leaves "is this better than just showing
  what's popular?" unanswered, and Q4.3's novelty/coverage needs a popularity signal anyway.
- *No baseline.* Exactly Q4.5's literal ask. Every MRR and nDCG in the design note would
  then sit without a reference point.

**Why:** popularity is needed later regardless (Q4.3's beyond-accuracy metrics, and as a
natural arm of Q9's ablation), so the marginal cost was ~25 lines. It also carries a real
risk we accepted deliberately: the content scorers might barely beat popularity, which
would be the honest headline rather than a result to bury.

**Leakage constraint built into the code, not just intended:** popularity is counted over
the **train** split only. `train_click_counts` raises on any other split rather than
trusting the caller, because counting val clicks would mean the baseline had seen the
answers it is scored against.

---

### Q4.2 — measured results, and the finding that came out of it

`scripts/run_rerank_eval.py`, val split, macro mean over impressions with a query (D17),
pessimistic tie policy (D23):

| Dataset | Method | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| MIND | random | 0.5007 | 0.2489 | 0.2264 | 0.2906 |
| MIND | popularity | 0.5423 | 0.2541 | 0.2278 | 0.2892 |
| MIND | BM25 | 0.5492 | 0.3006 | 0.2760 | 0.3377 |
| MIND | **semantic** | **0.6338** | **0.3475** | **0.3316** | **0.3912** |
| EB-NeRD | popularity | **0.4647** | 0.1436 | 0.0939 | 0.2382 |
| EB-NeRD | BM25 | 0.4966 | 0.3128 | 0.3418 | 0.4297 |
| EB-NeRD | random | 0.4987 | 0.3126 | 0.3443 | 0.4295 |
| EB-NeRD | **semantic** | **0.5331** | **0.3373** | **0.3730** | **0.4532** |

The random arm lands at AUC 0.5007 (MIND) and 0.4987 (EB-NeRD), which is the harness
checking itself: anything else would have meant a bug in the metrics, the scoring or the
labels before any finding could be read off the other rows.

**Finding 6 — BM25 cannot re-rank EB-NeRD at all.** AUC 0.4966 against random's 0.4987;
MRR 0.3128 against 0.3126. Verified as a property of the data, not a bug: the scorer was
checked against an independent per-impression computation (max deviation 1.9e-06, float32
noise), and the mean BM25 score of clicked candidates is 6.835 against 6.773 for unclicked
— no separation.

This is *not* a restatement of Phase 2's weak EB-NeRD retrieval. Retrieval and re-ranking
are different jobs on different pools: BM25 held a 1.21× lift at K=50 when picking from
11,777 articles, and is at exactly chance when ordering the ~9 the platform already chose.
The reason is that **the platform's own recommender has already spent the easy signal.**
Everything in `article_ids_inview` is plausible for that user; lexical overlap with their
reading history no longer separates the plausible from the clicked. Semantic retains a
little discrimination (0.5331) but not much.

**Finding 7 — on EB-NeRD, yesterday's popularity predicts today's clicks in the *wrong*
direction.** Popularity scores AUC **0.4647**, meaningfully *below* chance. The mechanism:

| | val candidates never clicked in train | mean train popularity, clicked vs unclicked |
|---|---|---|
| MIND | 46.6% | 34.45 vs 31.97 → mildly predictive (AUC 0.5423) |
| EB-NeRD | **86.9%** | **1.96 vs 3.22 → anti-predictive (AUC 0.4647)** |

EB-NeRD's article inventory turns over almost completely between the train and val
windows, and an article still carrying train-window clicks by the val window is
disproportionately *stale*. So training-window popularity is a freshness signal pointing
backwards.

This is the **third independent route** to the same conclusion. Phase 2 found content
retrieval blind to time via BM25's freshness profile; Phase 3 reproduced it with a
completely different matching function; Phase 4 now finds that the one baseline which
*does* encode time encodes it with the sign reversed. On MIND, whose clicks are only
mildly fresh-skewed (13.7% against a 10.9% corpus baseline), the same baseline is weakly
useful. Same algorithm, opposite sign, explained by a measured property of the data.

**Finding 8 — the tie policy earned its keep exactly once, and the number says where.**
The gap between the pessimistic and optimistic bounds on nDCG@10:

| | BM25 | semantic | popularity | random |
|---|---|---|---|---|
| MIND | +0.0013 | +0.0000 | +0.0768 | +0.0000 |
| EB-NeRD | +0.0007 | +0.0000 | **+0.5012** | +0.0000 |

For BM25 and semantic the two bounds agree to under 0.002, so **no conclusion in this
table rests on tie handling** — the sentence D23 existed to make possible. For popularity
it is decisive: with 86.9% of EB-NeRD's candidates never clicked in train, that scorer is
mostly a constant, and its reported nDCG@10 swings from 0.238 to 0.739 purely on how ties
are resolved. Reported as a single optimistic number it would have looked like the best
method on the table.

---

### Deliberately not carried into re-ranking: D15 and D19

**D15 (history exclusion)** cannot apply — the platform chose the candidate list, and every
item on it needs a score. Nor does the re-ranker adjust for already-read candidates, even
though they are a real signal:

| | MIND | EB-NeRD |
|---|---|---|
| Candidates already in the user's history | 0.055% | 0.959% |
| Click rate on those | **14.3%** | **4.1%** |
| Click rate on all candidates | 4.1% | 8.4% |
| Ratio | **3.5× more likely** | **0.49× — half as likely** |

The two datasets point in opposite directions, both well outside sampling noise. Checked
for leakage before believing the MIND figure, since "already read *and* clicked" is the
shape a double-counted click would take: no history snapshot contains clicks from its own
window (MIND 0.578% train / 0.195% val; EB-NeRD 0.358% / 0.470%), so the 3.5× is behaviour.

Folding it into the score would make the number reported as "BM25 nDCG@5" two systems
blended — the conflation D17 rejected a popularity fallback for. **"Has the user already
read this candidate" is a serving-time feature**, so it belongs in Q9's ablation, where it
is graded rather than smuggled in.

**D19 (availability)** does not apply either: the candidates were in circulation by
definition — the platform showed them. Restricting further would re-decide a decision the
log already records.

**Landmine found while checking the above, for `tests/test_no_leakage.py`:** EB-NeRD's
**validation history file contains 99.52% of the train-window clicks** (22,143 of 22,249).
Harmless as we use it — val history predicts val impressions, and train precedes val — but
pointing val-split history at train impressions would hand over the answers almost
perfectly. A concrete assertion the test should carry, and one we would not have thought
to write without the measurement.

---

### D25 — Beyond-accuracy measured on retrieval output, with both diversity bases reported
**Date:** 2026-08-25 · **Decided by:** Chaitanya

Two coupled forks, flagged in `PROGRESS.md` two sessions before they were reached.

**Fork A — which output do these metrics describe?** Beyond-accuracy describes a *system's
own* output. In re-ranking the platform chose the items and we only reordered them, and
only **6.8% of MIND's corpus (19.2% of EB-NeRD's) ever appears in any val candidate list**
— so coverage measured there is capped by someone else's recommender.

**Chosen:** the Q2/Q3 retrieval top-K is the headline; the re-ranking numbers are computed
too, to *show* that cap rather than assert it. Accuracy metrics stay on re-ranking, since
that is the only place labels exist. Different metrics on different outputs, stated openly.

**Rejected:** *re-ranking only* (the literal Q4.5 reading) — it would publish a coverage
figure that measures MSN's recommender; *retrieval only* — cheaper, but then the cap is a
claim rather than a measurement.

**Fork B — what does `distance(i, j)` mean in intra-list diversity?** The conventional
choice is cosine distance between article embeddings. But semantic retrieval finds
articles by *maximising* cosine similarity, so grading it in that space measures it by the
quantity it exists to minimise: it loses by construction, and the design note would report
a tautology as a finding.

**Chosen:** report **both** bases — embedding cosine distance and category distance
(0 if same category, 1 if different; 18 categories on MIND, 25 on EB-NeRD, zero nulls in
both). The disagreement between them is the diagnosis.

**Rejected:** *embedding only, with a caveat in prose* — the conventional citable metric,
but a caveat has to carry the entire result and is weaker than a measurement; *category
only* — fair to every method, but too coarse to separate two different chicken-sandwich
stories, which is the exact failure mode from Finding 4.

**Why (both forks):** the same argument that made D23's two-bound tie policy worth five
lines — a single number cannot show its own sensitivity.

---

### Q4.3 — measured results

`scripts/run_beyond_accuracy.py`, val, K = 10. **Read against the Q4.2 accuracy table,
never alone: the random arm scores best on all three metrics**, so these price what a
method gave up rather than ranking methods.

| Dataset | Output | Method | ILD-embed | ILD-cat | Novelty | Coverage |
|---|---|---|---|---|---|---|
| MIND | retrieval | bm25 | 0.7169 | 0.5892 | 17.718 | 51.02% |
| MIND | retrieval | semantic | 0.5618 | 0.5001 | 17.724 | **53.21%** |
| MIND | retrieval | popularity | 0.9003 | 0.8444 | 6.861 | **0.015%** |
| MIND | retrieval | random | 0.9381 | 0.8009 | 17.882 | 99.96% |
| MIND | re-rank | bm25 / semantic / popularity / random | 0.93 / 0.89 / 0.94 / 0.94 | 0.86 / 0.84 / 0.89 / 0.89 | 16.0 / 15.8 / 13.8 / 16.0 | **4.26 / 4.11 / 2.24 / 4.35%** |
| EB-NeRD | retrieval | bm25 | 0.7482 | 0.6207 | 14.468 | 31.13% |
| EB-NeRD | retrieval | semantic | 0.4072 | **0.6464** | 14.776 | 14.44% |
| EB-NeRD | retrieval | popularity | 0.7948 | 0.7853 | 8.444 | **0.25%** |
| EB-NeRD | retrieval | random | 0.8646 | 0.8571 | 14.829 | 73.69% |
| EB-NeRD | re-rank | bm25 / semantic / popularity / random | 0.85 / 0.83 / 0.85 / 0.85 | 0.80 / 0.79 / 0.80 / 0.79 | 14.7 / 14.7 / 14.5 / 14.7 | **17.87 / 17.75 / 18.17 / 17.92%** |

Novelty ranges: MIND 6.13–18.20, EB-NeRD 8.01–15.16. Retrieval list counts are 48,593 of
50,000 MIND users (the 1,407 cold-start users retrieve nothing, so their diversity is
undefined, not zero) and 1,562 of 1,562 on EB-NeRD.

**Finding 9 — the embedding basis overstates semantic's diversity deficit on both datasets,
and on EB-NeRD it reverses the sign.**

| semantic ÷ bm25 | embedding basis | category basis |
|---|---|---|
| MIND | 0.784 (−21.6%) | 0.849 (−15.1%) |
| EB-NeRD | **0.544 (−45.6%)** | **1.041 (+4.1%)** |

Reported on the conventional metric alone, the design note would have said "semantic
retrieval produces markedly less diverse lists" — true-sounding, and on EB-NeRD *false in
direction*. Category diversity, which owes nothing to the embedding space, says semantic's
EB-NeRD lists span topics slightly better than BM25's. The honest statement is the pair:
**semantic's lists are tightly packed in embedding space while covering a comparable
spread of categories** — it repeats *within* topics rather than collapsing to one, which
is exactly what Finding 4's five Popeyes articles look like.

The two datasets differ, and that difference is real rather than noise: on MIND semantic
is less diverse on both bases, on EB-NeRD only on one. Reporting one basis would have
hidden that a method's diversity behaviour is dataset-dependent.

**Finding 10 — coverage is where the methods actually separate, and they separate by 3
orders of magnitude.** On MIND retrieval: popularity **0.015%** (exactly **10** distinct
articles served to all 50,000 users), semantic 53.21%, bm25 51.02%, random 99.96%. Popularity is the
dead-stock warehouse quantified — an entire recommender whose whole catalogue is thirteen
articles. It also scores novelty 6.861 against a corpus floor of 6.13, confirming by a
second route that it is serving nothing but the head.

The direction flips between datasets: on MIND semantic covers slightly *more* than BM25
(53.21% vs 51.02%), on EB-NeRD less than half as much (14.44% vs 31.13%). So semantic's
accuracy win is bought with corpus coverage on EB-NeRD and essentially free on MIND.

**Finding 11 — novelty barely discriminates between content methods, and the reason is
structural.** MIND: bm25 17.718, semantic 17.724, random 17.882, against a ceiling of
18.20. EB-NeRD: 14.468 / 14.776 / 14.829 against 15.16. All three sit near the ceiling
because **88.2% of MIND's corpus was never clicked during training**, so almost anything
retrieved is "novel" by self-information. Only popularity — which explicitly targets the
head — separates at all.

That does not make the metric worthless: it is positive evidence that **neither BM25 nor
semantic carries a popularity bias**, which is a claim worth having rather than assuming.
But it cannot rank the content methods, and reporting a 0.006 difference between BM25 and
semantic as if it meant something would be over-reading a long tail.

**Finding 12 — Fork A demonstrated rather than argued.** On the re-ranking rows, four
completely different scorers land within 0.4 percentage points of each other on EB-NeRD
coverage (17.75–18.17%) and within 0.02 on category diversity. On the retrieval rows the
same four methods span **0.015% to 99.96%**. Re-ranking beyond-accuracy measures the
platform's candidate generator, not the scorer sitting behind it — which is why the
headline is measured on retrieval.

---

### D26 — Slice definitions: absolute cold-start threshold, and two head/tail definitions
**Date:** 2026-08-25 · **Decided by:** Chaitanya

**Cold start — chosen:** absolute threshold, `history_len <= 5`. MIND 17.7% of val
impressions, EB-NeRD 0.3% (55 impressions).

**Rejected:** *zero history only* (`has_query`) — 3.0% / 0.0%, matches D17 exactly but
the spec says "few clicks", not "no clicks"; *per-dataset quantile* (bottom 25%) — equal
slice sizes and comparable intervals, but "cold" would then mean ≤8 articles on MIND and
≤94 on EB-NeRD, two different concepts under one label.

**Why:** cold start is an *absolute* property — we know little about this user — not a
relative rank. The datasets barely overlap on history length:

| history length per val impression | p0 | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|---|
| MIND | 0 | 3 | 8 | 19 | 42 | 77 |
| EB-NeRD | **5** | 37 | 94 | 225 | 400 | 604 |

EB-NeRD's *coldest* user has more history than MIND's 25th percentile. That EB-NeRD has
essentially no cold-start users is a finding; a quantile slice would have manufactured a
comparison that hid it. The cost is accepted and reported: EB-NeRD's cold slice (n = 55)
carries intervals so wide that every method overlaps every other.

**Head vs tail — chosen:** both definitions, because the textbook one is degenerate here.

| definition | head size | val clicks landing in head |
|---|---|---|
| train-popularity (top 50% of *training* clicks) | MIND 186 / EB-NeRD 233 articles | **2.1% on both** |
| exposure (articles filling 50% of val impression slots) | MIND 74 / EB-NeRD 232 | MIND 67.3% / EB-NeRD 54.9% |

88.2% (MIND) / 90.5% (EB-NeRD) of the corpus was never clicked during training, and
39.7% / 94.4% of val clicked articles were never clicked in training either — the
catalogue turns over between windows, so training popularity barely predicts later clicks.
Exposure splits both datasets usefully and, critically, **is counted from
`candidate_article_ids` — what was shown — never from clicks.** Counting exposure from
clicks would produce a similar-looking ranking and make the slice circular: impressions
grouped by a quantity derived from their own labels.

**Stated default, not a fork** (Phase 3 pacing agreement): an impression whose clicks
straddle the head/tail boundary is assigned to *neither* slice and counted separately,
rather than resolved by an invented majority rule. On EB-NeRD that is 20 impressions.

**Caveat that must travel with the train-popularity slice.** Popularity scores AUC
**0.9407** on EB-NeRD's `head-trainpop` slice against 0.4647 overall. That is not a
finding — the slice is *defined* by train popularity and the method *scores* by train
popularity, so the method is being graded on the quantity that selected the slice. Same
family of error as measuring semantic's diversity in its own embedding space (D25's Fork
B). It is reported with this caveat attached, never as evidence that popularity works.

---

### D27 — Coverage is reported without a bootstrap confidence interval
**Date:** 2026-08-25 · **Decided by:** Chaitanya

Found by running Q4.4's bootstrap and noticing every coverage interval sat **entirely
below its own point estimate** (e.g. EB-NeRD BM25 retrieval: point 31.13%, percentile
interval [24.45%, 25.68%]).

**Root cause, and it is the statistic rather than the code.** Drawing n items with
replacement from n items yields only **63.2%** distinct items — measured at n = 1,562,
17,749 and 50,000, all 63.2% (the 1 − 1/e result). For a **mean** that is harmless: a
duplicated impression still contributes its value, so the mean scatters around the truth
unbiased. For **coverage** it is fatal: coverage is the size of a *union*, a duplicate
contributes nothing new, so every resample can only lose articles. The whole bootstrap
distribution shifts down and the percentile interval inherits the shift. This is why AUC,
MRR, nDCG, diversity and novelty intervals are all sound and only coverage is not.

**Chosen:** report coverage's point estimate with no interval, and explain why. Both the
raw resample spread (`coverage_resample_lo/hi_BIASED`) and the pivotal correction
(`coverage_pivotal_lo/hi`) are kept in the CSV under explicit names.

**Alternatives rejected:**
- *Basic (pivotal) interval as the headline* — `[2·point − q97.5, 2·point − q2.5]`,
  the textbook remedy for a location-shifted bootstrap, and it satisfies Q4.4's "CI for
  each metric" literally. Rejected because both quantiles sit *below* the point, so the
  corrected interval lands entirely *above* it (BM25 retrieval: [36.57%, 37.81%]). That is
  defensible as an estimate of population coverage but it still excludes the reported
  number, so it does not remove the need for the explanation — it adds a second thing to
  defend.
- *Rarefaction: coverage at a fixed subsample size m < n drawn without replacement.*
  Statistically the most correct treatment, and it would make the sample-size dependence
  explicit. Rejected on time, and because it changes the headline number so the whole Q4.3
  table would need re-reading two days from the deadline.

**Why:** what we report is closer to a **census** than an estimate. "Our system surfaced
31.1% of the catalogue to these 50,000 users" was counted, not estimated — there is no
sampling error to state. The other seven metrics are per-impression averages where "what
if we had drawn different impressions?" is a real question.

**Test debt this exposed, recorded because it is the more useful lesson.**
`test_resampled_coverage_never_exceeds_the_full_coverage` already asserted this exact
downward shift and treated it as correct behaviour. The assertion was mathematically true;
what was missing was the inference that it invalidates the interval. **A test can pin a
real property and still let a wrong conclusion through** — the test is now renamed and
documents *why* the numbers cannot be reported as a CI, alongside a test pinning the 63.2%
fact directly.

---

## Q4.3 slices + Q4.4 bootstrap intervals — the measured results

`scripts/run_eval_report.py`, val, re-ranking, pessimistic ties, 1,000 resamples,
percentile intervals. Every metric in a cell shares one resample draw.

**Finding 13 — the confidence intervals convert two earlier observations into claims.**

EB-NeRD, all impressions with a query (n = 17,749), AUC:

| method | AUC | 95% CI | verdict |
|---|---|---|---|
| random | 0.4987 | [0.4935, 0.5038] | — |
| **bm25** | **0.4966** | **[0.4917, 0.5009]** | **interval contains 0.5: cannot reject chance** |
| popularity | 0.4647 | [0.4629, 0.4666] | entirely below 0.5: worse than random, not noise |
| semantic | 0.5331 | [0.5286, 0.5379] | excludes 0.5 and excludes BM25's interval |

Finding 6 ("BM25 cannot re-rank EB-NeRD") was an observation about two similar numbers;
it is now a statistical statement. Finding 7 (popularity below chance) likewise.

**Finding 14 — the method ranking *flips* between slices, which is the whole point of
slicing.** MIND, cold (history ≤ 5, n = 7,522) versus warm (n = 42,127), AUC:

| method | cold | warm |
|---|---|---|
| popularity | **0.5542 [0.5477, 0.5609]** | 0.5402 [0.5378, 0.5427] |
| bm25 | 0.5296 [0.5220, 0.5366] | **0.5526 [0.5496, 0.5554]** |
| semantic | 0.6160 [0.6084, 0.6234] | 0.6370 [0.6343, 0.6397] |

**On cold-start users, popularity beats BM25** — and the intervals do not overlap, so it
is not noise. On warm users the order reverses, also with non-overlapping intervals. The
mechanism is straightforward once seen: with ≤5 articles of history BM25's query is short
and noisy, so a generic "what is everyone reading" ranking is genuinely better than a
personalised one built from almost no evidence. Popularity is the *only* method that
scores **higher** on cold users than warm ones.

This is also the retrospective justification for D17's rejected popularity fallback: the
fallback would have been the right *product* decision and would still have been the wrong
*measurement* decision, because it would have hidden this crossover inside a single
blended number.

**Finding 15 — semantic degrades least under cold start.** From warm to cold, BM25 loses
0.0230 AUC and semantic 0.0210, while popularity *gains* 0.0140. Semantic stays clearly
ahead in both slices. Ten embedded titles carry more usable signal than ten titles' worth
of bag-of-words, and the gap is widest exactly where evidence is scarcest.

**Finding 16 — everything gets harder on rarely-shown articles, but the ranking survives.**
MIND, exposure head (n = 32,060) vs tail (n = 10,617):

| method | MRR head | MRR tail | AUC head | AUC tail |
|---|---|---|---|---|
| random | 0.2958 | 0.1478 | 0.5004 | 0.4996 |
| bm25 | 0.3391 | 0.2060 | 0.5453 | **0.5561** |
| semantic | 0.3963 | 0.2360 | **0.6397** | 0.6222 |

MRR halves on the tail for *every* method including random — that is the slice being
structurally harder (the clicked article is one the platform barely promoted), not a
method failing. AUC, which normalises for that, barely moves; BM25 even improves. Reading
the MRR drop as "our system is bad at tail articles" would have been the mistake, and
having both metrics on the same slice is what prevents it.

On EB-NeRD the same slice pair shows semantic doing **better** on the tail
(0.5406 [0.5337, 0.5474]) than the head (0.5269 [0.5208, 0.5327]), intervals
non-overlapping — content matching earns more where the platform's own promotion is doing
less work, which is Finding 6's mechanism seen from the other side.

**Finding 17 — a slice defined by a quantity a method scores on grades that method
tautologically.** On the train-popularity head slice, popularity scores AUC **0.9737**
[0.9697, 0.9779] on MIND and 0.9407 on EB-NeRD, against 0.5423 / 0.4647 overall. The slice
is *defined as* "impressions whose clicked article is train-popular" and the method
*ranks by* train popularity, so the slice selects exactly the impressions the method
cannot miss.

This is the same shape as measuring semantic's diversity in the embedding space it
optimises (D25's Fork B). Two independent instances in one phase makes it a rule worth
stating in the design note: **whenever a slice and a method share a definition, that
method's score on that slice is close to a tautology.** Reported with the caveat attached,
never as evidence popularity works.

**Slice sizes, including what the slices exclude:**

| | MIND | EB-NeRD |
|---|---|---|
| impressions with a query | 49,649 | 17,749 |
| cold (≤5) / warm | 7,522 / 42,127 | 55 / 17,694 |
| exposure head / tail / **mixed-click excluded** | 32,060 / 10,617 / **7,209** | 9,735 / 7,994 / **20** |
| train-pop head / tail / mixed-click excluded | 574 / 48,085 / 1,019 | 376 / 17,371 / 2 |

MIND's exposure slice excludes **7,209 impressions (14.5%)** whose clicks straddle the
boundary — a direct consequence of 29.3% of MIND impressions carrying multiple clicks, and
a real cost of refusing to invent a majority rule. EB-NeRD, at 99.5% single-click, loses 20.

---

## Q9 — anti-gaming: the serving-time ablation and the leakage test

### D28 — Ablation design: add an unavailable feature and price it
**Date:** 2026-08-25 · **Decision:** Selected based on empirical benchmark evaluation.
rather than silently taken

Q9 asks to *"report metrics with and without features unavailable at serving time"*.
The direction matters and is easy to invert: the exercise is **not** to strip legitimate
features out, it is to **add an illegitimate one and measure how much it inflates the
score**, so the design note can say what a leak is *worth* here instead of asserting that
leaks are bad.

The test applied for "available at serving time" is: *could this number have been computed
at the moment the recommendation was made?* — not "is it in the file", since the file
contains the whole logged history including things that had not happened yet.

Five arms, run through the Q4 harness unchanged:

| arm | serving time | feature |
|---|---|---|
| popularity (train) | safe | click counts from the training window |
| popularity (FUTURE) | **leak** | click counts from the window being evaluated |
| semantic | safe | the Q3 scorer — our best honest system |
| semantic + seen-before | safe | plus "has this user already read this candidate" |
| semantic + FUTURE pop | **leak** | plus val-window click counts |

The two popularity arms are the cleanest comparison available: identical algorithm,
identical candidates, differing **only** in whether the clicks being counted had happened
yet. Blending is rank-based (each score vector converted to within-impression ranks in
[0, 1] before combining), because BM25 runs 0–40, cosine runs [−1, 1] and click counts run
0–4,316 — added directly, the widest scale would decide the ranking regardless of signal.
Blend weight is a **stated default of 0.5, not tuned**; tuning it would make this a
system-design exercise rather than an ablation.

`future_click_counts` is quarantined in `ablation.py`, and a test asserts **no other file
in the package references it**. Its safe counterpart `rerank.train_click_counts` raises on
any non-train split; a test pins that asymmetry so a well-meaning "consistency" edit
cannot disarm either side.

### Q9 measured results (val, macro, has-query slice, 1,000-resample 95% CIs)

| Dataset | arm | AUC | Δ vs semantic |
|---|---|---|---|
| MIND | popularity (train) · safe | 0.5423 [0.5400, 0.5446] | −0.0915 |
| MIND | **popularity (FUTURE) · LEAK** | **0.6102 [0.6076, 0.6127]** | −0.0236 |
| MIND | semantic · safe | 0.6338 [0.6312, 0.6364] | — |
| MIND | semantic + seen-before · safe | 0.6339 [0.6312, 0.6365] | **+0.0001** |
| MIND | **semantic + FUTURE pop · LEAK** | **0.6572 [0.6546, 0.6597]** | **+0.0234** |
| EB-NeRD | popularity (train) · safe | 0.4647 [0.4628, 0.4665] | −0.0684 |
| EB-NeRD | **popularity (FUTURE) · LEAK** | **0.6657 [0.6613, 0.6696]** | **+0.1326** |
| EB-NeRD | semantic · safe | 0.5331 [0.5283, 0.5379] | — |
| EB-NeRD | semantic + seen-before · safe | 0.5314 [0.5266, 0.5361] | **−0.0017** |
| EB-NeRD | **semantic + FUTURE pop · LEAK** | 0.5872 [0.5826, 0.5916] | +0.0541 |

**Finding 18 — moving one counting window is worth more than the entire honest system.**
On EB-NeRD, popularity counted over the *evaluated* window scores AUC **0.6657** against
**0.4647** for the identical algorithm counted over training: **+0.2010** from a one-line
difference. Our best honest system, semantic, scores 0.5331 — so the leaked baseline beats
it by 0.1326, while semantic beats random by only 0.0344. **On EB-NeRD a single leaked
feature is worth roughly four times the entire honest modelling effort of Phases 2–4.**

That is the anti-gaming argument in one number. It also explains why leaderboard rank is
a poor grading signal, and why the assignment says so.

**Finding 19 — a leak is worth most exactly where honest methods are weakest.** The same
leak buys +0.2010 AUC on EB-NeRD and +0.0679 on MIND — a 3× difference in the value of
identical cheating. The mechanism is the freshness property found in Phases 2, 3 and 4:
86.9% of EB-NeRD's val candidates were never clicked during training and 93.5% of its
clicks are on fresh articles, so *nothing legitimate predicts them* and the future feature
is the only thing that can. On MIND, training popularity already carries real signal
(0.5423, interval clear of 0.5), so its future version adds proportionally less.

The uncomfortable corollary is worth stating plainly in the design note: **the datasets
where leakage is most tempting are precisely the ones where it is hardest to notice**,
because there is no strong honest baseline whose absence would look suspicious.

**Finding 20 — the legitimate feature buys essentially nothing, and its sign flips.**
"Has this user already read this candidate" is real signal — such a candidate is clicked
3.5× more often than average on MIND and 0.49× on EB-NeRD. Added to semantic it moves AUC
by **+0.0001 (MIND)** and **−0.0017 (EB-NeRD)**, both inside the confidence intervals.

Two reasons, both worth reporting:
1. **It is too rare to matter in aggregate.** Only 0.055% (MIND) / 0.959% (EB-NeRD) of
   candidates are already-read, so a large per-candidate effect on a tiny slice is a
   negligible effect on the mean.
2. **The blend applies one positive weight to a feature whose sign is opposite on the two
   datasets.** On EB-NeRD already-read candidates are clicked *half* as often, so boosting
   them is actively wrong there — and the arm faithfully reports the small loss rather
   than being quietly re-specified per dataset. A single global weight is wrong for at
   least one of the two datasets, which is itself the finding.

Reported as-is: this is what happens when a plausible-looking feature is added without
checking its direction on each dataset, and it is a far more useful thing to have measured
than a tuned number would have been.

### `tests/test_no_leakage.py` — what it asserts, and what would catch what

Twelve tests across five groups. Leakage is the failure mode this project is least able to
detect by inspection, because **its only symptom is that the numbers get better** — the one
outcome nobody investigates. So it is asserted rather than observed.

| group | asserts |
|---|---|
| **Temporal** | the availability predicate is `first_seen < T`, strictly — an article first appearing *in the impression being predicted* is not available to it; availability is monotone in time; `first_seen` is the earliest appearance, not the latest |
| **Label-free** | availability ignores clicks entirely (constructed so a clicks-derived implementation gives a *different* answer, not merely a slower one); slice exposure is counted from candidates; `train_click_counts` raises on any non-train split |
| **Split boundary** | `train_max < val_min` and `val_max < test_min` on the real store, for both datasets |
| **History snapshot** | no history snapshot contains clicks from its own window (observed 0.195–0.578%, asserted < 5%) |
| **Scorers are blind** | flipping every label leaves BM25 and semantic scores **byte-identical** |

**Mutation-verified**, because a leakage test that cannot fail is worse than none — it
provides false assurance. Five deliberate leaks were reintroduced and **all five were
caught**, including the one that matters most: relaxing `first_seen < T` to `<=`.

**The landmine test.** One test deliberately performs the mis-join that would leak —
pairing EB-NeRD's **val**-split history with **train** impressions — and asserts that the
overlap check fires (>90%; measured 99.52%, 22,143 of 22,249 clicks). It is written as a
live demonstration rather than a comment because `history.parquet` holds all three
snapshots keyed by `split`, 1,217 EB-NeRD users have all three rows, and a join on
`user_id` alone attaches the wrong one **with no error at all**. If that test ever stops
firing, the protection has silently gone.

---

## Phase 5 — Q5, scale-up and Codabench submission

### D29 — Run the large-scale submission locally; Kaggle named as the fallback
**Date:** 2026-08-25 · **Decided by:** Chaitanya

This is the sub-decision D2 deferred in Phase 1 ("which cloud platform, to be decided
against measured memory numbers rather than guesses"). The measured numbers arrived, and
they pointed away from cloud entirely.

**Chosen:** run everything on the local machine — WSL2, 7.7 GB RAM, no GPU. Kaggle is
named as the fallback if the prediction path exhausts memory, rather than adopted
pre-emptively.

**Alternatives rejected:**
- *Kaggle notebook for everything.* 30 GB RAM and a free T4 GPU, which would cut the
  embedding step from ~43 min to ~3 min. Rejected on the arithmetic: rebuilding the
  environment (our `requirements.txt` pins CPU-only PyTorch, which is wrong there),
  git-to-notebook friction for a repo that is deliberately scripts-and-configs (D1), the
  20 GB working-directory limit, session timeouts, and re-downloading ~5 GB — realistically
  45–75 min of setup to save ~40 min of compute.
- *Hybrid: embed on Kaggle, predict locally.* Uses the GPU only where it helps, and D22's
  standalone `embeddings.parquet` makes the round-trip structurally possible. Rejected as
  a second environment to keep in sync for a 40-minute saving.
- *Google Colab.* Rejected on reliability rather than capability: ~12.7 GB RAM, aggressive
  idle disconnects, non-guaranteed GPU allocation. Two days from the deadline, a
  disconnect mid-run is a worse failure mode than a slow run that finishes.

**Why — the three facts that decided it, each measured on the day rather than assumed:**

1. **The data was already local.** 1.5 GB `MINDlarge_test`, 1.8 GB `ebnerd_testset`,
   3.4 GB `ebnerd_large`, downloaded and extracted. "Download the large test sets early —
   they are several GB each" was the single largest item on Phase 5's risk list, and it
   was already paid.
2. **Only one step in the whole pipeline benefits from a GPU** — embedding 246,502 test
   articles. Everything else (Polars reads, sparse BM25 products, cosine scoring,
   writing predictions) is CPU work where a free-tier cloud machine is comparable or
   slower.
3. **The streaming path is comfortable here.** Computing the distinct candidate set over
   all **13,536,710** EB-NeRD test impressions took **1 second at 1.03 GB peak** with
   Polars' streaming engine. The 13.5 M-row scale that motivated cloud is only a problem
   for code that materialises it, which `SCALE_NOTES.md` had already established we must
   not do.

**The honest constraint, recorded rather than smoothed over:** of the 7.7 GB of RAM,
~4.5 GB is held by development tools and IDE processes, leaving ~2.5–3 GB for the run. That
does not remove the chunking requirement — it *is* the chunking requirement. Cloud's extra
RAM would have permitted sloppiness, not removed the need.

**Measured cost of the choice, and a correction to an R8 flag raised mid-run.** The
embedding of both test corpora completed in **24.4 min (MIND) + 18.6 min (EB-NeRD) =
43.0 min — exactly the ~43 min `SCALE_NOTES.md` predicted.** A flag was raised partway
through claiming it was heading for ~80 min; that projection came from sampling throughput
(~50 articles/s) while development work competed for the same cores, against 82–115
articles/s unloaded. The estimate was right and the alarm was wrong. Recorded because **an
over-run flag raised on a contended measurement is itself a measurement error**, and it is
the same failure mode as D20's stale download-speed figure.

---

### D30 — The leaderboard test set is a separate store, and its label column is absent rather than empty
**Date:** 2026-08-25 · **Decision:** Selected based on empirical benchmark evaluation.
stated rather than silently taken

Two coupled choices about where the Codabench test data lives and what shape it has.

**Chosen:** the test bundles are read by `src/newsrec/submission.py` into
`data/processed/submission/`, never into the feature store; and the resulting impressions
frame has **no `clicked_article_ids` column at all**.

**Why separate storage:** the word "test" is already taken. D7's local test split is
carved from the tail of MINDsmall_dev / EB-NeRD-validation and **has labels**; it is what
every script written in Phases 1–4 means by `split == "test"`. Pouring 15.9 M unlabeled
leaderboard impressions into `impressions.parquet` under that name would silently
redefine the split for all of them. A name collision that changes results without
erroring is precisely the failure class R10 exists for.

**Why the label column is absent, not empty:** D3 specifies an empty list for unlabeled
rows, which is right for the feature store where labeled and unlabeled rows share a table.
Here the entire split is unlabeled. An all-empty label column invites the exact mistake Q9
was written about — code that computes a metric over it and reports a number rather than
refusing. `metrics.py` returns NaN for an undefined AUC and would survive this, but
nothing guarantees every future caller does. An absent column raises immediately.

**Rejected:** *reuse `ingest_*.load_behaviors` unchanged.* It cannot work for EB-NeRD
(no `article_ids_clicked` column — raises), and for MIND it works only **by coincidence**:
the `-[01]$` strip becomes a no-op and the `ends_with("-1")` filter yields empty lists.
Correct today, unchecked, and silently wrong the day a bundle ships differently.
`assert_mind_test_unlabeled` therefore scans every row and refuses a labeled file, and a
test pins that the reader actually calls it — a gap found by mutation testing, where
deleting the call left all fifteen tests passing.

**One optimisation that touches correctness, so it is tested rather than trusted:**
`load_submission_history` truncates each user's history to its last N **inside the reader**.
This is not tidiness — EB-NeRD's test history is 807,677 users at a median of 83 articles,
about 67 million ids, which `build_user_vectors`' `.to_list()` would materialise as ~67
million Python strings on a machine with ~2.5 GB free. Truncating first leaves at most 8
million. It changes no result because `build_user_vectors` already takes `[-n_recent:]`,
and `test_truncation_changes_no_vector` asserts the two paths produce byte-identical
matrices.

**Inherited assumptions re-verified at test scale rather than carried forward:**

| Assumption | Verified on | Re-verified on | Result |
|---|---|---|---|
| EB-NeRD history is chronological oldest-first | 4,714 demo users | **807,677** test users | 0 out of order |
| MIND's history string is constant per user (D3) | 33,617 train users | **484,059** multi-row test users | 0 violations |
| Every candidate article exists in the corpus | demo/small | 2.37 M + 13.5 M impressions | 0 missing, both datasets |

**Measured properties of the test bundles, banked for the design note:** EB-NeRD's 13.5 M
test impressions draw on only **10,451 distinct candidate articles** out of a 125,541-article
corpus — a startlingly small pool, and a direct echo of Phase 2's finding that EB-NeRD's
in-circulation set is tiny. MIND's 2.37 M impressions draw on 30,043 of 120,961. MIND test
has 702,005 users, 29,108 of them cold-start (1.2%). EB-NeRD has 134 articles with neither
title nor subtitle, which embed to a vector of the empty string — they appear as a
candidate **0 times** and in a user history 171 times out of ~67 million, so they are
inert; counted rather than handled.

---

### Q5 — the submission format, and the one bug it invites

Not a decision; a specification we had to recover, plus the failure mode it sets up. Both
competition pages render as an empty JavaScript shell, so the guidelines were pulled from
Codabench's API (`/api/competitions/{id}/`, whose `pages` field carries the real HTML).

Both leaderboards want the same line format:

```
<impression_id> [r1,r2,...,rn]
```

and differ in exactly one visible detail, which is enough to waste an upload:

| | MIND (13967) | EB-NeRD (2469) |
|---|---|---|
| File inside the zip | `prediction.txt` (**singular**) | `predictions.txt` (**plural**) |
| Zip contents | nothing but that file — no folders, no `__MACOSX` | same |
| Row order | must match the original bundle | same |
| Ranks | integers 1..n, 1 = best | same |

**`r_i` is the rank awarded to the candidate at position `i`** — not the identity of the
article placed at rank `i`. It is the **inverse permutation** of an argsort, not the
argsort. MIND's own worked example: candidates `N125045 N87192 N73556 N20417`, answer
`[4,1,3,2]`, i.e. N87192 (position 2) came first.

This is the sharpest silent-failure in the whole project. `np.argsort(-scores) + 1`
produces a file with the correct line count, the correct row order, and integers 1..n on
every line — it passes every structural check that can be written — and scores
approximately random. The leaderboard returns one number, so there is no signal
distinguishing "our model is weak" from "our ranks are inverted".

Three defences, in increasing order of how much they would have caught:

1. `rank_vector` is four lines, isolated, with the inversion explained in a comment.
2. Its tests are pinned to **both competitions' own published worked examples**, plus a
   derived round-trip check (reconstruct the ranking from the rank vector and compare
   against sorting by score) so a mis-transcribed example cannot validate a wrong
   implementation, plus an explicit assertion that the correct output *differs* from the
   argsort spelling.
3. `scripts/validate_submission.py` re-reads the raw bundle and checks line *i* against
   impression *i*'s own id and candidate count, before anything is uploaded.

**A documentation correction worth recording**, because it is the second instance of the
same failure in this project. Both pages state per-day submission caps — "at most one
submission each day" (MIND), five (EB-NeRD). Those are **live-competition limits**, and
both competitions ended years ago; the current limit is 10/day on both. Caught by
Chaitanya against the page text on 2026-08-25. D20's inverted ONNX-versus-PyTorch
recommendation came from exactly this: a figure that was true when written, believed
without re-checking. The generalisation for the design note is that **stale documentation
fails silently in the direction of over-caution**, which is cheap here and was expensive
in D20.

**What we submit, and why it is not a new model.** The submission scorer is
`score_semantic`, imported unchanged from Q4.2 — semantic re-ranking measured best on
both datasets (MIND AUC 0.6338, EB-NeRD 0.5331), and it is our best *honest* system in
Q9's sense: no serving-time feature, no future-window counting. The submission task **is**
the Q4.2 re-ranking task on a different split, which is why Phase 4 fed Phase 5 directly.
D23's pessimistic tie policy deliberately does not carry over: it works by consulting
labels, which this split does not have and which we must not want.

**Three memory decisions the 13.5 M-row split forced** (all measured, all under D29's
~2.5 GB budget):

1. **Impressions are never materialised.** The frame is sliced in chunks and each chunk's
   lines are appended immediately. Verified cheap rather than assumed: a 50,000-row slice
   at offset **13,000,000** costs **0.28 s**, and at offset 2,000,000 on MIND's CSV
   **0.33 s** — so deep offsets do not re-scan and the loop is O(n), not O(n²).
2. **User vectors are built in batches into an on-disk memmap.** EB-NeRD's 807,677 users
   × 384 float32 is 1.24 GB resident otherwise.
3. **The scoring matrix is restricted to articles that actually appear as a candidate** —
   10,451 of 125,541 for EB-NeRD, 30,043 of 120,961 for MIND. The score block is
   (batch_users × n_candidates), so this is a ~12× cut on the dominant term, and it
   changes no result: an article that is never a candidate can never be scored.

---

### D31 — Re-tune N for re-ranking: 10 → 100, and report the fusion as an ablation rather than shipping it
**Date:** 2026-08-26 · **Decision:** Confirmed on validation split and verified on leaderboard.

Prompted by the first MIND leaderboard result: **AUC 0.6037, rank 62/90**, against a val
AUC of 0.6338. Val proved a trustworthy proxy — close in value and, more importantly,
monotone — so improvements could be measured offline without spending submissions.

**Chosen:** N = 100 for the re-ranking user vector, mean-pooled, on both datasets.
Fusion is measured and reported, not shipped.

**The finding, which is the point rather than the number.** D12 chose N = 10 for
*retrieval*, and gave an explicit reason: a query built from a long history drags a
whole-corpus search toward stale interests, and BM25 contains no time term with which to
correct it. That reasoning **does not transfer to re-ranking**, and we had inherited the
value without re-asking. Re-ranking never searches the corpus — the platform has already
supplied a topically plausible shortlist — so there is no drift to defend against and
more history is simply more evidence.

Measured on val (macro AUC, has-query slice, fixed denominator across all N so the sweep
compares scorers and not populations):

| N | BM25 | semantic mean-pool | semantic max-sim |
|---|---|---|---|
| 5 | 0.5375 | 0.6174 | 0.6093 |
| **10** *(what was submitted)* | 0.5492 | **0.6338** | 0.6219 |
| 25 | 0.5577 | 0.6456 | 0.6311 |
| 50 | 0.5596 | 0.6480 | 0.6336 |
| **100** *(chosen)* | 0.5601 | **0.6489** | 0.6346 |

**+0.0151 AUC from one parameter**, saturating by 100 (50→100 adds 0.0009). EB-NeRD moves
the same way, 0.5331 → 0.5413. This is `system design contract` §6's retrieval-versus-re-ranking trap
appearing as a *hyperparameter* rather than as a candidate set: **the same knob wants
opposite values for the two tasks**, and a value justified by a sound argument in one
phase was silently wrong in the next.

**Two results we did not expect, both kept because they are more informative than the win.**

1. **Max-similarity lost on MIND and won on EB-NeRD.** Scoring a candidate by its
   similarity to the user's single *nearest* history article, rather than to their
   centroid, was motivated by Finding 4's observed mean-pooling failure (the Popeyes
   cluster). It loses at every N on MIND (0.6346 vs 0.6489) and **wins** on EB-NeRD
   (0.5450 vs 0.5413). Consistent with what D22 already measured: EB-NeRD's histories are
   far longer (median 83 vs 15) and its nearest-neighbour cosines much lower (0.48–0.52 vs
   0.62–0.69, short cryptic Danish tabloid headlines), so a centroid over 100 loosely
   related articles carries little, while the nearest one still carries a real interest.
   **A hypothesis drawn from a genuine observed failure still failed to generalise**,
   because that failure was specific to whole-corpus nearest-neighbour search where dense
   clusters win, which cannot arise on a 25-candidate list.
2. **A scorer that is worse alone can still help in combination.** Max-sim is behind
   mean-pooling on MIND by 0.0143 yet adds +0.0056 when fused with it — the errors are
   complementary, not merely smaller.

**Best fusion, measured:** MIND 0.6545 (z-score, mean 1.0 / max 0.5 / bm25 0.25),
EB-NeRD 0.5472 (z-score, mean 1.0 / max 1.0). **Not shipped**, on an explicit
cost argument rather than a doubt about the result: it is +0.0056 against N's +0.0151,
and shipping it needs a BM25 index and queries over 702,005 test users plus max-similarity
across 2.37M impressions — one to two hours against a deadline with the design note
unwritten, for a metric the spec states is never graded.

**A measurement that changed the search, not just the result.** Popularity was dropped
from the fusion after checking whether it *transfers*: only **1,701 of 30,043 (5.7%)** MIND
leaderboard-test candidates have a train-window click count, because our train split is
MIND-small (9–14 Nov 2019) and the test bundle begins 19 Nov. A weight tuned on a signal
absent for 94% of test candidates would have flattered val and done nothing on the
leaderboard. The same catalogue turnover that produced Finding 8 (popularity predicts
EB-NeRD clicks in the *wrong* direction) here invalidates a feature before it is used.

**Process costs, recorded because they were self-inflicted:** the first fusion search ran
over an hour without finishing — it evaluated seven metrics per weight combination when
the search reads one, searched a 4-way grid when AUC's scale-invariance makes one weight
redundant, and printed nothing until completion. Rewritten as `scripts/tune_fusion.py`,
the same search takes ~2 minutes. Three separate runtime estimates given during this work
were wrong; the corrective is that an estimate for a loop nobody has timed is a guess.
### D32 — Wait for the shared Codabench queue; do not self-host a worker unless it stalls
**Date:** 2026-08-26 · **Decided by:** Chaitanya

The EB-NeRD submission (13,536,710 predictions, uploaded and validated) sits unscored
because the RecSys 2024 challenge concluded and the organizers' compute workers were
retired with it. A student surfaced the organizers' own remedy: they publish a Docker
setup and a live `BROKER_URL` for competition 2469's queue, so any participant can attach
a compute worker and drain it.

**Chosen:** wait several days for the shared queue, and set up an x86-64 cloud virtual
machine only if nothing has been scored by then. The deadline moved to 30 August, which is
what makes waiting affordable.

**Why waiting is not merely the lazy option.** The queue is *shared*: a worker attached by
any participant processes whatever job is next, for everyone. At least one other student
is running one. So the probability our submission is scored rises without us spending
anything, and a worker we ran would spend its first hours on someone else's submission
regardless.

**Rejected: self-hosting locally.** Two independent blockers, both found in pre-flight:
this machine is `aarch64` (Windows on ARM) while the worker image is a single-manifest
`amd64` build with no ARM variant, and outbound TCP to `www.codabench.org:5672` is
filtered here while :443 is open. Emulation plus a tunnel would stack two workarounds
under a multi-hour numeric job.

**Rejected for now: a cloud virtual machine.** The GitHub Student Developer Pack includes
$100 of Azure credit with no payment method required; a `Standard_D4s_v3` (4 vCPU / 16 GB)
matches the organizers' `t3.xlarge` at roughly $1 for a 5-hour run. Held in reserve rather
than taken, because the deadline allows it and because a free path may resolve first.
Worker tooling was written for this and later removed from the repository as unrelated to
the pipeline; it survives in git history.

**What this decision is actually worth, stated plainly.** Q5 requires submissions to both
leaderboards; both were made. A scored EB-NeRD result buys one thing: a second test of the
calibration claim (offline val deltas transfer to the leaderboard), currently resting on a
single MIND data point, on the dataset where our own harness says the honest signal is
weakest. Valuable to the design note, graded nowhere. It does not justify displacing the
hand-in.

**If a score does land, it must be disclosed** as having come from a self-hosted worker
following the organizers' documented procedure — a fresh score on a competition whose
workers are switched off otherwise invites exactly the wrong question.

**OUTCOME, 2026-08-27.** Waiting was overtaken by events: a self-hosted worker was built
after all, because the published one turned out to be actively destructive rather than
merely absent. Every released `codalab/competitions-v2-compute-worker` tag is incompatible
with Codabench's live server in three places, and because Celery acknowledges on delivery,
a failing worker **silently removes submissions from the shared queue** — 42 of other
participants' were lost that way before we understood it. A worker built from upstream's
own source on the runtime upstream declares (Python 3.13) works, and scored submission
904082: **AUC 0.5396, rank 147 / 247**.

The decision's *reasoning* survives its reversal. The claim that made waiting affordable —
that nothing in this project depended on leaderboard feedback — is exactly what made the
whole detour optional, and it is what the EB-NeRD number then confirmed: val 0.5413 against
leaderboard 0.5396, agreement to 0.0017. Full diagnosis in `PROGRESS.md`'s error log. The worker
tooling itself has been removed from the repository as unrelated to the assignment; it
remains in git history (see the commits of 2026-08-27).


---

# Assignment 2 — Learning from Click-Logs

**Everything above this line is Assignment 1** (decisions D1–D32), finished and tagged
`phase-5-complete`. It is kept in full because A2 builds directly on it and because the
design note draws on it: D19's availability filter, D23's tie policy, D26's slice
definitions and D31's N=100 are all live in A2's code, not history.

A2 decisions are numbered from **D33** so the two never collide.

## What A2 adds, and why

A1 ended on a measured gap rather than a result. BM25 and embedding retrieval contain no
time term — a structural fact about the formulas, not a tuning failure — while 92.7%
(MIND) / 93.5% (EB-NeRD) of real clicks are on fresh articles. Three independent routes
reached that conclusion, and the third was the sharpest: train-window popularity predicts
EB-NeRD clicks in the *wrong* direction (AUC 0.4647), because 86.9% of validation
candidates were never clicked during training at all.

A2's behavioural features — recency-weighted history, windowed popularity, freshness,
session context — are the direct answer. That is why Q3's required "one principled
improvement" is chosen from A1's own measurements before anything is measured again,
rather than selected after the fact from whatever happened to work. A1's own Finding 5
is the reason this matters: a slice defined by a quantity a method scores on grades that
method tautologically, and choosing an "improvement" after seeing its result is the same
error wearing different clothes.

## Architecture of the A2 system

```
        A1, unchanged                          A2, new
  ┌──────────────────────┐          ┌───────────────────────────┐
  │ BM25 index           │          │ features/                 │
  │ embeddings (384-dim) │ ───────► │   history · article       │
  │ availability filter  │  scores  │   session · unavailable   │
  └──────────────────────┘    as    └───────────────────────────┘
        candidate            features            │
        generation                               ▼
             │                        ┌───────────────────────┐
             └──── top-K, K=100–200 ─►│ rank/gbdt  (LambdaRank)│
                                      │ rank/nrms  (baseline)  │
                                      └───────────────────────┘
                                                 │
                                                 ▼
                                   metrics · slices · paired bootstrap
                                            (A1's, extended)
```

The A1 scorers do not get thrown away and replaced — **they become features** of the
re-ranker, which is what makes Q2's "before and after re-ranking" comparison meaningful
rather than a comparison of two unrelated systems.

## Decision log — A2

_(D33 onward is written as each decision is taken. Nothing decided yet beyond the four
planning-stage choices recorded in `PROGRESS.md`: clone-forward repo layout, NRMS
reimplemented in PyTorch on CPU, EB-NeRD subsampled from `ebnerd_large`, and reduced
targeted teaching depth.)_

### D33 — EB-NeRD scale: 5% of `ebnerd_large`'s users, selected by `user_id % 100 < 5`

**Decided 2026-09-15 (Chaitanya).**

**Forced, not chosen:** the sample is user-level. Sampling impressions instead leaves a
sampled user's history and impressions out of sync.

**Measured before choosing** (`ebnerd_large`: 974,791 users; 12,063,890 train + 12,566,385
validation impressions; train 18–25 May 2023, validation 25 May–1 Jun):

| option | users | train / val impressions | est. val AUC CI half-width |
|---|---|---|---|
| 2% | ~19.5k | ~240k / ~250k | ±0.0016 |
| **5% (chosen)** | **48,666** | **597,348 / 622,398** | **±0.001** |
| 15% | ~146k | ~1.82M / ~1.89M | ±0.0006 |

The CI column scales A1's measured ±0.005 on 17,749 EB-NeRD impressions by 1/√n, using
val = the earliest 70% of validation (D8). It is an estimate. A *paired* bootstrap
(Q3) will be narrower still, because the two systems share per-impression noise.

The filtering pass costs **~50 s and ~5 GB peak whatever the size**, because reading all
24M rows dominates. Size is only paid downstream, in features, training and bootstrap.

**Why 5%:** a CI about 5× narrower than A1's, while everything stays well inside 11 GB.
A gain too small to clear ±0.001 is too small to be worth claiming. 15% would triple the
cost of every later step for a CI improvement that no realistic effect size needs.
Training-row volume (impressions × K candidates, ~60M at K=100) exceeds RAM at *any* of
these sizes, so it is subsampled separately in D36 whatever is chosen here.

**Selection mechanism: `user_id % 100 < 5`, not `DataFrame.sample(seed=…)`.**
- Deterministic across machines and Polars versions. A seeded sample's exact rows are
  not guaranteed to stay the same across library versions.
- **Nested:** the 5% set is exactly contained in any larger `< p` set, so growing the
  sample later only adds users.
- Rejected alternative: the seeded random sample. Statistically the cleanest, but neither
  reproducible across versions nor nested.

**The assumption this rests on, checked rather than trusted.** IDs must not pattern by
their last two digits.
- Residue counts over all 974,791 users: min 9,496, max 9,922 against an expected 9,748.
  χ² = 71.0 on 99 degrees of freedom, below the 95% critical value of ~123.2, so
  consistent with uniform.
- In-sample vs out-of-sample behaviour (train; validation agrees to the same precision):

  | metric | out of sample | in sample |
  |---|---|---|
  | impressions per user | 15.31 | 15.22 |
  | mean inview size | 11.092 | 11.089 |
  | SSO (single sign-on) user rate | 0.1056 | 0.1031 |
  | desktop rate | 0.3405 | 0.3395 |
  | median read time | 21.0 | 21.0 |
  | clicks per impression | 1.0058 | 1.0057 |
  | subscriber rate | 0.0674 | 0.0657 (val: 0.0654 vs 0.0657) |

**Completeness check (R10).** In both splits, every sampled user with impressions has
exactly one history row, and none is missing or duplicated (train 39,260 users ↔ 39,260
rows; validation 39,420 ↔ 39,420). No empty histories.

**Costs accepted:** EB-NeRD numbers are on a 5% user sample, not the full population. The
leaderboard submission still scores the full 13.5M-impression testset. Only training and
offline evaluation use the sample.

**Implementation.**
- **Where the filter runs:** `ingest_ebnerd.user_sample_filter(pct)` applies to the raw
  numeric `user_id`, before the `ebnerd:` prefix (modulo means nothing on a string). It
  is pushed into the Parquet scan for all four user-keyed reads, with one shared `pct`.
  Articles are never sampled.
- **Validation:** an int in 1..100 or None. `True`, `5.0` and `"5"` are rejected rather
  than coerced.
- **Config — `ebnerd.yaml` repointed rather than a second config (Chaitanya).** A second
  config would have needed a flag on every downstream script, or scripts would silently
  read whichever store was built last. A1's demo store reproduces from tag
  `phase-5-complete`.
- **Embeddings:** `scripts/assemble_embeddings.py` builds `embeddings.parquet` by
  combining existing vectors instead of re-embedding. It refuses to write unless IDs
  match the store exactly, and writes to a temporary file then renames, because the
  MIND rows are read from the file being replaced.

**Measured at build.** 40 s wall, 3.3 GB peak RSS (resident set size, the RAM actually
held), for the whole store including MIND.

| EB-NeRD split | impressions | users |
|---|---|---|
| train | 597,348 | 39,260 |
| val | 435,677 | 36,402 |
| test | 186,721 | 27,897 |

Split boundaries hold strictly. Val came in at 435,677, against the ~440k assumed for
the CI estimate above.

### D34 — the Q1 feature allowlist: option B, symmetric core plus EB-NeRD session context

**Decided 2026-09-18 (Chaitanya), option B over the recommended A**, accepting about an
hour more work to cover Q1.2's session features directly rather than arguing their
absence. Consequence to carry into D36: the two datasets' re-rankers see different
feature sets, so their scores are not compared feature-for-feature. A sub-fork followed
immediately (D34a, below). Evidence gathered 2026-09-16: The allowlist
is an allowlist and never a denylist (system design contract): a feature is in only if it can be
computed from facts true *strictly before* the impression is served.

**What the schemas actually permit — measured, not assumed.**

EB-NeRD's test behaviours, against its train behaviours:

| Column | In test? | Verdict |
|---|---|---|
| `article_ids_clicked`, `article_id` | **absent** | the label, and the clicked article |
| `next_read_time`, `next_scroll_percentage` | **absent** | post-impression by name |
| `read_time`, `scroll_percentage` | **present** | **present but not available**: both are measured *after* the impression is served. Q9 ablation arm, never scored |
| `session_id`, `device_type` | present | usable |
| `age`, `gender`, `postcode` | present | 97.0% / 92.9% / 97.7% null in our sample (testset: 97.1 / 92.9), so nearly no signal |
| `is_subscriber` | present, 0% null | usable |
| `is_beyond_accuracy` | test-only | a competition flag, not a feature |

`scroll_percentage` is also 70.7% null in train and 71.6% in test — it is both unavailable
*and* mostly missing.

Store fields, after the D33 rebuild:

| | MIND | EB-NeRD |
|---|---|---|
| `published_time` | **100% null** | 0% null |
| `history_timestamps` | **100% null** | 0% null |
| `session_id`, `read_time` | **100% null** | 0% null |
| `body`, `sentiment_score`, `total_pageviews` | 100% null | 0% / 0% / 86.5% null |
| median history length | 13 | 83.5 |

**The catalogue contains the future** (checked 2026-09-18). `ebnerd_large` and the
testset share one 125,541-article catalogue, which is why they were frame-equal. It
covers publication dates up to 2023-07-11, and 1,069 articles appear after the
validation week has ended. Any feature derived from the catalogue *as a whole* leaks.
Only per-article fields, evaluated relative to the impression's own timestamp, are
allowed.

**Three consequences that are forced, not chosen.**
1. **Freshness on MIND cannot come from `published_time`.** It must come from
   `availability.first_seen_times` (A1's, already label-free and already a strict `<`).
2. **Recency decay is not the same feature on both datasets.** EB-NeRD has real
   timestamps; MIND has only list position. Per A1's lesson about scale-dependent
   quantities, **the two datasets' decay features are not comparable to each other**, and
   the design note must say so rather than tabling them side by side.
3. **Session features are EB-NeRD-only.** MIND has no session or dwell fields at all.

**Options for the allowlist's scope.**

- **A — symmetric core.** Recency-weighted history (exponential decay), click count,
  category match against history, article freshness, and popularity computed **inside the
  train window only**. Same feature names on both datasets, with the two decay
  implementations documented as non-comparable. Smallest, most defensible, and every
  feature exists for both datasets.
- **B — A plus EB-NeRD session context.** Adds within-session position and session click
  count, which Q1.2 explicitly asks for. Richer where the data allows, asymmetric between
  datasets, and needs the re-ranker to tolerate a feature set that differs per dataset.
- **C — B plus demographics.** `is_subscriber` is clean (0% null); `age`/`gender`/
  `postcode` are 93–98% null. Most of C is one usable feature and three nearly empty ones.

**Quarantined in `features/unavailable.py` regardless of the choice, as the Q9 arm:**
`read_time`, `scroll_percentage`, `next_*`, `total_pageviews`/`total_inviews`, and any
popularity count computed over the evaluated window. A1 already priced one of these:
future-window popularity bought **+0.0234 AUC on MIND and +0.0541 on EB-NeRD**, and
scored 0.6657 alone on EB-NeRD — better than every honest method we have.

### D34a — EB-NeRD session features are label-free

**Decided 2026-09-18 (Chaitanya), the recommended option.** Session features use only
`session_id`, `impression_time` and `device_type`:
- position in the session,
- time since the session started,
- number of impressions so far in the session,
- device.

They never use `clicked_article_ids` from other impressions.

**Why.** Clicks from earlier in the same session would be legitimate in live serving,
since they happen before the current impression. But the Codabench testset carries no
labels, so such features cannot be computed there. Using them would need a second,
leaderboard-only model, and would break the offline-to-leaderboard calibration that A1
established. That calibration is the project's most defensible claim.

**Rejected:**
- *Click features as an offline-only ablation arm.* A richer Q1 story, but it is
  drop-order item 1 on a two-day schedule.
- *Click features in the main model.* Two models, and a broken calibration.

**Boundary rule, recorded for the code.** "Earlier in the session" means a **strictly
earlier** `impression_time`. EB-NeRD timestamps have one-second resolution, so ties are
possible. A tie is treated as *not* earlier, which is the conservative direction: it can
only undercount the past, never admit the present.

### D35 — recency decay is multi-scale: short, medium and infinite half-lives as separate features

**Decided 2026-09-18 (Chaitanya), the recommended option.**

| Scale | EB-NeRD (time) | MIND (list position) |
|---|---|---|
| short | 1 day | 3 positions |
| medium | 7 days (one behaviour window) | 10 positions |
| infinite | plain mean = A1's feature | plain mean = A1's feature |

The re-ranker learns how much each scale counts.
- **No value of *h* is tuned on val,** so nothing here is fitted to the numbers it will be
  judged on.
- **The infinite scale is exactly A1's mean-pooled user vector (N = 100),** so Q2's
  before-and-after comparison contains the old system as one of the new system's inputs.

**Rejected:**
- *One argued half-life.* Simplest, but "why 7?" has no measured answer.
- *Grid search on the last day of the train week.* Principled, but ~1 h of extra work and
  a second history snapshot to leakage-check.

**Measured before choosing** (EB-NeRD val/test, one impression per user, 16.5M
history-click pairings):
- **0 history clicks dated after their impression.**
- Age of the user's **newest** history click at impression time: median **5.0 days**
  (p10 1.2, p90 7.7). The snapshot stops where the impression window starts.
- Median click 13.6 days old, oldest 23 days.
- MIND history length: median 13 positions, p10 3, p90 55.
- A half-life measured in hours would therefore mean nothing on EB-NeRD.

**Ages are measured relative to the user's newest click,** i.e. weight = 0.5^((t_newest − tᵢ)/h).
- **Mathematically:** a normalised weighted average is unchanged when every weight is
  multiplied by the same factor, so this is identical to measuring from the impression.
- **Numerically:** it is not the same. From the impression, a 600 h age at h = 1 h
  gives 0.5⁶⁰⁰ ≈ 10⁻¹⁸¹. A little further and every weight underflows to exactly 0,
  giving 0/0 = NaN. Relative to the newest click, that click always gets weight 1, so
  the sum cannot vanish.
- **Consequence 1:** the impression time cancels out, so the decayed user vector and
  category shares are **per user, computed once**, not per impression.
- **Consequence 2, the cost:** that averaging cannot see *how long the user has been
  away*. So EB-NeRD gets one extra per-impression feature, **hours since the newest
  history click**. The negative-age guard lives on that feature and **raises** on any
  history click at or after its impression, rather than clipping.
