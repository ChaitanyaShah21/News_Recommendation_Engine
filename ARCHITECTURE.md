# Architecture & Decision Log

Last updated: 2026-08-21

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
  Python — not actually an inverted index. 37,777 queries × 65,238 documents would not
  finish.
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

**Judgment call made without a decision point** (R6 trivia exception — changes runtime,
not results): the query depends only on the user's history, which is fixed per user
within a split, so scoring runs **once per unique user** and joins back to impressions
rather than once per impression. Identical numbers; 26% less work on MIND and 12× less on
EB-NeRD (1,437 users behind 17,749 impressions).

---

_Further decisions appended as they are made._
