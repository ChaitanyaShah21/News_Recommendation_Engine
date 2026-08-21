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

_Further decisions appended as they are made._
