# Glossary

Every term defined once — plain language first, then technically.
Grows as we go. If a term is used anywhere in this project and is not here, that is a bug.

---

## Core vocabulary of the task

### Impression
**Plain:** One moment when a website showed somebody a list of headlines. Like one glance
at a newsstand — these particular papers were on display, at this particular time.

**Technical:** A single logged event containing a user identifier, a timestamp, the list
of candidate articles displayed (`article_ids_inview` in EB-NeRD, the `impressions` field
in MIND), and — in training data only — which of them were clicked.

---

### Candidate
**Plain:** One of the headlines that was on display in that glance.

**Technical:** An article present in an impression's display list. The set we must rank.
Typically 5–100 articles in EB-NeRD (average ~11), variable in MIND.

---

### Click label
**Plain:** Whether the person actually picked up that paper.

**Technical:** A binary target, 1 if clicked and 0 otherwise. In MIND it is encoded as a
suffix on the article identifier (`N55689-1` clicked, `N35729-0` not). In EB-NeRD it is a
separate list column `article_ids_clicked`. **Absent from both test sets** — that is what
we predict.

---

### History
**Plain:** The list of articles this person has read before now.

**Technical:** The user's prior click sequence. MIND stores it inline in every behaviour
row as a space-separated string. EB-NeRD stores it in a separate `history.parquet` file,
one row per user, as parallel lists (article identifiers, timestamps, read times, scroll
percentages).

---

### Retrieval vs re-ranking
**Plain:** Retrieval is walking into a library of 120,000 books and pulling the 100 that
might interest you. Re-ranking is being handed 11 books and putting them in the best order.

**Technical:** *Retrieval* (also *candidate generation*) searches the entire article corpus
and returns top-K, measured by recall@K. *Re-ranking* scores a supplied candidate list and
orders it, measured by ranking metrics. **This assignment needs both** — questions Q2/Q3
grade retrieval, the leaderboards grade re-ranking.

---

## Terms added per phase

### Phase 0

#### BM25 (Best Match 25)
**Plain:** A card-catalog search: count how many of the query's words appear in a
document, but give more credit for rare, distinctive words than common ones, and
discount very long documents so they don't win just by containing more words.

**Technical:** A ranking function scoring document relevance to a query by combining
term frequency (how often each query term appears in the document), inverse document
frequency (how rare that term is across the whole corpus — rarer terms score higher),
and document-length normalisation. Purely lexical: matches surface word forms, blind to
synonymy. Used in Q2.

---

#### Embedding
**Plain:** A librarian who has read every article and can place two articles near each
other on a map if they're about similar things — even if they don't share a single word.

**Technical:** A dense numeric vector representation of text (or other data) such that
semantically similar inputs map to nearby points in the vector space. Computed by a
model (Word2Vec, BERT, XLM-RoBERTa) or provided pre-computed (EB-NeRD ships article
embeddings). Used in Q3.

---

#### ANN (Approximate Nearest Neighbour) index
**Plain:** Instead of comparing a query to every single article on the shelf one by one
to find the closest matches, use a shortcut structure that finds ones that are *close
enough*, fast, and almost always correct.

**Technical:** A data structure/algorithm (e.g. FAISS, ScaNN) for retrieving vectors
near a query vector in high-dimensional space without the cost of an exhaustive
brute-force comparison against every stored vector. Trades a small amount of accuracy
for large speed gains at scale. Used in Q3 to search article embeddings.

---

#### Recall@K
**Plain:** Of all the articles the person actually went on to read, what fraction showed
up somewhere in your shortlist of K candidates — regardless of where in that list?

**Technical:** `(# ground-truth positives present in the top-K retrieved set) / (# ground-truth
positives total)`. Order-blind within the top-K: only presence counts. Reported for
K ∈ {50, 100, 200} in Q2 and Q3.

---

#### AUC (Area Under the Curve)
**Plain:** If you pick one article the person clicked and one they didn't, how often does
your model correctly rank the clicked one higher?

**Technical:** Area under the ROC (Receiver Operating Characteristic) curve; equivalent
to the probability a randomly chosen positive is ranked above a randomly chosen negative.
Used in Q4 as an overall ranking-quality metric.

---

#### MRR (Mean Reciprocal Rank)
**Plain:** How close to the very top of the list did the first correct answer land,
averaged across all users? Landing it at #1 scores much better than #10.

**Technical:** For each query, `1 / rank of first relevant item`; averaged across all
queries. Sensitive to *order*, unlike recall@K. Used in Q4.

---

#### nDCG@k (normalised Discounted Cumulative Gain at k)
**Plain:** A score for how good the top-k results are overall — correct answers near the
top count a lot, the same correct answer buried near the bottom of the top-k counts for
much less.

**Technical:** Discounted Cumulative Gain sums relevance scores of the top-k results,
each divided by a logarithmic discount of its rank position; normalised by the DCG of
the ideal (perfectly sorted) ordering, giving a score in [0, 1]. Reported at k = 5 and
k = 10 in Q4.

---

#### Bootstrap confidence interval
**Plain:** Resample your set of users (with replacement) many times, recompute the
metric each time, and look at the spread of results — that spread tells you how much the
score might have wobbled if you'd tested on a slightly different group of users.

**Technical:** A non-parametric method for estimating a statistic's sampling
distribution by repeated resampling (with replacement) of the observed data; the 2.5th
and 97.5th percentiles of the resulting distribution give a 95% confidence interval.
Required alongside every Q4 metric.

---

#### Cold-start user
**Plain:** A brand-new reader with little or no history to go on — hard to personalise
for because there's nothing to learn from yet.

**Technical:** A user with few or no prior interactions in the click-history window,
contrasted with "warm" users who have substantial history. One of the required Q4
evaluation slices.

---

#### Temporal leakage
**Plain:** Accidentally letting the model peek at tomorrow's newspaper sales while
deciding today's — makes an offline test look better than the system would actually
perform in production.

**Technical:** Any situation where information from after the prediction timestamp
(e.g. a future click, a feature computed over a window including future events) enters
training or feature computation for an earlier-timestamped example. Must be structurally
prevented, not just avoided by convention — Q9 requires an automated test
(`tests/test_no_leakage.py`) asserting it cannot happen.

---

### Phase 1

#### Unified schema
**Plain:** Translating both companies' paperwork into one common form before any
downstream process touches it, so nothing has to special-case "which company is this."

**Technical:** A small, fixed set of tables (articles, impressions, click-history) with
consistent column names and types that both MIND and EB-NeRD get transformed into during
ingestion. A superset schema — columns unique to one dataset (sentiment, body text,
entity embeddings) are kept and simply null for the other, not discarded. Confirmed
structural differences requiring translation: MIND's `impressions` string
(`"N55689-1 N35729-0"`) must split into candidate-list + clicked-list to match
EB-NeRD's already-split `article_ids_inview` / `article_ids_clicked`; MIND's inline
per-impression `history` string must collapse to one-row-per-user to match EB-NeRD's
`history.parquet`. [[retrieval-vs-re-ranking]]

---

#### TSV (Tab-Separated Values)
**Plain:** A plain-text table, one row per line, columns separated by tab characters —
no type information, often no header row.

**Technical:** MIND's file format. `behaviors.tsv` and `news.tsv` have **no header row**;
column names/types are supplied externally by the reading code, not stored in the file.

---

#### Parquet
**Plain:** A compressed, typed table file that remembers its own column names and types,
and lets you read just the columns you need without loading the whole file.

**Technical:** EB-NeRD's file format — columnar, binary, self-describing schema. Enables
`pl.scan_parquet` lazy scanning and per-row-group batch reads, which is how the 12M+ row
EB-NeRD-large behaviors file is explored/processed without exhausting RAM.

---

#### Polars expression (`pl.col`, `pl.lit`, `.alias`)
**Plain:** A small, reusable instruction for "how to compute one column" — Polars only
actually runs it once it reaches the end of a `.select()`/`.with_columns()` call, rather
than computing each piece the instant you write it.

**Technical:** `pl.col("x")` references an existing column unchanged. `pl.lit(v)`
creates a column repeating a fixed value (or `None`) on every row, independent of any
existing column. `.alias("name")` renames whatever expression precedes it. Chained
together inside `.select(...)`, these are how `ingest_mind.py` builds the unified
`articles` schema column by column, including dataset-prefixing IDs
(`(pl.lit("mind:") + pl.col("article_id")).alias("article_id")`) and null-padding
columns MIND doesn't have (`pl.lit(None, dtype=pl.Utf8).alias("body")`).

---

#### Eager vs. lazy evaluation
**Plain:** Eager is doing each instruction the moment you give it, one at a time. Lazy
is handing over the *whole list* of instructions first, letting a planner figure out the
smartest order and which parts to skip, and only then actually doing the work.

**Technical:** `pl.read_csv(...)` and plain `pl.read_parquet(...)` are eager — they load
data into memory immediately. `pl.scan_parquet(...)` is lazy — it returns a query plan
(`LazyFrame`) that isn't executed until `.collect()` is called, letting Polars' query
optimizer decide what to actually compute. Used eagerly for MIND (small enough to load
directly) and lazily for EB-NeRD-large in the provided notebook (12M+ row files that
don't fit comfortably in 7 GB RAM).

_Phase 2 terms appear here once taught._
