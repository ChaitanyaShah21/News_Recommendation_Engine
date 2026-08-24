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

---

#### `.list.eval()` and `pl.element()`
**Plain:** A mini for-loop that runs one instruction on every item inside each row's
list, individually — `pl.element()` is "the current item," the way a loop variable
stands for the current item in a plain Python `for` loop.

**Technical:** `.list.eval(expr)` applies `expr` to every element of a list column,
row-wise; `pl.element()` inside that expression refers to the element being processed.
Composable and chainable — `ingest_mind.load_behaviors` uses one pass to filter tokens
(`.filter(pl.element().str.ends_with("-1"))`) and a second to transform the survivors
(strip suffix, add dataset prefix), rather than combining both into one dense expression.

---

#### Regex (regular expression)
**Plain:** A pattern language for describing "text that looks like this," so you can
find or replace it without listing every exact string it might be.

**Technical:** `r"-[01]$"` means: a literal `-`, then either `0` or `1`, anchored to the
end of the string (`$`). Used in `.str.replace(r"-[01]$", "")` to strip MIND's click-label
suffix. Verified against real data (R10) that every token matches this pattern exactly,
so it can't accidentally strip something that isn't the intended suffix.

---

#### strptime format string
**Plain:** A template describing how a date/time is written as text, so it can be turned
into an actual, comparable date value instead of staying a string.

**Technical:** `%m/%d/%Y %I:%M:%S %p` decodes MIND's `"11/11/2019 9:05:58 AM"` — month,
day, 4-digit year, 12-hour clock hour, minute, second, AM/PM marker. Turning timestamps
into real `Datetime` values (not strings) is what later makes the Q1.3 temporal split
possible — you can't reliably compare "is this before that" on text.

---

#### `.unique(subset=..., keep=...)`
**Plain:** Keep only one row per distinct value in a chosen column, throwing the rest
away — like collapsing a stack of duplicate library cards for the same person into one.

**Technical:** `.unique(subset="user_id", keep="first")` deduplicates rows by `user_id`,
keeping the first occurrence of each. Used in `ingest_mind.load_history` to collapse
MIND's repeated per-impression history rows into one row per user — safe here
specifically because R10 verified every row for a given user carries an identical
`history` string, so *which* row survives doesn't affect the result.

---

#### `.fill_null(value)`
**Plain:** Replace every "we don't know" with a specific, usable stand-in value, so
nothing downstream has to keep asking "wait, is this null?"

**Technical:** Replaces every null in a column with `value`. `ingest_mind.load_history`
uses `.fill_null([])` to turn a cold-start user's null history (an artefact of
`.str.split()` propagating null rather than returning an empty list — the exact scenario
`system design contract`'s R9 example is built around) into a genuine empty list, so every downstream
caller can safely call `.list.len()` without a separate cold-start check every time.

---

#### Nested function (closure)
**Plain:** A small helper defined *inside* another function, existing only to avoid
repeating the same few lines twice, and signalling "this only makes sense in here."

**Technical:** `ingest_ebnerd.load_behaviors` defines `def prefixed(list_col): ...`
inside itself, called twice (`article_ids_inview`, `article_ids_clicked`) to avoid
duplicating the cast-and-prefix expression. It returns a Polars expression, not a
computed value — nothing runs until the outer `.collect()`.

---

#### Lazy pattern in practice: `pl.scan_parquet` + `.collect()`
**Plain:** Build the whole "what to do" plan first, lazily, then trigger it once at the
very end — versus loading everything immediately the way `pl.read_parquet` does.

**Technical:** `ingest_ebnerd.py`'s `load_behaviors`/`load_history` use `pl.scan_parquet`
(returns a `LazyFrame`) and only call `.collect()` at the very end, letting Polars'
query optimizer push column selection down before materializing anything — the pattern
that matters at EB-NeRD-large's 12M+ row scale (Phase 5), not yet needed at demo scale,
but written this way now so the same function doesn't need rewriting later. `.collect()`
at the end keeps the *return type* (`DataFrame`) consistent with `ingest_mind.py`'s
eager functions — full memory-safe batched processing at 12M+ rows is still a Phase 5
concern this alone doesn't solve.

---

#### `pl.when().then().otherwise()`
**Plain:** A row-by-row if/else that produces a whole column at once — "if this
condition holds for this row, use value A, otherwise value B," for every row
simultaneously, rather than looping row by row.

**Technical:** `pl.when(condition).then(value_if_true).otherwise(value_if_false)`
builds a conditional expression. Used in `temporal_split.add_impressions_split` to tag
each dev/validation row `"val"` or `"test"` depending on whether its `timestamp` falls
before or after the computed cutoff.

---

#### Feature store
**Plain:** A prep fridge for data — clean, ready-to-use tables computed once and stored,
so every later script grabs what it needs instead of re-washing the lettuce (re-parsing
raw files) every single time it runs.

**Technical:** The unified-schema tables, computed by ingestion + the temporal split,
persisted to `data/processed/` as Parquet (D9: three combined files —
`articles.parquet`, `impressions.parquet`, `history.parquet` — both datasets together,
filtered by the existing `dataset`/`split` columns). Read directly by Q2/Q3/Q4 instead
of recomputed from raw files each time. Carries a real risk — staleness: if ingestion
code changes but the store isn't rebuilt, downstream code silently reads outdated data —
which is exactly why Q1.5's one-command rebuild matters, not just as a convenience.

---

#### `__file__`, `.resolve()`, and script-relative paths
**Plain:** A script asking "where am I actually located on disk?" instead of guessing
based on wherever the terminal happened to be standing when it got run.

**Technical:** `__file__` is a variable Python sets automatically inside every module to
that file's own path. `.resolve()` turns it into an absolute path with no `..`/symlink
ambiguity. `scripts/build_pipeline.py` uses `Path(__file__).resolve().parent.parent`
to find the repo root reliably regardless of the caller's working directory, then adds
`src/` to `sys.path` so `from newsrec.build import ...` can find the package.

---

#### stderr and exit codes
**Plain:** Two separate channels for a program's output — one for normal results
(stdout), one for diagnostics/errors (stderr) — plus a number reporting whether the
program succeeded or failed, so other programs/scripts can check without reading text.

**Technical:** `print(msg, file=sys.stderr)` sends output to stderr instead of the
default stdout. `sys.exit(1)` stops the program with exit code 1 — by Unix convention,
`0` means success, nonzero means failure, which matters if this script is ever chained
in a Makefile or CI pipeline that checks `$?`.

---

#### `if __name__ == "__main__":`
**Plain:** "Only actually run this if the file was executed directly, not if something
else merely imported it."

**Technical:** Python sets a module's `__name__` variable to `"__main__"` when that file
is run directly, but to the module's real name when it's imported elsewhere. Guarding
the main logic behind this check lets a script be safely imported (e.g. for testing)
without immediately triggering its side effects.

### Phase 2

#### Tokenisation
**Plain:** Cutting a sentence into the individual words you're going to count.

**Technical:** Splitting text into the atomic units a retrieval model indexes and scores.
Ours (D11): lowercase, then split on anything that is not a Unicode letter or digit
(`[^\W_]+`), applied identically to English and Danish. Choices here are not neutral — a
non-Unicode-aware pattern turns `"Rådden kørsel"` into `r dden k rsel`, silently.

---

#### TF (Term Frequency), written `f(t,D)`
**Plain:** How many times this word appears in this one article.

**Technical:** The raw count of term `t` in document `D`. In BM25 it enters through a
*saturating* function, `f(k₁+1)/(f + k₁·…)`, with a hard ceiling of `k₁+1` — not through
a logarithm (that's TF-IDF) and not linearly. Ten mentions score barely more than three.

---

#### DF (Document Frequency), written `n(t)`
**Plain:** In how many *different* articles does this word appear at all? (Never "how
many times" — an article mentioning *vote* five times still counts once.)

**Technical:** The number of documents containing `t` at least once. Confusing it with TF
is the easiest BM25 bug to write and the hardest to notice: it inflates `n(t)`, deflates
every IDF, and raises no error. In our implementation `n(t)` is literally the **length of
the term's posting list** — verified equal to `indptr[j+1] - indptr[j]` for all 92,593
terms across both corpora.

---

#### IDF (Inverse Document Frequency)
**Plain:** A rarity bonus. Matching on *inflation* tells you far more than matching on
*the*, so rare words are worth more.

**Technical:** `IDF(t) = ln((N − n(t) + 0.5)/(n(t) + 0.5) + 1)`, where `N` is corpus size.
The `+0.5` terms are smoothing; the `+1` inside the log keeps IDF non-negative, so a
document is never *penalised* for containing a common query word (older BM25 variants
allowed that). Depends only on the **term and corpus** — never on the document — so it
cannot explain why one document outranks another for a given query; it decides how much
each *query word* matters relative to the others. Measured on MIND: `the` → 0.26, rarest
terms → 10.68.

---

#### Length normalisation, `b`
**Plain:** Discounting a long article so it doesn't win just by containing more words.

**Technical:** The denominator factor `1 − b + b·|D|/avgdl`, where `|D|` is the document's
token count and `avgdl` the corpus mean. `b=0` ignores length entirely, `b=1` divides
fully by relative length, `b=0.75` is the usual blend. Note that the whole bracket is
multiplied by `k₁` — so **`b` only has effect through `k₁`**, and setting `k₁=0` deletes
length normalisation along with term-frequency saturation.

---

#### Inverted index
**Plain:** The index at the back of a book — for each word, the list of pages it's on —
rather than reading every page to find the word.

**Technical:** A mapping from term → **posting list** (the documents containing it). Lets
scoring touch only documents sharing at least one query term instead of the whole corpus.
Ours is a sparse document-term matrix; its compressed-sparse-column form *is* the index
(D14). See [[csr-csc]].

---

#### CSR / CSC (Compressed Sparse Row / Column) {#csr-csc}
**Plain:** One long paper tape listing only the non-empty entries, plus a small index card
saying where each row's lines start and stop.

**Technical:** Three parallel arrays. `data` — the non-zero values. `indices` — which
column (CSR) or row (CSC) each value sits in, read in lockstep with `data` at the same
position. `indptr` — **positions into `data`/`indices`**, not column numbers, with one
more entry than there are rows; row `i` occupies `indptr[i]:indptr[i+1]`, so an empty row
falls out naturally as an empty slice. CSC is the same construction applied to columns,
which is why its per-column runs are exactly posting lists. Measured on MIND: 19.2 MB
sparse against 15.9 GB dense, at 0.0594% density.

---

#### Query term frequency, `qtf`
**Plain:** How many of the user's last 10 clicked headlines contained this word.

**Technical:** The count of a term within the *query*, distinct from `f(t,D)` within the
document. Matters here only because our queries are manufactured from click history
rather than typed. We count it linearly (D16); the alternatives are binary (kept as an
ablation) and `k₃`-saturated (Robertson's full BM25).
