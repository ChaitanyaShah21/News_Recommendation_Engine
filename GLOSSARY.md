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

---

## Phase 3 vocabulary — semantic retrieval

### Embedding
**Plain:** A warehouse with no labels on the shelves, where every article has been placed
at a spot chosen so that articles about similar things sit near each other. To find things
like what you read, you stand where your reading sits and grab what is nearby.

**Technical:** A function mapping a piece of text to a fixed-length vector of real
numbers — 384 of them for our model. Geometric closeness in that space approximates
semantic similarity. Contrast with BM25's representation, which is *sparse* (65,238 x
60,951 with 2.36M non-zeros, one dimension per vocabulary term) where this is *dense*
(77,015 x 384, essentially every entry non-zero). The dense one handles synonyms and
paraphrase for free but has no drawer to look in for an exact rare term.

---

### Cosine similarity
**Plain:** How closely two arrows point in the same direction, ignoring how long they are.

**Technical:** `cos(a,b) = (a·b) / (‖a‖‖b‖)`. Three named parts. The **dot product**
`a·b = Σᵢ aᵢbᵢ` is one number, large when both vectors are large in the same dimensions.
The **norm** `‖a‖ = √(Σᵢ aᵢ²)` is the vector's length, pure magnitude. **Dividing by both
norms** strips magnitude out, leaving only the angle: 1 = identical direction,
0 = perpendicular/unrelated, −1 = opposite. Angle rather than straight-line distance,
because length tracks incidental things like how long the text is.

---

### L2 normalisation
**Plain:** Rescaling every arrow to be exactly one unit long, so only direction is left.

**Technical:** Dividing each vector by its own norm, so `‖a‖ = 1` for all. Then cosine
similarity's denominator is 1 and **cosine collapses into a plain dot product** — which
turns "rank every article for this user" into one matrix multiplication.

Why it is not optional: `u·a = cos(u,a) × ‖u‖ × ‖a‖`. When ranking all articles for **one**
user, `‖u‖` is the same constant in every score and cannot reorder anything — but `‖a‖`
varies per article and distorts the ranking directly. Measured on our model: raw norms
spread **1.71×**, against a real signal gap of 0.936 (translation pair) versus 0.130 (same
language, different topic). Enough to reorder results on its own, and the bias is
*systematic* — the same high-norm articles get promoted into every user's list, which
would show up in Q4's coverage and diversity metrics as a fake finding.

---

### Mean pooling
**Plain:** Averaging the positions of the ten articles you last read, and standing there.

**Technical:** The user representation Q3.3 asks for — the component-wise average of the
embeddings of a user's last N clicked articles. It is the direct analogue of D12's
"concatenate the last N titles" for BM25, and we reuse N = 10 so Q3.5 varies only the
algorithm.

**The property that matters, and it is exact:** maximising `Σᵢ cos(u, aᵢ)` over unit
vectors `u` gives `u ∝ Σᵢ aᵢ` — the normalised mean. So the mean-pooled user vector is
**precisely the point in the whole space with the highest average similarity to that
user's own history articles**. The self-match is not emergent as it was with BM25's bag of
words; it falls out of the arithmetic. Every article is also its own nearest neighbour at
cosine exactly 1.000 (verified on our corpus). Both facts are why D15's history exclusion
binds *more* tightly here than it did in Q2, and must be applied before top-K.

**Known weakness:** the mean of vectors pointing different ways lands *between* them, at a
spot where nothing real may sit. A reader of both football and parliamentary politics gets
a vector representing neither. BM25 does not have this failure — a union of terms still
matches football articles *and* politics articles.

---

### ANN (Approximate Nearest Neighbour)
**Plain:** Instead of measuring the distance to every article in the warehouse, organise it
into aisles with signs and only search the nearest few. Much faster, but you will miss an
article sitting just over an aisle boundary.

**Technical:** Methods (FAISS — Facebook AI Similarity Search; ScaNN — Scalable Nearest
Neighbors) that pre-partition or compress vectors so each query touches a fraction of the
corpus, reported as a recall-versus-speed curve rather than exact answers. **We use exact
brute force instead (D21)**, which Q3.2 permits explicitly: one dense matrix product plus
a top-K, batched. Being exact, it cannot cost us the recall@K that Q3.4 reports. The
constraint is memory, not arithmetic — a full 37,777 × 65,238 float32 score matrix is
9.9 GB against 7 GB of RAM.

---

### Subword token
**Plain:** Models do not read words, they read word-fragments from a fixed list. An unusual
word gets chopped into several pieces.

**Technical:** The unit a transformer's tokeniser produces. Our model's vocabulary holds
250,037 of them and it truncates past **128 tokens**, silently. Measured on our corpus:
MIND averages 1.55 subword tokens per whitespace word and EB-NeRD 1.68 (Danish fragments
more), so **7.7% of MIND articles exceed the limit — but only 1.37% of MIND's total tokens
are lost, and 0.00% of EB-NeRD's** (max 121). Accepted rather than worked around; recorded
because it is a real dataset asymmetry in Q3.5's comparison.

---

## Phase 4 — evaluation metrics

### Re-ranking (versus retrieval)
**Plain:** Head office sends the newsagent 22 newspapers. He didn't pick them; his only
job is deciding what goes at eye level. That is re-ranking. Choosing which 22 to order
out of every paper printed nationwide is retrieval.

**Technical:** Retrieval (Q2/Q3) selects top-K from the whole corpus and is graded by
recall@K. Re-ranking (Q4, Q5) orders the platform's own supplied candidate list —
MIND's `impressions` field, EB-NeRD's `article_ids_inview` — and is graded by AUC, MRR
and nDCG. The split is **forced, not stylistic**: every Q4 metric needs a per-item
clicked/not-clicked label, and only the shown list carries one. An unshown corpus article
is *unlabelled*, not negative. Both Codabench leaderboards score the re-ranking task.

---

### AUC (Area Under the Curve — the Receiver Operating Characteristic curve)
**Plain:** Take one paper the customer bought and one they ignored, at random. Did we put
the bought one higher? AUC is the fraction of such pairs we got right. 0.5 is a coin flip.

**Technical:** over one impression, with `P` clicked candidates and `N` unclicked,

```
  AUC = (1/(P*N)) * SUM_p SUM_n [ 1 if s_p > s_n ; 0.5 if s_p == s_n ; 0 otherwise ]
```

Equivalently, and what we compute (O(n log n) rather than O(P·N)):
`(sum of the clicked items' average ranks − P(P+1)/2) / (P·N)`. Average ranks are what
produce the 0.5 tie credit, so **AUC needs no tie policy of its own**. Undefined when
`P = 0` or `N = 0`. Sensitive to candidate-list length in a way MRR is not: a clicked item
at rank 3 scores 0.895 in a rack of 20 and 0.333 in a rack of 4 — which is why MIND's and
EB-NeRD's AUCs are not comparable to each other.

---

### MRR (Mean Reciprocal Rank)
**Plain:** Only where the *first* correct paper landed matters. First shelf 1.0, second
0.5, fifth 0.2. Average over customers.

**Technical:** mean over impressions of `1 / (rank of the first clicked candidate)`,
ranks 1-indexed. Ignores every correct item after the first, and is blind to rack size.
Undefined when the impression has no clicks.

---

### DCG, IDCG, nDCG@k (Discounted Cumulative Gain, Ideal, normalised)
**Plain:** Only the top k shelves are graded, and lower shelves earn less credit. Then
divide by the best score that customer's impression could possibly have earned, so
someone who bought 3 papers isn't automatically scored higher than someone who bought 1.

**Technical:**
```
  DCG@k  = SUM over i = 1..k of  rel(i) / log2(i + 1)      rel(i) = 1 if clicked
  IDCG@k = the same with min(P, k) clicked items packed into the top slots
  nDCG@k = DCG@k / IDCG@k
```
Discount by position: rank 1 → 1.000, rank 2 → 0.631, rank 5 → 0.387, rank 10 → 0.289.
The logarithm decays more gently than MRR's `1/i` (slot 2 is worth 63% of slot 1, not
50%), modelling a user who scans several items. **`min(P, k)` is load-bearing:** MIND has
impressions with 21 clicks, and normalising by all 21 would cap a flawless ranking at
~0.51 inside nDCG@10 for a reason about the cutoff, not the ranking.

**When k exceeds the candidate list, nDCG@k silently becomes nDCG@all** — a full-list
ordering metric, close to what AUC already measures. EB-NeRD's median rack is 9
candidates, so its nDCG@10 is routinely this case and nDCG@5 is the only cutoff there
still asking "did the click reach the visible slots?".

---

### Tie policy (pessimistic / optimistic bounds)
**Plain:** Two papers we rated identically — which do we call "higher"? If we always
guess in our own favour, our grade flatters us.

**Technical:** AUC defines its own tie rule; MRR and nDCG need a total order, so ties must
be broken outside the score. **`np.argsort` is stable**, so "do nothing" silently means
"rank ties by position in the raw candidate list" — leakage if that order carried click
signal. D23 breaks ties explicitly: *pessimistic* puts clicked items last within a tie
group (every metric becomes a lower bound), *optimistic* first, and both are reported so
the gap between them measures how much tie handling matters. Ties are detected by exact
float equality, which catches the structurally important group — candidates sharing no
term with the query, scoring exactly 0.0 — and not near-ties differing by float noise.

---

### Rank vector (and why it is the *inverse* of a sorted order)
**Plain:** A judge at a dog show can write the results two ways. Either *"first place:
the poodle; second place: the beagle"* — a list of winners in order. Or she walks down the
line of dogs in the order they happen to be standing and writes a number on each one's
collar: *"you got 2nd, you got 1st, you got 3rd"*. Both describe the same result. They are
different pieces of paper, and handing in the wrong one is not obviously wrong to look at
— it is still a list of numbers 1, 2, 3 with none repeated.

Both Codabench leaderboards want the **collar numbers**, in the order the dogs were
standing.

**Technical:** given per-candidate scores `s` of length n, the submission line is a vector
`r` where `r[i]` is the rank of candidate `i` in its **original position** in
`article_ids_inview` / `impressions`. This is the inverse permutation of the argsort:

```
order = argsort(-s)        # order[j] = index of the candidate that finished j-th
r[order[j]] = j + 1        # write the placing INTO the candidate's own slot
```

Emitting `argsort(-s) + 1` instead is the classic error. It is *also* a permutation of
1..n, also has the right length, and also passes every structural validation — so the
only symptom is a leaderboard score near random, on a channel that returns one number and
no diagnostics. See `src/newsrec/predict.py::rank_vector`, whose tests are pinned against
both competitions' published worked examples.

**Related trap in the same family:** `"mind:12345".lstrip("mind:")` returns `"2345"`.
`str.lstrip` removes *characters in the set*, not the prefix, so it silently corrupts only
those ids whose digits start with m/i/n/d/: — a partial corruption, which is harder to
notice than a total one. `removeprefix` is the correct call.

---

### Unlabeled split
**Plain:** The exam paper with the answer key torn off. You can still write answers on it;
you just cannot mark it yourself.

**Technical:** the Codabench test bundles carry candidate lists but no clicks — MIND's
test `behaviors.tsv` has 5 fields instead of 6 with no `-0`/`-1` suffixes, and EB-NeRD's
test `behaviors.parquet` simply has no `article_ids_clicked` column. Every accuracy metric
(AUC, MRR, nDCG) is therefore **undefined** on this split, which is the whole reason a
separate val split exists (D7). D30 represents the missing labels as an **absent column**
rather than an empty one, so code that reaches for them raises immediately instead of
averaging over fabricated zeros.

---

## Assignment 2 — Phase A1 (data scale-up)

### User-level sample
**Plain:** To survey a school, pick whole classrooms, not random pages torn out of random
exercise books. A torn-out page arrives without the pages it depends on.

**Technical:** the sample keeps or drops a *user*, and with them every impression and
history row they own. Sampling impressions independently would keep some of a user's
impressions while dropping others, and their history row would then describe clicks the
sample does not contain. D33.

### Modulo (hash-style) sampling
**Plain:** "Everyone whose ticket number ends in 00–04" picks about 5% of a crowd, picks
the same people every time, and "ends in 00–14" contains all of them plus more.

**Technical:** keep a row when `user_id % 100 < p`. The `%` (modulo) operator gives the
remainder after division, so the last two digits of the ID decide membership. It is
deterministic (no random generator, so no version drift) and **nested** (the set for p is
a subset of the set for any larger p). It is valid only if those last digits carry no
information about the user, which has to be checked (χ² below) rather than assumed.

### χ² (chi-squared) goodness-of-fit
**Plain:** Roll a die 600 times. You expect about 100 of each face. χ² adds up how far
each face's count strays from 100 and says whether the total is more than luck explains.

**Technical:** χ² = Σ (observed − expected)² / expected, summed over categories. Compare
it with the critical value for (categories − 1) degrees of freedom. Below that value,
the counts are consistent with the expected (here uniform) distribution. For D33: 100
residues, 99 degrees of freedom, χ² = 71.0 against a 95% critical value of ~123.2.

### Confidence-interval width versus sample size
**Plain:** Tasting one spoonful of soup tells you less than tasting ten. To halve your
uncertainty you need four times as many spoonfuls, not twice as many.

**Technical:** a CI's width shrinks with **1/√n**. Going from 17,749 to ~440,000
impressions (×25) narrows it about ×5. This is why 15% of users (×3 the data) would only
narrow the interval by about ×1.7 (√3) over 5%.

## Assignment 2 — Phase A2 (behavioural features)

### Exponential decay / half-life
**Plain:** yesterday's lunch tells you more than last month's. Older evidence counts
less, smoothly.
**Technical:** weight = 0.5^(age / h). *h* is the half-life, the age at which a click
counts half. D35 uses three values of *h* as separate features and measures age from the
user's newest click, so the weights cannot underflow to 0/0.

### Exposure share
**Plain:** what the newsagent is putting in the window right now.
**Technical:** the fraction of impressions in [T − w, T) whose candidate list contained
the article. Label-free, and a share rather than a count, so it survives the 5%-sample
vs full-population gap (D34b).

### Future-deletion invariance
**Plain:** if deleting tomorrow's newspaper changes today's answer, today's answer was
reading tomorrow's paper.
**Technical:** delete every impression after τ and recompute. Features of impressions at
or before τ must be bit-identical. It covers any feature, present or future, without
listing how each could leak. It is the core of A2's Q9 test.

### Label blindness
**Plain:** shuffle the answer sheet; the questions must not change.
**Technical:** permute the click labels across impressions. Every feature column must be
identical, and only the label column may move.

### Equivalent mutant
**Plain:** a deliberate "bug" that turns out not to be a bug, because something else
already guarantees the right answer.
**Technical:** a mutation that cannot change behaviour. Example: removing the window
clamp in `exposure_shares`, which the padded key span already covers. It is proven
equivalent by removing *both* guards and seeing the test fail.

### Rack
**Plain:** the shelf of newspapers a reader is actually shown at one moment — not
everything the shop stocks, just what is in front of them.
**Technical:** the candidate list of a single impression (`article_ids_inview` in EB-NeRD,
the `impressions` field in MIND). It is what both leaderboards ask us to order. Every
metric we report is computed inside one rack and then averaged across racks, never across
candidates from different impressions.

### Rack-relative (within-impression) normalisation
**Plain:** the shopkeeper does not care that this paper sold 400 copies; they care that it
outsold the four papers next to it on the same shelf this morning.
**Technical:** replacing (or, in D42, accompanying) a feature by its position *within its
own impression's candidate list*. Two forms are used: `f_pct`, the percentile rank in
[0, 1]; and `f_z`, `(f − rack mean) / rack standard deviation`. It exists because a
gradient-boosted tree splits on absolute values, so a feature whose *level* drifts between
impressions — measured: 73.8% of `bm25`'s variance on MIND is between-rack — gives the
tree a threshold that means different things in different racks.

### Between-rack variance share
**Plain:** how much of a number's wobble is "this shelf is different from that shelf"
rather than "this paper is different from the one beside it".
**Technical:** the variance of the per-impression means, weighted by rack size, divided by
the feature's total variance. Near 1 means almost all the signal is a level that shifts
between impressions and almost none distinguishes candidates within one; near 0 means the
opposite. It is the diagnostic that chose which features D42 normalises.

### Percentile rank vs z-score
**Plain:** "third of eight" versus "two lengths clear of the field".
**Technical:** the percentile rank is scale-free and unaffected by an outlier, but discards
magnitude — a runaway leader and a photo finish both give 1.0. The z-score keeps the
margin but lets one extreme candidate compress everything else. Neither is strictly
better, which is why D42 supplies both and lets the model choose.

### Serving tier (corpus / per-user / per-request)
**Plain:** things the shop owns, things it knows about you, and things it has to do while
you stand at the counter.
**Technical:** the three lifetimes D44 separates. **Corpus-lifetime** state (embeddings,
indexes, first-seen times) is built once at start-up and shared by every request.
**Per-user** state (decayed profile vectors, category shares, the BM25 query) depends only
on click history, so it can be built in a nightly batch job — which trades profile
freshness for request latency. **Per-request** work is what is left once those two are
precomputed, and is the only part that belongs on the latency path.

### Word-level vs sentence-level news encoder
**Plain:** reading the headline word by word and deciding what each word means from its
neighbours, versus being handed somebody else's one-line summary of it.
**Technical:** NRMS's own news encoder runs multi-head self-attention over the title's
GloVe word vectors and then pools them with additive attention, so a word is encoded in
the context of the other words. D39 substituted a learned projection of a pre-pooled
384-dim sentence embedding, which can rescale that summary but cannot revisit it. D43
restores the paper's version.
