"""BM25 (Best Match 25) lexical retrieval - the document side (Q2.1).

This module builds the inverted index. It does not score queries; that is the
query side, which lives separately because the two have genuinely different
lifetimes: the index is built once per dataset and reused for every user,
while queries are constructed per user from click history (D12).

The index is a sparse document-term matrix. In compressed-sparse-column form
(`BM25Index.term_doc`) that matrix *is* an inverted index: for each term it
stores a contiguous run of the documents containing it - a posting list - which
is exactly why scoring never has to touch a document that shares no word with
the query. See D14 in ARCHITECTURE.md for why this is implemented rather than
imported.

The stored weights are the *document side* of BM25 pre-multiplied:

    weight(t, D) = IDF(t) * [ f(t,D) * (k1 + 1) ]
                            -------------------------------------
                            [ f(t,D) + k1 * (1 - b + b*|D|/avgdl) ]

so that scoring a query later is a single sparse matrix product rather than a
formula evaluated per (document, term) pair. Every symbol is defined in
GLOSSARY.md's BM25 entry.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

# D11: Unicode-aware tokenisation. `\w` in Python 3 matches Unicode letters and
# digits by default, so `[^\W_]` is "word character but not underscore" - it
# keeps æ/ø/å intact. The naive `[a-z0-9]+` would silently turn the real
# EB-NeRD title "Rådden kørsel på blå plader" into "r dden k rsel p bl plader" -
# five corrupted tokens, no error raised. tests/test_bm25_index.py asserts
# against that exact title.
_TOKEN_RE = re.compile(r"[^\W_]+")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str | None) -> list[str]:
    """Lowercase `text` and split it into Unicode word tokens.

    Returns an empty list for null or blank input rather than raising or
    producing a spurious token - 5.2% of MIND abstracts and 6.8% of EB-NeRD
    abstracts are missing or blank, so this path is taken tens of thousands of
    times per build, not just in tests.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """An inverted index over one dataset's articles, with BM25 document-side
    weights precomputed.

    Attributes:
        article_ids: row i of the matrices is `article_ids[i]`. This is the only
            thing mapping matrix rows back to real articles, so it must stay in
            step with the matrices - it is built from the same DataFrame in the
            same order and never reordered afterwards.
        vocab: term -> column index.
        doc_term: (n_docs x n_terms) CSR matrix of BM25 document-side weights.
            Row-compressed: fast to ask "what is in document i".
        term_doc: the same matrix in CSC form. Column-compressed: fast to ask
            "which documents contain term j". This is the inverted index.
        idf: idf[j] is the inverse document frequency of the term with column
            index j. Already folded into the weights; kept for inspection and
            for the design note.
        doc_freq: doc_freq[j] is how many documents contain that term.
        doc_len: doc_len[i] is the token count of document i, counted over *all*
            tokens - including any dropped by `max_df`, since document length is
            a property of the document, not of the retained vocabulary.
        avgdl: mean of doc_len over the corpus.
    """

    article_ids: list[str]
    vocab: dict[str, int]
    doc_term: sparse.csr_matrix
    term_doc: sparse.csc_matrix
    idf: np.ndarray
    doc_freq: np.ndarray
    doc_len: np.ndarray
    avgdl: float
    k1: float
    b: float

    @property
    def n_docs(self) -> int:
        return len(self.article_ids)

    @property
    def n_terms(self) -> int:
        return len(self.vocab)

    def postings(self, term: str) -> list[tuple[str, float]]:
        """The posting list for `term`: every article containing it, with that
        article's precomputed BM25 weight for the term, highest first.

        Empty list for an unknown term. This is a teaching/inspection helper -
        scoring never calls it, because scoring reads all posting lists at once
        via the matrix product.
        """
        column = self.vocab.get(term)
        if column is None:
            return []
        start, end = self.term_doc.indptr[column], self.term_doc.indptr[column + 1]
        rows = self.term_doc.indices[start:end]
        weights = self.term_doc.data[start:end]
        pairs = [(self.article_ids[r], float(w)) for r, w in zip(rows, weights)]
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)


def build_index(
    articles: pl.DataFrame,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    max_df: float | None = None,
) -> BM25Index:
    """Build a BM25 inverted index over one dataset's article titles + abstracts.

    Args:
        articles: rows of the unified `articles` table, already filtered to a
            single dataset. Needs columns `dataset`, `article_id`, `title`,
            `abstract`.
        k1: term-frequency saturation. The per-term contribution can never
            exceed `k1 + 1` no matter how often the term repeats (D13).
        b: length-normalisation strength, 0 (ignore length) to 1 (fully
            proportional to length) (D13).
        max_df: if set, drop terms appearing in more than this *fraction* of
            documents - the stopword-free speed knob from D11. Unused by default.

    Raises:
        ValueError: on an empty corpus, on more than one dataset in `articles`,
            on duplicate article_ids, or on a corpus with no tokens at all.
    """
    if articles.height == 0:
        raise ValueError("cannot build an index over zero articles")

    # Two indexes, one per dataset (D11/D14): MIND queries must never retrieve
    # Danish articles, and IDF is meaningless across two corpora whose
    # vocabularies barely overlap - `N` and `n(t)` would be measured over a
    # corpus that no query is ever asking about.
    datasets = articles.get_column("dataset").unique().to_list()
    if len(datasets) != 1:
        raise ValueError(
            f"build_index expects exactly one dataset, got {sorted(datasets)}; "
            "filter with .filter(pl.col('dataset') == ...) first"
        )

    article_ids = articles.get_column("article_id").to_list()
    if len(set(article_ids)) != len(article_ids):
        raise ValueError(
            "duplicate article_id in `articles`; the row -> article mapping "
            "would be ambiguous and retrieved ids would be silently wrong"
        )

    # Q2.1 says "over article titles and abstracts". A null abstract must
    # contribute nothing rather than the string "None", which fill_null("")
    # guarantees and str() would not.
    texts = (
        articles.select(
            (pl.col("title").fill_null("") + " " + pl.col("abstract").fill_null(""))
        )
        .to_series()
        .to_list()
    )

    # --- Pass 1: tokenise, count term frequencies per document, count document
    # frequencies across the corpus. ---
    per_doc: list[Counter[str]] = []
    doc_freq_counter: Counter[str] = Counter()
    doc_len = np.zeros(len(texts), dtype=np.float32)

    for i, text in enumerate(texts):
        tokens = tokenize(text)
        doc_len[i] = len(tokens)
        counts = Counter(tokens)
        per_doc.append(counts)
        # .keys(), not the counts themselves: document frequency asks "in how
        # many documents", never "how many times".
        doc_freq_counter.update(counts.keys())

    avgdl = float(doc_len.mean())
    if avgdl == 0.0:
        raise ValueError(
            "every article tokenised to zero tokens; check the tokeniser "
            "pattern against this dataset's alphabet"
        )

    # --- Build the vocabulary. ---
    n_docs = len(texts)
    kept = doc_freq_counter.keys()
    if max_df is not None:
        cutoff = max_df * n_docs
        kept = [term for term, df in doc_freq_counter.items() if df <= cutoff]
    # Sorted, so the same corpus always produces byte-identical column indices -
    # otherwise two runs could disagree on tie-breaking between equal scores.
    vocab = {term: column for column, term in enumerate(sorted(kept))}
    if not vocab:
        raise ValueError("vocabulary is empty; max_df may have dropped every term")

    doc_freq = np.zeros(len(vocab), dtype=np.int64)
    for term, column in vocab.items():
        doc_freq[column] = doc_freq_counter[term]

    # --- IDF. The +0.5 terms are smoothing (no division by zero at the
    # extremes); the +1 inside the log makes IDF non-negative, so a document is
    # never *penalised* for containing a very common query term the way older
    # BM25 variants allowed. ---
    idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).astype(np.float32)

    # --- Pass 2: lay the term frequencies out as CSR arrays. `indptr[i]:
    # indptr[i+1]` delimits document i's slice of `indices`/`data`. ---
    indptr = np.zeros(n_docs + 1, dtype=np.int64)
    indices_list: list[int] = []
    data_list: list[float] = []
    for i, counts in enumerate(per_doc):
        for term, term_freq in counts.items():
            column = vocab.get(term)
            if column is None:  # dropped by max_df
                continue
            indices_list.append(column)
            data_list.append(term_freq)
        indptr[i + 1] = len(indices_list)

    indices = np.asarray(indices_list, dtype=np.int32)
    term_freq = np.asarray(data_list, dtype=np.float32)

    # --- Apply the BM25 document side to every non-zero. ---
    # Each non-zero needs its own document's length. np.diff(indptr) is how many
    # non-zeros each document owns, so repeating doc_len by that count lines the
    # lengths up with `indices`/`term_freq` element for element.
    doc_len_per_nonzero = np.repeat(doc_len, np.diff(indptr))
    denominator = term_freq + k1 * (1.0 - b + b * doc_len_per_nonzero / avgdl)
    weights = (term_freq * (k1 + 1.0) / denominator) * idf[indices]

    doc_term = sparse.csr_matrix(
        (weights, indices, indptr), shape=(n_docs, len(vocab)), dtype=np.float32
    )
    # Counter preserves first-seen order, so a document's column indices come out
    # in the order its words happened to appear - valid CSR, but not scipy's
    # canonical form, which some operations assume. Sorting once here costs
    # nothing and keeps every downstream operation on the well-trodden path.
    doc_term.sort_indices()

    return BM25Index(
        article_ids=article_ids,
        vocab=vocab,
        doc_term=doc_term,
        term_doc=doc_term.tocsc(),
        idf=idf,
        doc_freq=doc_freq,
        doc_len=doc_len,
        avgdl=avgdl,
        k1=k1,
        b=b,
    )
