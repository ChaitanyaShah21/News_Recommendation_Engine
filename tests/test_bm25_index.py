"""Adversarial tests for the BM25 inverted index (R10).

These use deliberately constructed data, not samples of the real corpus. The
point is to exercise cases the real files may not happen to contain - a blank
abstract next to a blank title, a duplicate id, a term in every document - and
to pin the arithmetic against numbers worked out by hand rather than against
whatever the code currently returns.
"""

import math

import numpy as np
import polars as pl
import pytest

from newsrec.retrieval import bm25


def _articles(rows: list[tuple[str, str | None, str | None]], dataset: str = "mind"):
    """Build a minimal `articles`-shaped frame from (id, title, abstract)."""
    return pl.DataFrame(
        {
            "dataset": [dataset] * len(rows),
            "article_id": [r[0] for r in rows],
            "title": [r[1] for r in rows],
            "abstract": [r[2] for r in rows],
        },
        schema={
            "dataset": pl.Utf8,
            "article_id": pl.Utf8,
            "title": pl.Utf8,
            "abstract": pl.Utf8,
        },
    )


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------


def test_danish_letters_survive_tokenisation():
    """The failure D11 was written to prevent: a non-Unicode-aware pattern
    shreds æ/ø/å silently. This is a real EB-NeRD title.
    """
    assert bm25.tokenize("Rådden kørsel på blå plader") == [
        "rådden",
        "kørsel",
        "på",
        "blå",
        "plader",
    ]


def test_danish_uppercase_folds_correctly():
    """EB-NeRD abstracts open with shouted section labels ("ISHOCKEY:"), so
    case folding has to work on non-ASCII letters, not just ASCII.
    """
    assert bm25.tokenize("ÅRHUS Æblet ØSTJYLLAND") == ["århus", "æblet", "østjylland"]


def test_null_and_blank_text_produce_no_tokens():
    """5.2% of MIND abstracts are null. A `str(None)` bug would inject a
    spurious "none" term into ~3,400 documents and quietly make it a real,
    scoreable word.
    """
    assert bm25.tokenize(None) == []
    assert bm25.tokenize("") == []
    assert bm25.tokenize("   ") == []
    assert "none" not in bm25.tokenize(None)


def test_punctuation_and_underscores_split_tokens():
    assert bm25.tokenize("COVID-19: what's next_now?") == [
        "covid",
        "19",
        "what",
        "s",
        "next",
        "now",
    ]


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------


def test_mixed_datasets_rejected():
    """Building one index over both corpora would compute IDF over a vocabulary
    no single query ever asks about, and let a MIND query retrieve Danish text.
    """
    frame = pl.concat(
        [
            _articles([("mind:A", "budget vote", None)], dataset="mind"),
            _articles([("ebnerd:1", "blå plader", None)], dataset="ebnerd"),
        ]
    )
    with pytest.raises(ValueError, match="exactly one dataset"):
        bm25.build_index(frame)


def test_duplicate_article_ids_rejected():
    """Row -> article_id is the only mapping back to reality. A duplicate makes
    retrieved ids ambiguous without anything raising downstream.
    """
    frame = _articles([("mind:A", "budget", None), ("mind:A", "election", None)])
    with pytest.raises(ValueError, match="duplicate article_id"):
        bm25.build_index(frame)


def test_empty_corpus_rejected():
    with pytest.raises(ValueError, match="zero articles"):
        bm25.build_index(_articles([]))


def test_corpus_with_no_tokens_rejected():
    """avgdl = 0 would divide by zero inside the length normalisation. Fail
    loudly instead - this is what a broken tokeniser pattern looks like.
    """
    frame = _articles([("mind:A", "", None), ("mind:B", "   ", "")])
    with pytest.raises(ValueError, match="zero tokens"):
        bm25.build_index(frame)


def test_document_with_no_tokens_is_kept_but_unreachable():
    """A single blank document among real ones must not shift row alignment.
    It should occupy its row, own zero non-zeros, and never appear in a
    posting list.
    """
    frame = _articles(
        [("mind:A", "budget", None), ("mind:B", "", None), ("mind:C", "budget", None)]
    )
    index = bm25.build_index(frame)

    assert index.article_ids == ["mind:A", "mind:B", "mind:C"]
    assert index.doc_len[1] == 0.0
    assert index.doc_term[1].nnz == 0
    assert [aid for aid, _ in index.postings("budget")] == ["mind:A", "mind:C"]


# --------------------------------------------------------------------------
# Arithmetic, checked against hand computation
# --------------------------------------------------------------------------


def test_weights_match_hand_computed_bm25():
    """Three documents, one shared term, hand-worked numbers.

    Corpus: A = "tariff tariff tariff" (3 tokens), B = 12 tokens with "tariff"
    once, C = "unrelated words here" (3 tokens, no "tariff").
    N = 3, n(tariff) = 2, so IDF = ln((3-2+0.5)/(2+0.5) + 1) = ln(1.6).
    avgdl = (3 + 12 + 3)/3 = 6.
    """
    b_filler = " ".join(f"w{i}" for i in range(11))  # 11 distinct filler tokens
    frame = _articles(
        [
            ("mind:A", "tariff tariff tariff", None),
            ("mind:B", f"tariff {b_filler}", None),
            ("mind:C", "unrelated words here", None),
        ]
    )
    k1, b = 1.5, 0.75
    index = bm25.build_index(frame, k1=k1, b=b)

    assert index.avgdl == pytest.approx(6.0)
    assert list(index.doc_len) == [3.0, 12.0, 3.0]

    column = index.vocab["tariff"]
    assert index.doc_freq[column] == 2
    expected_idf = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1.0)
    assert index.idf[column] == pytest.approx(expected_idf, rel=1e-5)

    def expected(term_freq: float, doc_length: float) -> float:
        denominator = term_freq + k1 * (1 - b + b * doc_length / 6.0)
        return term_freq * (k1 + 1) / denominator * expected_idf

    assert index.doc_term[0, column] == pytest.approx(expected(3, 3), rel=1e-5)
    assert index.doc_term[1, column] == pytest.approx(expected(1, 12), rel=1e-5)
    assert index.doc_term[2, column] == 0.0


def test_short_document_beats_long_one_at_equal_term_frequency():
    """The Q1 comprehension-check case, as an executable assertion: same term
    frequency, different lengths - the length bracket must decide it.
    """
    filler = " ".join(f"w{i}" for i in range(50))
    frame = _articles(
        [
            ("mind:short", "tariff tariff tariff", None),
            ("mind:long", f"tariff tariff tariff {filler}", None),
        ]
    )
    index = bm25.build_index(frame)
    column = index.vocab["tariff"]
    assert index.doc_term[0, column] > index.doc_term[1, column]


def test_term_frequency_saturates_below_k1_plus_one():
    """Repetition can never buy more than k1+1 times the IDF, however extreme.
    Guards against accidentally reintroducing linear term-frequency scoring.
    """
    k1 = 1.5
    frame = _articles(
        [
            ("mind:A", "tariff " * 1000, None),
            ("mind:B", "tariff", None),
            ("mind:C", "other", None),
        ]
    )
    index = bm25.build_index(frame, k1=k1, b=0.0)  # b=0 removes length effects
    column = index.vocab["tariff"]
    ceiling = (k1 + 1) * index.idf[column]
    assert index.doc_term[0, column] < ceiling
    assert index.doc_term[0, column] < 3 * index.doc_term[1, column]


def test_idf_never_negative_even_for_a_term_in_every_document():
    """Older BM25 variants let a term present in >half the corpus score
    negative, so a document could be *penalised* for containing a query word.
    The +1 inside the log rules that out; assert it holds at the extreme.
    """
    frame = _articles([(f"mind:{i}", "the budget", None) for i in range(50)])
    index = bm25.build_index(frame)
    assert index.idf.min() >= 0.0
    assert index.doc_term.data.min() >= 0.0


# --------------------------------------------------------------------------
# Structure and reproducibility
# --------------------------------------------------------------------------


def test_csr_and_csc_views_agree():
    """term_doc is the inverted index; doc_term is the same data row-wise. If
    they ever disagree, posting lists and scores describe different corpora.
    """
    frame = _articles(
        [
            ("mind:A", "budget vote", "the budget passes"),
            ("mind:B", "election night", None),
            ("mind:C", "vote counting", "vote vote"),
        ]
    )
    index = bm25.build_index(frame)
    assert np.allclose(index.doc_term.toarray(), index.term_doc.toarray())


def test_postings_returns_only_documents_containing_the_term():
    frame = _articles(
        [
            ("mind:A", "budget vote", None),
            ("mind:B", "election night", None),
            ("mind:C", "vote counting", None),
        ]
    )
    index = bm25.build_index(frame)
    assert {aid for aid, _ in index.postings("vote")} == {"mind:A", "mind:C"}
    assert index.postings("nonexistent") == []


def test_index_build_is_deterministic():
    """Two builds of the same corpus must produce identical column indices, or
    tie-breaking between equal scores could differ between runs and make the
    reported recall@K irreproducible.
    """
    frame = _articles(
        [("mind:A", "budget vote", None), ("mind:B", "vote election", None)]
    )
    first, second = bm25.build_index(frame), bm25.build_index(frame)
    assert first.vocab == second.vocab
    assert np.array_equal(first.doc_term.toarray(), second.doc_term.toarray())


def test_max_df_drops_ubiquitous_terms_without_changing_document_length():
    """The D11 speed knob. Document length must stay the *full* token count -
    it is a property of the document, not of the surviving vocabulary.
    """
    frame = _articles(
        [
            ("mind:A", "the budget", None),
            ("mind:B", "the election", None),
            ("mind:C", "the vote", None),
        ]
    )
    index = bm25.build_index(frame, max_df=0.9)  # "the" is in 100% of documents
    assert "the" not in index.vocab
    assert "budget" in index.vocab
    assert list(index.doc_len) == [2.0, 2.0, 2.0]
    assert index.postings("the") == []


def test_max_df_dropping_everything_fails_loudly():
    frame = _articles([("mind:A", "the", None), ("mind:B", "the", None)])
    with pytest.raises(ValueError, match="vocabulary is empty"):
        bm25.build_index(frame, max_df=0.5)
