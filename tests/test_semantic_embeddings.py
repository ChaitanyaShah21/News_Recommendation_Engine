"""R10 adversarial tests for semantic.py (Q3.1).

Constructed data throughout, not sampled. The real corpus happens to contain no
article whose title AND abstract are both empty, and happens to contain no
duplicate ids - so testing only against it would prove nothing about either case.

A stub model stands in for sentence-transformers: these tests are about our
plumbing (null handling, alignment, dtype, normalisation invariants), not about
whether BERT works. It also keeps the suite offline and instant.
"""

import hashlib

import numpy as np
import polars as pl
import pytest

from newsrec.retrieval.semantic import (
    EMBEDDING_DIM,
    build_article_embeddings,
    build_document_texts,
    embed_texts,
    load_article_embeddings,
)


class StubModel:
    """Deterministic fake encoder: same text always gives the same unit vector.

    Determinism is the point - it lets a test assert that vector i really belongs
    to article i, which a random stub could not.
    """

    def __init__(self):
        self.seen_batch_sizes = []

    def encode(self, texts, batch_size=64, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False):
        self.seen_batch_sizes.append(batch_size)
        out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v) if normalize_embeddings else v
        return out


def frame(rows):
    return pl.DataFrame(
        rows,
        schema={"dataset": pl.String, "article_id": pl.String,
                "title": pl.String, "abstract": pl.String},
        orient="row",
    )


# --------------------------------------------------------------- null handling

def test_null_abstract_does_not_null_the_whole_text():
    """MIND stores a missing abstract as null (3,415 articles).

    pl.concat_str propagates null, so an unguarded version turns the entire text
    into null and the article is embedded as an empty string - silently, with a
    perfectly valid-looking vector coming back out.
    """
    out = build_document_texts(frame([("mind", "mind:N1", "Trump wins Ohio", None)]))
    assert out["text"][0] == "Trump wins Ohio"


def test_blank_abstract_is_handled_too():
    """EB-NeRD stores a missing abstract as "" (803 articles), not null.

    A fix that only guards against null would leave a trailing separator here.
    """
    out = build_document_texts(frame([("ebnerd", "ebnerd:1", "Prins Harry tvunget til dna-test", "")]))
    assert out["text"][0] == "Prins Harry tvunget til dna-test"


def test_null_title_still_produces_usable_text():
    out = build_document_texts(frame([("mind", "mind:N1", None, "some abstract")]))
    assert out["text"][0] == "some abstract"


def test_both_fields_empty_yields_empty_string_not_null():
    """No such article exists in our corpus today. That is not a guarantee."""
    out = build_document_texts(frame([("mind", "mind:N1", None, "")]))
    assert out["text"][0] == ""
    assert out["text"].null_count() == 0


def test_danish_characters_survive():
    """Mirrors D11's tokeniser test - the failure mode that silently mangles Danish."""
    title = "Rådden kørsel på blå plader"
    out = build_document_texts(frame([("ebnerd", "ebnerd:1", title, "Ekstra Bladet afslører")]))
    assert title in out["text"][0]
    assert "å" in out["text"][0] and "ø" in out["text"][0]


# ------------------------------------------------------------------- alignment

def test_vectors_are_aligned_with_their_own_article_ids():
    """The failure D22 chose option C to make impossible - checked anyway."""
    rows = [("mind", f"mind:N{i}", f"headline number {i}", f"abstract {i}") for i in range(25)]
    stub = StubModel()
    out = build_article_embeddings(frame(rows), model=stub)

    for i in range(25):
        row = out.filter(pl.col("article_id") == f"mind:N{i}")
        expected = stub.encode([f"headline number {i} abstract {i}"])[0]
        np.testing.assert_allclose(np.array(row["embedding"][0]), expected, rtol=1e-6)


def test_batch_size_does_not_change_the_output():
    rows = [("mind", f"mind:N{i}", f"title {i}", f"abs {i}") for i in range(70)]
    a = build_article_embeddings(frame(rows), model=StubModel(), batch_size=8)
    b = build_article_embeddings(frame(rows), model=StubModel(), batch_size=64)
    np.testing.assert_allclose(
        np.array(a["embedding"].explode().to_numpy(), dtype=np.float32),
        np.array(b["embedding"].explode().to_numpy(), dtype=np.float32),
    )


# ----------------------------------------------------------------- loud errors

def test_duplicate_article_ids_raise():
    rows = [("mind", "mind:N1", "a", "b"), ("mind", "mind:N1", "c", "d")]
    with pytest.raises(ValueError, match="duplicate article_id"):
        build_article_embeddings(frame(rows), model=StubModel())


def test_model_returning_wrong_dimension_raises():
    class BadModel:
        def encode(self, texts, **kw):
            return np.zeros((len(texts), 128), dtype=np.float32)

    with pytest.raises(ValueError, match="expected"):
        build_article_embeddings(frame([("mind", "mind:N1", "a", "b")]), model=BadModel())


# ------------------------------------------------------------------ edge sizes

def test_empty_frame_returns_correct_schema_not_a_crash():
    out = build_article_embeddings(frame([]), model=StubModel())
    assert out.height == 0
    assert out.columns == ["dataset", "article_id", "embedding"]


def test_empty_text_list_returns_correctly_shaped_array():
    v = embed_texts([], model=StubModel())
    assert v.shape == (0, EMBEDDING_DIM)
    assert v.dtype == np.float32


# ------------------------------------------------------------------ round trip

def test_parquet_round_trip_preserves_float32_and_values(tmp_path):
    """Phase 1's Float32/Float64 concat bug is why dtype is asserted, not assumed."""
    rows = [("mind", f"mind:N{i}", f"t{i}", f"a{i}") for i in range(10)]
    rows += [("ebnerd", f"ebnerd:{i}", f"dansk {i}", f"resume {i}") for i in range(5)]
    built = build_article_embeddings(frame(rows), model=StubModel())

    path = tmp_path / "embeddings.parquet"
    built.write_parquet(path)

    ids, matrix = load_article_embeddings(path)
    assert ids == built["article_id"].to_list()
    assert matrix.shape == (15, EMBEDDING_DIM)
    assert matrix.dtype == np.float32
    assert matrix.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(
        matrix, np.array(built["embedding"].explode().to_numpy(),
                         dtype=np.float32).reshape(15, EMBEDDING_DIM), rtol=1e-6)


def test_round_trip_dataset_filter_keeps_alignment(tmp_path):
    rows = [("mind", f"mind:N{i}", f"t{i}", f"a{i}") for i in range(10)]
    rows += [("ebnerd", f"ebnerd:{i}", f"dansk {i}", f"resume {i}") for i in range(5)]
    built = build_article_embeddings(frame(rows), model=StubModel())
    path = tmp_path / "embeddings.parquet"
    built.write_parquet(path)

    ids, matrix = load_article_embeddings(path, dataset="ebnerd")
    assert len(ids) == 5 and matrix.shape == (5, EMBEDDING_DIM)
    assert all(i.startswith("ebnerd:") for i in ids)

    stub = StubModel()
    np.testing.assert_allclose(matrix[0], stub.encode(["dansk 0 resume 0"])[0], rtol=1e-6)


def test_loading_unnormalised_vectors_raises(tmp_path):
    """The invariant every later dot product depends on, checked at the boundary.

    Without this, un-normalised vectors make `matrix @ user_vec` silently stop
    being cosine similarity, and every recall@K computed from it is wrong with
    no error anywhere.
    """
    bad = pl.DataFrame({
        "dataset": ["mind"],
        "article_id": ["mind:N1"],
        "embedding": [[3.0] * EMBEDDING_DIM],  # norm is 3*sqrt(384), nowhere near 1
    }, schema={"dataset": pl.String, "article_id": pl.String,
               "embedding": pl.List(pl.Float32)})
    path = tmp_path / "bad.parquet"
    bad.write_parquet(path)

    with pytest.raises(ValueError, match="not L2-normalised"):
        load_article_embeddings(path)


def test_load_of_empty_selection_returns_empty_matrix(tmp_path):
    rows = [("mind", "mind:N1", "t", "a")]
    built = build_article_embeddings(frame(rows), model=StubModel())
    path = tmp_path / "e.parquet"
    built.write_parquet(path)

    ids, matrix = load_article_embeddings(path, dataset="ebnerd")
    assert ids == [] and matrix.shape == (0, EMBEDDING_DIM)
