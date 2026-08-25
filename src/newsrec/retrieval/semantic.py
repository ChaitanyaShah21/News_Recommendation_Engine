"""article embeddings: text -> dense vector, stored in the feature store.

The lexical counterpart of this file is `bm25.py`, and the two are deliberately
parallel. BM25 represents an article as a *sparse* vector over the vocabulary
(65,238 x 60,951, 2.36M non-zeros). Here an article becomes a *dense* vector of
384 real numbers where no dimension has a name. Same corpus, same text fields,
different geometry.

Decisions this file implements:
  D20 - one multilingual model for both datasets, so Q3.5 compares methods and
        not setups. `model_type: "bert"`, which is one of the two families Q3
        names explicitly.
  D22 - vectors live in their own `embeddings.parquet`, one row per article,
        each vector beside its own `article_id` so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# D20. Pinned as a constant rather than a default argument so there is exactly
# one place to change it if we ever swap to the XLM-RoBERTa variant.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# The model truncates past this many *subword* tokens, silently. Measured on our
# corpus (2026-08-25): 7.7% of MIND articles exceed it but only 1.37% of MIND's
# total tokens are lost, and 0.00% of EB-NeRD's. Accepted, not fixed - recorded
# here so the number is next to the code that incurs it.
MAX_SEQ_TOKENS = 128


def build_document_texts(articles: pl.DataFrame) -> pl.DataFrame:
    """Add a `text` column: title and abstract joined, mirroring BM25's document side.

    Returns the frame with `text` added, rather than a bare list, so that the
    text stays glued to its own `article_id` for the whole pipeline. Handing
    around two parallel lists is exactly how alignment bugs start.
    """
    # Both fields go through fill_null("") BEFORE concat_str. Two separate
    # reasons, and each dataset only exercises one of them:
    #   - MIND stores a missing abstract as null (3,415 articles). concat_str
    #     propagates null, so without this the whole text becomes null - the
    #     same trap as `.str.split()` on cold-start history in Phase 1.
    #   - EB-NeRD stores a missing abstract as "" (803 articles), which is
    #     already safe. A null-only guard would have missed these entirely.
    text = pl.concat_str(
        [pl.col("title").fill_null(""), pl.col("abstract").fill_null("")],
        separator=" ",
    ).str.strip_chars()

    return articles.with_columns(text.alias("text"))


def embed_texts(
    texts: list[str],
    *,
    model=None,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode `texts` into an (n, 384) float32 matrix of L2-normalised vectors.

    `model` is injected rather than constructed here so tests can pass a stub and
    never touch the network or the 470 MB download.
    """
    if model is None:  # imported lazily: torch takes ~10s to import
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, device="cpu")

    if not texts:
        # An empty batch must still return a correctly-shaped array, or the
        # caller's np.vstack / reshape blows up with a confusing message.
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        # D-note: normalising HERE, once, is what turns cosine similarity into a
        # plain dot product for every later retrieval call. Measured norm spread
        # on raw vectors was 1.71x - large enough to reorder rankings on its own.
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )

    # float32 deliberately, not float16 (59 MB saved against 901 GB free is not a
    # trade worth precision in a 384-term dot product), and not float64 - which
    # is what an unguarded round-trip drifted to in Phase 1's concat bug.
    return np.ascontiguousarray(vectors, dtype=np.float32)


def build_article_embeddings(
    articles: pl.DataFrame,
    *,
    model=None,
    batch_size: int = 64,
    show_progress: bool = False,
) -> pl.DataFrame:
    """Embed every article; return (dataset, article_id, embedding) - the D22 layout."""
    if articles.is_empty():
        return pl.DataFrame(
            schema={
                "dataset": pl.String,
                "article_id": pl.String,
                "embedding": pl.List(pl.Float32),
            }
        )

    n_dup = articles.height - articles["article_id"].n_unique()
    if n_dup:
        # Loud, not silent. A duplicate id means retrieval would return the same
        # article twice and burn a top-K slot, and the store is supposed to have
        # been deduplicated back in build.py (D9).
        raise ValueError(f"{n_dup} duplicate article_id values; store should be deduplicated")

    with_text = build_document_texts(articles)

    # Both columns pulled from the SAME frame in one pass, so row i of `texts`
    # and row i of `ids` are the same article by construction. This is the
    # alignment guarantee D22 chose option C to get.
    texts = with_text["text"].to_list()
    vectors = embed_texts(texts, model=model, batch_size=batch_size, show_progress=show_progress)

    if vectors.shape != (len(texts), EMBEDDING_DIM):
        raise ValueError(f"expected ({len(texts)}, {EMBEDDING_DIM}) vectors, got {vectors.shape}")

    return with_text.select(
        "dataset",
        "article_id",
        # from_numpy + implode is what turns an (n, 384) matrix into one Polars
        # list column of n rows. Done via a Series so the float32 dtype survives.
        pl.Series("embedding", vectors, dtype=pl.Array(pl.Float32, EMBEDDING_DIM)).cast(
            pl.List(pl.Float32)
        ),
    )


def load_article_embeddings(
    path, *, dataset: str | None = None
) -> tuple[list[str], np.ndarray]:
    """Read embeddings.parquet back into (article_ids, contiguous (n, 384) matrix).

    This is the conversion D22 accepted as option C's cost: Parquet gives a list
    column, retrieval needs a contiguous matrix to matrix-multiply against.
    """
    frame = pl.read_parquet(path)
    if dataset is not None:
        frame = frame.filter(pl.col("dataset") == dataset)

    ids = frame["article_id"].to_list()
    if not ids:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    # explode() flattens the list column into one long column of n*384 floats,
    # which reshapes into the matrix without ever building an array-of-arrays.
    flat = frame["embedding"].explode().to_numpy()
    if flat.size != len(ids) * EMBEDDING_DIM:
        raise ValueError(
            f"expected {len(ids) * EMBEDDING_DIM} floats for {len(ids)} articles, got {flat.size}"
        )

    matrix = np.ascontiguousarray(flat.reshape(len(ids), EMBEDDING_DIM), dtype=np.float32)

    # Assert the invariant every later dot product silently depends on. If these
    # vectors are not unit length, `matrix @ user_vec` is not cosine similarity
    # and every recall number computed from it is quietly wrong.
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"embeddings are not L2-normalised: norms range "
            f"[{norms.min():.4f}, {norms.max():.4f}]"
        )

    return ids, matrix
