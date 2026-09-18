"""assembly: the allowlist is enforced, and the Q9 quarantine holds."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from newsrec.features import assemble

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "newsrec"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
requires_store = pytest.mark.skipif(
    not (PROCESSED / "impressions.parquet").exists(), reason="feature store not built"
)


def test_the_quarantine_module_is_imported_nowhere_in_the_package():
    offenders = [
        p.relative_to(PACKAGE).as_posix()
        for p in PACKAGE.rglob("*.py")
        if p.name != "unavailable.py"
        and ("unavailable_features" in p.read_text() or "features.unavailable" in p.read_text()
             or "import unavailable" in p.read_text())
    ]
    assert offenders == [], f"serving-unavailable features referenced by {offenders}"


def test_no_allowlisted_feature_is_a_known_unavailable_one():
    from newsrec.features.unavailable import UNAVAILABLE
    for ds, feats in assemble.FEATURES.items():
        assert not set(feats) & set(UNAVAILABLE), ds
        assert assemble.LABEL not in feats


@requires_store
def test_real_table_has_exactly_the_allowlisted_columns():
    from newsrec.retrieval import semantic
    imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == "mind")
    target = imps.filter(pl.col("split") == "val").head(50)
    hist = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == "mind") & pl.col("user_id").is_in(target["user_id"].implode()))
    arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == "mind")
    ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset="mind")
    ft = assemble.build_feature_table("mind", target, imps, hist, arts, ids, emb)
    assert ft.columns == assemble.KEYS + [assemble.LABEL] + assemble.FEATURES["mind"]
    n_candidates = target.select(pl.col("candidate_article_ids").list.unique().list.len().sum()).item()
    assert ft.height == n_candidates
