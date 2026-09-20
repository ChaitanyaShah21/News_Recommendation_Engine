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
    """Both sanctioned sets are checked: D34's base allowlist and D42's
    base-plus-rack. A rack column is derived from a base one, so a quarantined
    column reaching RACK_BASE would smuggle the future in twice over."""
    from newsrec.features.unavailable import UNAVAILABLE
    for ds in assemble.FEATURES:
        for label, feats in (("base", assemble.FEATURES[ds]),
                             ("base+rack", assemble.ALL_FEATURES[ds])):
            assert not set(feats) & set(UNAVAILABLE), f"{ds} {label}"
            assert assemble.LABEL not in feats, f"{ds} {label}"


def test_every_rack_feature_is_derived_from_an_allowlisted_base_feature():
    """D42's columns are a function of columns already on the table. If a rack
    column ever named a base feature that is not itself allowlisted, the
    allowlist would have a hole the exact-columns test could not see."""
    from newsrec.features import rack as rk
    for ds in assemble.FEATURES:
        assert set(rk.RACK_BASE) <= set(assemble.FEATURES[ds]), ds
        for f in rk.RACK_BASE:
            for kind in rk.KINDS:
                assert f"{f}_{kind}" in assemble.RACK_FEATURES[ds]


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
    # Exact equality, not a subset: the point of D34's allowlist is that a
    # feature cannot join the model by accident, so an UNEXPECTED column must
    # fail here just as a missing one does. ALL_FEATURES is base + D42's rack
    # columns, and the order is asserted too because `gbdt.feature_matrix`
    # builds its matrix positionally.
    assert ft.columns == assemble.KEYS + [assemble.LABEL] + assemble.ALL_FEATURES["mind"]
    n_candidates = target.select(pl.col("candidate_article_ids").list.unique().list.len().sum()).item()
    assert ft.height == n_candidates
