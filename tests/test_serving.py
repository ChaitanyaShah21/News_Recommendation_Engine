"""D44: the online scoring path must agree with the batch feature builder.

This is the only reason a second implementation is safe to serve. `serving.py`
exists to remove Polars from the request path, NOT to change a single feature
definition - so the test is equality against `assemble.build_rows` on real
impressions, feature by feature, including the null pattern.

Nulls are the part most likely to go wrong and least likely to be noticed. The
batch path carries them as Polars nulls; the request path carries them as NaN,
because that is what `gbdt.feature_matrix` converts them to and what LightGBM
reads as missing. The comparison is therefore done AFTER `feature_matrix`, on
the two float32 matrices the model would actually be handed - which is the
thing that has to match, rather than an intermediate that does not ship.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from newsrec import serving
from newsrec.features import assemble
from newsrec.rank import gbdt
from newsrec.retrieval import semantic

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
requires_store = pytest.mark.skipif(
    not (PROCESSED / "impressions.parquet").exists(), reason="feature store not built")
N_REQUESTS = 25


@pytest.fixture(scope="module")
def rig():
    """Built once for both datasets' tests: the context costs ~35-75 s."""
    made = {}

    def make(dataset):
        if dataset in made:
            return made[dataset]
        imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
            pl.col("dataset") == dataset)
        hist = pl.read_parquet(PROCESSED / "history.parquet").filter(
            (pl.col("dataset") == dataset) & (pl.col("split") == "test"))
        arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(
            pl.col("dataset") == dataset)
        ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet",
                                                    dataset=dataset)
        ctx = assemble.build_context(dataset, imps, arts, ids, emb)
        index = serving.ServingIndex.build(ctx)
        target = imps.filter(pl.col("split") == "test").sample(N_REQUESTS, seed=11)
        hist = hist.filter(pl.col("user_id").is_in(target["user_id"].implode()))
        profiles = serving.UserProfile.build_many(index, ctx, hist)
        made[dataset] = (ctx, index, target, hist, profiles)
        return made[dataset]

    return make


def _extra(dataset, row, batch_ft):
    """EB-NeRD's impression-level columns come from the request in production.
    Here they are read back from the batch table for the same impression, so
    the test compares the candidate-level arithmetic rather than re-deriving
    session bookkeeping that `features/session.py` already owns and tests."""
    if dataset != "ebnerd":
        return {}
    names = [f for f in assemble.FEATURES[dataset] if f not in assemble.FEATURES["mind"]]
    return {n: batch_ft[n][0] for n in names}


@requires_store
@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_serving_features_equal_the_batch_builder(rig, dataset):
    ctx, index, target, hist, profiles = rig(dataset)
    checked = 0
    for row in target.iter_rows(named=True):
        one = target.filter(pl.col("impression_id") == row["impression_id"])
        batch = assemble.build_rows(ctx, one, hist.filter(pl.col("user_id") == row["user_id"]))
        if batch.is_empty():
            continue
        online = serving.features(index, profiles.get(row["user_id"]),
                                  batch["article_id"].to_list(), row["timestamp"],
                                  _extra(dataset, row, batch))
        want = gbdt.feature_matrix(batch, index.features)
        assert online.shape == want.shape
        for j, name in enumerate(index.features):
            a, b = online[:, j], want[:, j]
            assert np.array_equal(np.isnan(a), np.isnan(b)), (
                f"{dataset} {name}: null pattern differs")
            ok = ~np.isnan(b)
            assert np.allclose(a[ok], b[ok], rtol=1e-4, atol=1e-5), (
                f"{dataset} {name}: values differ, max |diff| "
                f"{np.abs(a[ok] - b[ok]).max() if ok.any() else 0}")
        checked += 1
    assert checked >= 5, f"only {checked} requests compared"


@requires_store
@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_a_cold_user_gets_nulls_not_zeros(rig, dataset):
    """A user with no profile must produce null cos_*, cat_share_* and bm25 -
    the same fact the batch path records. Zeros would say "we looked and found
    nothing", which is a different and false claim."""
    ctx, index, target, hist, profiles = rig(dataset)
    row = target.row(0, named=True)
    one = target.filter(pl.col("impression_id") == row["impression_id"])
    batch = assemble.build_rows(ctx, one, hist.head(0))
    online = serving.features(index, None, batch["article_id"].to_list(), row["timestamp"],
                              _extra(dataset, row, batch))
    for name in ("cos_inf", "cat_share_inf", "bm25", "cos_inf_pct", "bm25_z"):
        col = online[:, index.features.index(name)]
        assert np.isnan(col).all(), f"{name} is not null for a user with no profile"
    # The features that do not need a user must still be real numbers.
    for name in ("freshness_hours", "exposure_share_1h", "freshness_hours_pct"):
        col = online[:, index.features.index(name)]
        assert np.isfinite(col).all(), f"{name} became null without a user"


@requires_store
def test_an_unknown_article_is_refused_rather_than_scored(rig):
    ctx, index, target, hist, profiles = rig("mind")
    row = target.row(0, named=True)
    with pytest.raises(ValueError, match="no embedding"):
        serving.features(index, None, ["mind:NOT_AN_ARTICLE"], row["timestamp"])


def test_rack_helper_matches_polars_on_constructed_edge_cases():
    """The NaN-native rack normalisation against the Polars one it mirrors,
    on the cases `test_rack_features.py` pins: ties, a constant rack, a single
    finite value, and nulls mixed with values."""
    from newsrec.features import rack as rk
    cases = [[1.0, 2.0, 3.0], [7.0, 7.0, 7.0], [0.0, 0.0, 0.0], [5.0],
             [1.0, np.nan, 3.0, 5.0], [np.nan, np.nan], [2.0, 2.0, 9.0]]
    for xs in cases:
        x = np.asarray(xs, dtype=np.float64)
        p, z = serving._rack(x)
        df = pl.DataFrame({"impression_id": ["a"] * len(x),
                           "bm25": pl.Series(x).fill_nan(None)})
        out = rk.add_rack_features(df, base=["bm25"])
        for got, col in ((p, "bm25_pct"), (z, "bm25_z")):
            want = np.array([np.nan if v is None else v for v in out[col].to_list()])
            assert np.array_equal(np.isnan(got), np.isnan(want)), (xs, col)
            m = ~np.isnan(want)
            assert np.allclose(got[m], want[m], atol=1e-6), (xs, col, got, want)
