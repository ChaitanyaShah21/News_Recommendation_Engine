"""the trained re-ranker - LightGBM with the `lambdarank` objective.

WHAT IS BEING TRAINED
One model per dataset. Input: the Q1 feature table, one row per
(impression, candidate). Output: one score per row. Only the ORDER of scores
within an impression matters; a score is never compared across impressions.

WHY LAMBDARANK, NOT A CLICK CLASSIFIER
A classifier (`binary` objective) is rewarded for getting every row's click
probability right in absolute terms, including across impressions. LambdaRank is
rewarded only for ordering each impression's clicked candidates above its
unclicked ones, with swaps near the top of the list weighted by how much they
would move nDCG. That is the thing both leaderboards grade.

LANDMINE 4 - GROUPS MUST BE CONTIGUOUS
LightGBM is told the impression structure only as a list of group SIZES
("the first 37 rows are one impression, the next 12 are the next, ..."). It
never sees impression ids. If rows of one impression are not adjacent - after a
shuffle, a join, a sort by article - the sizes still sum to the row count, so
training runs happily and learns to rank garbage. `group_sizes` therefore
checks contiguity from the ids themselves and raises, rather than trusting the
caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl

from newsrec.features.assemble import FEATURES, LABEL

SEED = 20260919

# Hyperparameters fixed in advance, not tuned (D36). Conservative defaults for a
# few million rows and 10-14 features: shallow-ish trees, a modest learning
# rate, and early stopping on the validation split deciding the tree count.
PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 20,
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 8,
}
MAX_ROUNDS = 1000
EARLY_STOP = 50


def group_sizes(impression_ids: pl.Series) -> np.ndarray:
    """Run lengths of consecutive equal ids, refusing non-contiguous groups.

    [a a a b b c] -> [3 2 1]. [a a b a] raises: `a` appears in two runs.
    """
    if impression_ids.is_empty():
        raise ValueError("no rows to group")
    runs = impression_ids.rle()  # struct {len, value}
    values = runs.struct.field("value")
    if values.n_unique() != values.len():
        dup = values.filter(values.is_duplicated()).head(3).to_list()
        raise ValueError(
            f"impression rows are not contiguous (Landmine 4): {dup} appear in more "
            f"than one run. Sort by impression, keeping candidate order, before training.")
    return runs.struct.field("len").to_numpy().astype(np.int64)


def feature_matrix(ft: pl.DataFrame, features: list[str]) -> np.ndarray:
    """float32 matrix in the given column order; nulls become NaN, which LightGBM
    treats as 'missing' and learns a direction for (a cold user has no cos_*)."""
    return ft.select([pl.col(f).cast(pl.Float32) for f in features]).to_numpy()


@dataclass
class Ranker:
    booster: lgb.Booster
    features: list[str]
    best_iteration: int
    history: dict = field(default_factory=dict)

    def score(self, ft: pl.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in ft.columns]
        if missing:
            raise ValueError(f"feature table lacks {missing}")
        return self.booster.predict(feature_matrix(ft, self.features),
                                    num_iteration=self.best_iteration)

    def importance(self) -> pl.DataFrame:
        gain = self.booster.feature_importance("gain", iteration=self.best_iteration)
        return (pl.DataFrame({"feature": self.features, "gain": gain})
                .with_columns((pl.col("gain") / pl.col("gain").sum()).alias("share"))
                .sort("gain", descending=True))


def train(
    train_ft: pl.DataFrame,
    valid_ft: pl.DataFrame,
    features: list[str],
    params: dict | None = None,
    max_rounds: int = MAX_ROUNDS,
) -> Ranker:
    """Fit on `train_ft`, early-stopping on `valid_ft`'s nDCG@10.

    `valid_ft` chooses the number of trees, so it is no longer an unbiased
    place to report results: headline numbers come from a third split (D36).
    """
    p = dict(PARAMS, **(params or {}))
    if LABEL in features:
        raise ValueError("the label is not a feature")
    dtrain = lgb.Dataset(feature_matrix(train_ft, features), label=train_ft[LABEL].to_numpy(),
                         group=group_sizes(train_ft["impression_id"]),
                         feature_name=features, free_raw_data=True)
    dvalid = lgb.Dataset(feature_matrix(valid_ft, features), label=valid_ft[LABEL].to_numpy(),
                         group=group_sizes(valid_ft["impression_id"]),
                         feature_name=features, reference=dtrain, free_raw_data=True)
    hist: dict = {}
    booster = lgb.train(
        p, dtrain, num_boost_round=max_rounds, valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True, verbose=False),
                   lgb.record_evaluation(hist)],
    )
    return Ranker(booster, list(features), booster.best_iteration, hist)


def per_impression(ft: pl.DataFrame, scores: np.ndarray) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Split row-aligned scores and labels into one array per impression, in row order."""
    sizes = group_sizes(ft["impression_id"])
    bounds = np.cumsum(sizes)[:-1]
    ids = ft["impression_id"].rle().struct.field("value").to_list()
    y = ft[LABEL].to_numpy().astype(np.int8)
    return ids, np.split(np.asarray(scores), bounds), np.split(y, bounds)


def default_features(dataset: str) -> list[str]:
    return list(FEATURES[dataset])
