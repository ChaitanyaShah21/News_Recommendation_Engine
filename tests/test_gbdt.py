"""re-ranker: group contiguity (Landmine 4), alignment, and that it learns at all."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from newsrec.rank import gbdt


def _ids(*xs):
    return pl.Series("impression_id", list(xs))


def test_group_sizes_are_run_lengths():
    assert gbdt.group_sizes(_ids("a", "a", "a", "b", "b", "c")).tolist() == [3, 2, 1]


def test_single_row_groups_and_single_group():
    assert gbdt.group_sizes(_ids("a", "b", "c")).tolist() == [1, 1, 1]
    assert gbdt.group_sizes(_ids("a", "a")).tolist() == [2]


def test_non_contiguous_group_raises():
    # the sizes [2, 1, 1] would sum to 4 = row count and LightGBM would accept them
    with pytest.raises(ValueError, match="contiguous"):
        gbdt.group_sizes(_ids("a", "a", "b", "a"))


def test_a_shuffled_real_shape_table_is_refused():
    rng = np.random.default_rng(0)
    ids = np.repeat([f"i{k}" for k in range(50)], 7)
    with pytest.raises(ValueError, match="contiguous"):
        gbdt.group_sizes(pl.Series(rng.permutation(ids)))


def test_empty_raises():
    with pytest.raises(ValueError):
        gbdt.group_sizes(pl.Series("impression_id", [], dtype=pl.String))


def _planted(n_imp: int, seed: int, rack: int = 8) -> pl.DataFrame:
    """One click per impression; the clicked row carries +1 on `signal`, while
    `decoy` is noise plus an impression-level shift that says nothing about which
    row inside the rack was clicked.

    The shift is deliberately NOT added to `signal`: trees split on absolute
    values, so a signal that is only meaningful relative to its own rack is
    largely invisible to them (measured: top-1 fell to 0.32 when it was). That is
    a property of GBDT, recorded in ARCHITECTURE.md, not something to test away."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_imp):
        hit = rng.integers(rack)
        offset = rng.normal(0, 5)  # impression-level shift: useless within a rack
        for j in range(rack):
            rows.append({"impression_id": f"{seed}-{i}", "user_id": "u", "article_id": f"a{j}",
                         "clicked": int(j == hit),
                         "signal": (1.0 if j == hit else 0.0) + rng.normal(0, 0.3),
                         "decoy": offset + rng.normal(0, 1)})
    return pl.DataFrame(rows)


def test_planted_signal_is_learned_and_decoy_is_not():
    tr, va, te = _planted(400, 1), _planted(200, 2), _planted(300, 3)
    r = gbdt.train(tr, va, ["signal", "decoy"], params={"min_data_in_leaf": 5, "num_threads": 1},
                   max_rounds=200)
    ids, scores, labels = gbdt.per_impression(te, r.score(te))
    top1 = np.mean([y[np.argmax(s)] for s, y in zip(scores, labels)])
    assert top1 > 0.8, top1  # random is 1/8
    imp = r.importance()
    assert imp["feature"][0] == "signal"


def test_per_impression_keeps_row_alignment():
    ft = pl.DataFrame({"impression_id": ["x", "x", "y", "z", "z", "z"],
                       "clicked": [0, 1, 1, 0, 0, 1]})
    scores = np.arange(6.0)
    ids, s, y = gbdt.per_impression(ft, scores)
    assert ids == ["x", "y", "z"]
    assert [a.tolist() for a in s] == [[0, 1], [2], [3, 4, 5]]
    assert [a.tolist() for a in y] == [[0, 1], [1], [0, 0, 1]]


def test_the_label_cannot_be_a_feature():
    tr = _planted(20, 1)
    with pytest.raises(ValueError, match="label"):
        gbdt.train(tr, tr, ["signal", "clicked"])


def test_score_refuses_a_table_missing_a_feature():
    tr, va = _planted(100, 1), _planted(50, 2)
    r = gbdt.train(tr, va, ["signal"], params={"min_data_in_leaf": 5, "num_threads": 1},
                   max_rounds=20)
    with pytest.raises(ValueError, match="lacks"):
        r.score(va.drop("signal"))
