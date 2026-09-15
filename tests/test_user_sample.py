"""D33: the EB-NeRD user sample keeps or drops whole users, never parts of one.

Constructed data rather than ebnerd_large, because the properties worth proving
sit at boundaries a real file does not point at: residue 4 (kept at 5%) against
residue 5 (dropped), a multi-impression user who must not be split, and IDs
above 100 where the last two digits, not the whole number, decide membership.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from newsrec import ingest_ebnerd

# residue:     0    4    5    10   99   3 (ID > 1000, so `id < 5` alone would drop it)
# 110 is the only ID whose last *one* digit (0) is under 5 while its last two
# (10) are not - without it, `% 10` in place of `% 100` passed every test
# (mutation M2, 2026-09-15).
USER_IDS = [100, 104, 105, 110, 199, 1203]
IMPRESSIONS_PER_USER = 3
KEPT_AT_5 = {"ebnerd:100", "ebnerd:104", "ebnerd:1203"}


def _write_bundle(tmp_path: Path) -> tuple[Path, Path]:
    rows = [(u, i) for u in USER_IDS for i in range(IMPRESSIONS_PER_USER)]
    behaviors = pl.DataFrame(
        {
            "impression_id": [u * 10 + i for u, i in rows],
            "user_id": pl.Series([u for u, _ in rows], dtype=pl.UInt32),
            "impression_time": [datetime(2023, 5, 20, i) for _, i in rows],
            "article_ids_inview": [[1, 2, 3] for _ in rows],
            "article_ids_clicked": [[2] for _ in rows],
            "read_time": pl.Series([10.0] * len(rows), dtype=pl.Float32),
            "scroll_percentage": pl.Series([50.0] * len(rows), dtype=pl.Float32),
            "device_type": pl.Series([1] * len(rows), dtype=pl.Int8),
            "session_id": pl.Series(list(range(len(rows))), dtype=pl.UInt32),
        }
    )
    history = pl.DataFrame(
        {
            "user_id": pl.Series(USER_IDS, dtype=pl.UInt32),
            "article_id_fixed": [[7, 8] for _ in USER_IDS],
            "impression_time_fixed": [
                [datetime(2023, 5, 1), datetime(2023, 5, 2)] for _ in USER_IDS
            ],
        }
    )
    b_path, h_path = tmp_path / "behaviors.parquet", tmp_path / "history.parquet"
    behaviors.write_parquet(b_path)
    history.write_parquet(h_path)
    return b_path, h_path


def test_the_sample_is_decided_by_the_last_two_digits_at_the_boundary(tmp_path):
    b_path, h_path = _write_bundle(tmp_path)
    imps = ingest_ebnerd.load_behaviors(b_path, user_sample_pct=5)
    hist = ingest_ebnerd.load_history(h_path, user_sample_pct=5)
    assert set(imps["user_id"]) == KEPT_AT_5
    assert set(hist["user_id"]) == KEPT_AT_5


def test_a_kept_user_keeps_every_one_of_their_impressions(tmp_path):
    b_path, _ = _write_bundle(tmp_path)
    imps = ingest_ebnerd.load_behaviors(b_path, user_sample_pct=5)
    per_user = dict(imps.group_by("user_id").len().iter_rows())
    assert per_user == {u: IMPRESSIONS_PER_USER for u in KEPT_AT_5}


def test_history_and_impressions_describe_the_same_users(tmp_path):
    b_path, h_path = _write_bundle(tmp_path)
    for pct in (1, 5, 6, 100):
        imps = ingest_ebnerd.load_behaviors(b_path, user_sample_pct=pct)
        hist = ingest_ebnerd.load_history(h_path, user_sample_pct=pct)
        assert set(imps["user_id"]) == set(hist["user_id"]), pct


def test_samples_are_nested_so_growing_one_only_adds_users(tmp_path):
    b_path, _ = _write_bundle(tmp_path)
    sets = [
        set(ingest_ebnerd.load_behaviors(b_path, user_sample_pct=p)["user_id"])
        for p in (1, 5, 6, 100)
    ]
    for smaller, larger in zip(sets, sets[1:]):
        assert smaller <= larger
    assert sets[2] - sets[1] == {"ebnerd:105"}


def test_no_sample_and_a_hundred_percent_both_keep_everyone(tmp_path):
    b_path, h_path = _write_bundle(tmp_path)
    everyone = len(USER_IDS) * IMPRESSIONS_PER_USER
    assert ingest_ebnerd.load_behaviors(b_path).height == everyone
    assert ingest_ebnerd.load_behaviors(b_path, user_sample_pct=100).height == everyone
    assert ingest_ebnerd.load_history(h_path).height == len(USER_IDS)


@pytest.mark.parametrize("bad", [0, 101, -5, 5.0, "5", True])
def test_a_malformed_percentage_is_rejected_not_coerced(bad):
    with pytest.raises(ValueError):
        ingest_ebnerd.user_sample_filter(bad)
