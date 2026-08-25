"""Adversarial tests for the Phase 5 / Q5 submission readers (R10).

Deliberately constructed files, not slices of the real bundles. The real
MINDlarge_test and ebnerd_testset files happen to be clean - verified on
2026-08-25: no duplicate article ids, 100% candidate coverage, no out-of-order
history. That is exactly why testing against them alone would prove nothing.
Every case below is one the real files do NOT contain.

The single most important test here is `test_truncation_changes_no_vector`.
`load_submission_history` truncates each user's history to its last N *inside
the reader*, for a memory reason (67 million Python strings would not fit).
Any change made for memory reasons that touches the data the model sees has to
be shown not to change the model's output, or it is a silent correctness bug
wearing a performance costume.
"""

import numpy as np
import polars as pl
import pytest

from newsrec import submission
from newsrec.retrieval.semantic_search import build_user_vectors


# --------------------------------------------------------------------------
# helpers: write tiny raw bundles in each dataset's real on-disk format
# --------------------------------------------------------------------------


def _write_mind_behaviors(path, rows):
    """rows: (impression_id, user_id, time, history_or_None, impressions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for imp_id, user, t, hist, imps in rows:
        # MIND writes a missing history as an empty field, which Polars reads
        # as null - the Phase 1 cold-start trap, reproduced faithfully here.
        lines.append(f"{imp_id}\t{user}\t{t}\t{'' if hist is None else hist}\t{imps}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ebnerd_testset(root, behaviors: pl.DataFrame, history: pl.DataFrame,
                          articles: pl.DataFrame):
    (root / "test").mkdir(parents=True, exist_ok=True)
    behaviors.write_parquet(root / "test" / "behaviors.parquet")
    history.write_parquet(root / "test" / "history.parquet")
    articles.write_parquet(root / "articles.parquet")


def _ebnerd_behaviors(rows):
    """rows: (impression_id, user_id, inview_list, is_beyond_accuracy)."""
    return pl.DataFrame(
        {
            "impression_id": pl.Series([r[0] for r in rows], dtype=pl.UInt32),
            "impression_time": pl.Series(
                [__import__("datetime").datetime(2023, 6, 1, 12, 0)] * len(rows),
                dtype=pl.Datetime,
            ),
            "article_ids_inview": pl.Series(
                [r[2] for r in rows], dtype=pl.List(pl.Int32)
            ),
            "user_id": pl.Series([r[1] for r in rows], dtype=pl.UInt32),
            "is_beyond_accuracy": pl.Series([r[3] for r in rows], dtype=pl.Boolean),
        }
    )


# --------------------------------------------------------------------------
# Landmine 2: MIND's test file must be unlabeled, and we must check, not hope
# --------------------------------------------------------------------------


def test_assert_unlabeled_passes_on_real_test_format(tmp_path):
    path = tmp_path / "behaviors.tsv"
    _write_mind_behaviors(path, [
        (1, "U1", "11/19/2019 11:37:45 AM", "N1 N2", "N10 N11 N12"),
        (2, "U2", "11/19/2019 11:38:45 AM", "N3", "N13 N14"),
    ])
    submission.assert_mind_test_unlabeled(path)  # must not raise


def test_assert_unlabeled_rejects_a_labeled_file(tmp_path):
    """The whole point of the guard: a labeled file must be refused loudly.

    Without this check `ingest_mind`'s suffix-stripping would run happily and
    we would submit predictions derived from a file we misunderstood.
    """
    path = tmp_path / "behaviors.tsv"
    _write_mind_behaviors(path, [
        (1, "U1", "11/19/2019 11:37:45 AM", "N1", "N10-0 N11-1"),
    ])
    with pytest.raises(ValueError, match="click labels"):
        submission.assert_mind_test_unlabeled(path)


def test_assert_unlabeled_rejects_a_PARTIALLY_labeled_file(tmp_path):
    """The nastier variant: most rows unlabeled, one row labeled.

    A spot-check of the first few rows would pass this file. The guard scans
    every row precisely so a single labeled row cannot slip through.
    """
    path = tmp_path / "behaviors.tsv"
    rows = [(i, f"U{i}", "11/19/2019 11:37:45 AM", "N1", "N10 N11")
            for i in range(1, 500)]
    rows.append((500, "U500", "11/19/2019 11:37:45 AM", "N1", "N10-1 N11-0"))
    _write_mind_behaviors(path, rows)
    with pytest.raises(ValueError, match="click labels"):
        submission.assert_mind_test_unlabeled(path)


def test_assert_unlabeled_ignores_hyphens_that_are_not_click_suffixes(tmp_path):
    """A false positive here would block a perfectly good submission.

    The regex is `-[01](\\s|$)`, so it must not fire on an id that merely
    contains a hyphen followed by a digit somewhere in the middle.
    """
    path = tmp_path / "behaviors.tsv"
    _write_mind_behaviors(path, [
        (1, "U1", "11/19/2019 11:37:45 AM", "N1", "N10-12 N11-034X N12"),
    ])
    submission.assert_mind_test_unlabeled(path)  # must not raise


# --------------------------------------------------------------------------
# The label column must be ABSENT, not empty
# --------------------------------------------------------------------------


def test_mind_submission_behaviors_has_no_label_column(tmp_path):
    _write_mind_behaviors(tmp_path / "behaviors.tsv", [
        (1, "U1", "11/19/2019 11:37:45 AM", "N1 N2", "N10 N11"),
    ])
    lf = submission.load_submission_behaviors("mind", tmp_path)
    assert "clicked_article_ids" not in lf.collect_schema().names()


def test_reader_itself_refuses_a_labeled_file(tmp_path):
    """Found by mutation testing, not by design: deleting the
    `assert_mind_test_unlabeled` CALL from `load_submission_behaviors` left all
    of the tests above passing, because they exercise the guard function
    directly and nothing pinned that the reader invokes it.

    A guard nothing proves is wired up is decoration. This test fails if the
    call is ever removed.
    """
    _write_mind_behaviors(tmp_path / "behaviors.tsv", [
        (1, "U1", "11/19/2019 11:37:45 AM", "N1", "N10-1 N11-0"),
    ])
    with pytest.raises(ValueError, match="click labels"):
        submission.load_submission_behaviors("mind", tmp_path)


def test_ebnerd_submission_behaviors_has_no_label_column_and_keeps_the_flag(tmp_path):
    _write_ebnerd_testset(
        tmp_path,
        _ebnerd_behaviors([(1, 7, [100, 200], False), (2, 8, [300], True)]),
        pl.DataFrame({
            "user_id": pl.Series([7, 8], dtype=pl.UInt32),
            "impression_time_fixed": pl.Series(
                [[], []], dtype=pl.List(pl.Datetime)),
            "article_id_fixed": pl.Series([[], []], dtype=pl.List(pl.Int32)),
        }),
        pl.DataFrame({"article_id": pl.Series([100], dtype=pl.Int32)}),
    )
    frame = submission.load_submission_behaviors("ebnerd", tmp_path).collect()

    assert "clicked_article_ids" not in frame.columns
    # Landmine 3 must survive the read, or the beyond-accuracy subset is
    # invisible to the submission writer.
    assert frame["is_beyond_accuracy"].to_list() == [False, True]
    assert frame["candidate_article_ids"][0].to_list() == ["ebnerd:100", "ebnerd:200"]
    assert frame["impression_id"].to_list() == ["ebnerd:1", "ebnerd:2"]


# --------------------------------------------------------------------------
# History: the null/empty traps, and that the TAIL is taken
# --------------------------------------------------------------------------


def test_mind_cold_start_history_is_empty_list_not_null(tmp_path):
    """The Phase 1 trap, third appearance. 29,108 real test rows hit this."""
    _write_mind_behaviors(tmp_path / "behaviors.tsv", [
        (1, "U1", "11/19/2019 11:37:45 AM", None, "N10 N11"),
        (2, "U2", "11/19/2019 11:38:45 AM", "N1 N2", "N12"),
    ])
    hist = submission.load_submission_history("mind", tmp_path).sort("user_id")

    assert hist["history_article_ids"].null_count() == 0
    by_user = dict(zip(hist["user_id"], hist["history_article_ids"].to_list()))
    assert by_user["mind:U1"] == []
    assert by_user["mind:U2"] == ["mind:N1", "mind:N2"]


def test_mind_empty_string_history_does_not_become_a_bare_prefix(tmp_path):
    """A history field of a single space splits to [""], which would prefix to
    the id "mind:" - a token shaped like an article id that matches nothing.

    Not present in the real bundle. Constructed here because "not present
    today" is not the same as "cannot happen".
    """
    _write_mind_behaviors(tmp_path / "behaviors.tsv", [
        (1, "U1", "11/19/2019 11:37:45 AM", " ", "N10"),
    ])
    hist = submission.load_submission_history("mind", tmp_path)
    assert hist["history_article_ids"].to_list() == [[]]


def test_mind_history_takes_the_last_n_not_the_first_n(tmp_path):
    """An off-by-direction here is invisible - you still get 10 plausible
    article ids, just the user's *oldest* interests instead of their newest.
    Nothing would error; the recommendations would simply be stale.
    """
    _write_mind_behaviors(tmp_path / "behaviors.tsv", [
        (1, "U1", "11/19/2019 11:37:45 AM", "N1 N2 N3 N4 N5", "N10"),
    ])
    hist = submission.load_submission_history("mind", tmp_path, n_recent=2)
    assert hist["history_article_ids"].to_list() == [["mind:N4", "mind:N5"]]


def test_ebnerd_history_truncation_keeps_ids_and_timestamps_aligned(tmp_path):
    """Two lists meant to correspond, truncated separately - the classic way
    for parallel columns to drift out of sync (R10)."""
    import datetime as dt

    times = [dt.datetime(2023, 6, 1, h) for h in range(5)]
    _write_ebnerd_testset(
        tmp_path,
        _ebnerd_behaviors([(1, 7, [100], False)]),
        pl.DataFrame({
            "user_id": pl.Series([7], dtype=pl.UInt32),
            "impression_time_fixed": pl.Series([times], dtype=pl.List(pl.Datetime)),
            "article_id_fixed": pl.Series(
                [[10, 11, 12, 13, 14]], dtype=pl.List(pl.Int32)),
        }),
        pl.DataFrame({"article_id": pl.Series([100], dtype=pl.Int32)}),
    )
    hist = submission.load_submission_history("ebnerd", tmp_path, n_recent=2)

    ids = hist["history_article_ids"].to_list()[0]
    stamps = hist["history_timestamps"].to_list()[0]
    assert ids == ["ebnerd:13", "ebnerd:14"]
    assert len(stamps) == len(ids)
    # Same tail, not the head: hours 3 and 4, not 0 and 1.
    assert [s.hour for s in stamps] == [3, 4]


def test_mind_history_is_one_row_per_user_across_repeated_impressions(tmp_path):
    """MIND repeats the history string on every impression row for a user.
    484,059 real test users have more than one row."""
    _write_mind_behaviors(tmp_path / "behaviors.tsv", [
        (1, "U1", "11/19/2019 11:37:45 AM", "N1 N2", "N10"),
        (2, "U1", "11/19/2019 11:39:45 AM", "N1 N2", "N11"),
        (3, "U1", "11/19/2019 11:41:45 AM", "N1 N2", "N12"),
    ])
    hist = submission.load_submission_history("mind", tmp_path)
    assert hist.height == 1
    assert hist["history_article_ids"].to_list() == [["mind:N1", "mind:N2"]]


# --------------------------------------------------------------------------
# The load-bearing one: truncating for memory must not change the model output
# --------------------------------------------------------------------------


def test_truncation_changes_no_vector():
    """Pre-truncating history in the reader must produce byte-identical user
    vectors to passing the full history to `build_user_vectors`.

    `build_user_vectors` takes `[-n_recent:]` itself, so the two paths should
    agree. "Should" is the word that makes this a test. The failure this
    guards against is subtle: if the reader's tail and the scorer's tail ever
    disagreed - a different N, an off-by-one, a reversed list - every
    submission would still be well-formed and simply worse, with nothing
    anywhere to indicate it.
    """
    rng = np.random.default_rng(0)
    article_ids = [f"mind:N{i}" for i in range(50)]
    embeddings = rng.standard_normal((50, 384)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    full = [
        [f"mind:N{i}" for i in range(30)],          # long: truncation bites
        [f"mind:N{i}" for i in range(3)],           # short: truncation is a no-op
        [],                                          # cold start
        [f"mind:N{i}" for i in range(10, 20)],       # exactly n_recent
    ]
    truncated = [ids[-10:] for ids in full]

    def frame(histories):
        return pl.DataFrame({
            "user_id": [f"mind:U{i}" for i in range(len(histories))],
            "history_article_ids": histories,
        })

    a = build_user_vectors(frame(full), article_ids, embeddings, n_recent=10)
    b = build_user_vectors(frame(truncated), article_ids, embeddings, n_recent=10)

    assert np.array_equal(a.matrix, b.matrix)
    assert np.array_equal(a.has_query, b.has_query)
    # And the truncation must actually have been doing something, or this
    # test would pass trivially on a no-op.
    assert len(full[0]) > len(truncated[0])


# --------------------------------------------------------------------------
# Articles
# --------------------------------------------------------------------------


def test_duplicate_article_ids_are_rejected_before_the_embedding_run(tmp_path):
    """`build_article_embeddings` raises on duplicates too - but 40 minutes in.
    This is the same failure, caught in the first second, naming the file."""
    news = tmp_path / "news.tsv"
    row = "N1\tsports\tgolf\tA title\tAn abstract\thttp://x\t[]\t[]"
    news.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate article_id"):
        submission.load_submission_articles("mind", tmp_path)


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="dataset must be one of"):
        submission.load_submission_articles("mindd", pl.__file__)
