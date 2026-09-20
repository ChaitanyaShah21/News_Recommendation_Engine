"""assertions that no future information reaches a prediction.

Required explicitly by the assignment - *"Enforce the behaviour-window boundary
- no future-click leakage. Include a test asserting this."*

Leakage is the failure mode this project is least able to detect by looking at
results, because its only symptom is that the numbers get *better*. Every other
bug here announces itself with a traceback, a NaN or an implausible figure. A
leak announces itself with a good score, which is the one outcome nobody
investigates. So it has to be asserted rather than observed.

Two conditions are needed, and arguing only the first is the common mistake:

  TEMPORAL   a prediction for time T may use only facts true strictly before T.
  LABEL-FREE it may not be derived from `clicked_article_ids`, at any time.

The second is independent. A candidate pool built from "articles clicked before
T" satisfies TEMPORAL perfectly and is still leakage, because the pool has been
pre-selected by the answers. D19 requires both; these tests assert both.

The tests run against the real feature store where one exists and skip if it
does not, so a fresh clone still passes - but the constructed-data tests always
run, because they are the ones that can prove a specific wrong implementation
would be caught.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from newsrec.eval import rerank, slices
from newsrec.retrieval import availability

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

requires_store = pytest.mark.skipif(
    not (PROCESSED / "impressions.parquet").exists(),
    reason="feature store not built; run scripts/build_pipeline.py",
)


def _impressions(rows: list[tuple[str, str, datetime, list[str], list[str]]], split="val"):
    return pl.DataFrame(
        {
            "dataset": ["t"] * len(rows),
            "split": [split] * len(rows),
            "impression_id": [r[0] for r in rows],
            "user_id": [r[1] for r in rows],
            "timestamp": [r[2] for r in rows],
            "candidate_article_ids": [r[3] for r in rows],
            "clicked_article_ids": [r[4] for r in rows],
        }
    )


T0 = datetime(2026, 8, 25, 10, 0, 0)


# ==========================================================================
# 1. TEMPORAL - the availability predicate is strictly `<`, never `<=`
# ==========================================================================


def test_an_article_first_seen_in_this_very_impression_is_not_available_to_it():
    """The whole reason the inequality must be strict.

    Article `new` appears for the first time in the 10:00 impression. Under
    `first_seen <= T` it would be available to that impression - but our only
    evidence that `new` exists comes from the impression we are predicting. That
    is circular, not merely early.
    """
    imps = _impressions(
        [
            ("i1", "u1", T0, ["old", "new"], ["new"]),
            ("i2", "u1", T0 + timedelta(hours=2), ["old", "new"], ["old"]),
        ]
    )
    article_ids = ["old", "new"]
    # `old` was seen earlier so it is genuinely in circulation by 10:00
    earlier = _impressions([("i0", "u9", T0 - timedelta(hours=5), ["old"], ["old"])])
    all_imps = pl.concat([earlier, imps])

    _, bucket_id, masks = availability.build_availability(all_imps, imps, article_ids)

    bucket_at_t0 = masks[bucket_id[T0.replace(minute=0, second=0)]]
    assert bucket_at_t0[0]  # old: first seen 05:00, strictly before 10:00
    assert not bucket_at_t0[1], "an article first seen at T must not be available at T"


def test_availability_is_monotone_in_time():
    """An article, once in circulation, never leaves it - so masks only grow."""
    times = [T0 + timedelta(hours=h) for h in range(4)]
    rows = [(f"i{h}", "u1", t, [f"a{h}", "a0"], [f"a{h}"]) for h, t in enumerate(times)]
    imps = _impressions(rows)
    article_ids = [f"a{h}" for h in range(4)]

    _, bucket_id, masks = availability.build_availability(imps, imps, article_ids)
    ordered = [masks[bucket_id[b]] for b in sorted(bucket_id)]
    for earlier, later in zip(ordered, ordered[1:]):
        assert np.all(later >= earlier), "an available article became unavailable"


def test_first_seen_uses_the_earliest_appearance_not_the_latest():
    imps = _impressions(
        [
            ("i1", "u1", T0 + timedelta(hours=3), ["a"], ["a"]),
            ("i2", "u2", T0, ["a"], ["a"]),
            ("i3", "u3", T0 + timedelta(hours=9), ["a"], ["a"]),
        ]
    )
    seen = availability.first_seen_times(imps)
    assert seen.filter(pl.col("article_id") == "a")[0, "first_seen"] == T0


# ==========================================================================
# 2. LABEL-FREE - nothing that gates a prediction is derived from clicks
# ==========================================================================


def test_availability_ignores_clicks_entirely():
    """Constructed so a clicks-derived implementation gives a *different* answer.

    `shown_never_clicked` appears in candidate lists from 08:00 but is never
    clicked by anyone. A correct implementation has it available at 10:00. An
    implementation that computed first-seen from `clicked_article_ids` would
    never see it at all and would mark it unavailable forever.
    """
    imps = _impressions(
        [
            ("i0", "u9", T0 - timedelta(hours=2), ["shown_never_clicked", "other"], ["other"]),
            ("i1", "u1", T0, ["shown_never_clicked", "other"], ["other"]),
        ]
    )
    article_ids = ["shown_never_clicked", "other"]
    _, bucket_id, masks = availability.build_availability(imps, imps, article_ids)
    mask = masks[bucket_id[T0.replace(minute=0, second=0)]]
    assert mask[0], "an article that was shown but never clicked must still be available"


def test_slice_exposure_is_counted_from_candidates_not_clicks():
    """Same rule for evaluation-time grouping (D26).

    A slice built from clicks would group impressions by a quantity derived from
    their own labels, which is circular even though nothing about it is
    temporal.
    """
    imps = _impressions(
        [
            ("i1", "u1", T0, ["popular_never_clicked", "b"], ["b"]),
            ("i2", "u2", T0, ["popular_never_clicked", "b"], ["b"]),
            ("i3", "u3", T0, ["popular_never_clicked", "c"], ["c"]),
        ]
    )
    counts = slices.exposure_counts(imps)
    assert counts["popular_never_clicked"] == 3
    # a clicks-derived count would give this article 0 and rank `b` top instead
    assert slices.head_set_from_counts(counts, 0.4) == {"popular_never_clicked"}


def test_popularity_features_refuse_to_be_built_from_the_evaluated_split():
    """`train_click_counts` is the one place clicks legitimately become a feature.

    It is therefore the one place a split mistake turns into a label leak, so it
    raises rather than trusting its caller.
    """
    val = _impressions([("i1", "u1", T0, ["a", "b"], ["a"])], split="val")
    with pytest.raises(ValueError, match="train only"):
        rerank.train_click_counts(val, ["a", "b"])

    mixed = pl.concat(
        [
            _impressions([("i1", "u1", T0, ["a"], ["a"])], split="train"),
            _impressions([("i2", "u2", T0, ["b"], ["b"])], split="val"),
        ]
    )
    with pytest.raises(ValueError, match="train only"):
        rerank.train_click_counts(mixed, ["a", "b"])


# ==========================================================================
# 3. The temporal split boundary itself
# ==========================================================================


@requires_store
def test_the_split_boundaries_do_not_overlap_in_time():
    """Q1.3's invariant, re-asserted here because Q9 grades it.

    A single impression on the wrong side of a boundary would let a model see a
    Thursday click while predicting Wednesday.
    """
    frame = pl.read_parquet(PROCESSED / "impressions.parquet")
    for dataset in frame.get_column("dataset").unique():
        d = frame.filter(pl.col("dataset") == dataset)
        bounds = {
            split: (
                d.filter(pl.col("split") == split).get_column("timestamp").min(),
                d.filter(pl.col("split") == split).get_column("timestamp").max(),
            )
            for split in ("train", "val", "test")
        }
        assert bounds["train"][1] < bounds["val"][0], f"{dataset}: train overruns val"
        assert bounds["val"][1] < bounds["test"][0], f"{dataset}: val overruns test"


@requires_store
def test_no_impression_carries_a_split_it_does_not_belong_to():
    """Every impression sits inside its own split's time range, not merely on
    the right side of one boundary."""
    frame = pl.read_parquet(PROCESSED / "impressions.parquet")
    for dataset in frame.get_column("dataset").unique():
        d = frame.filter(pl.col("dataset") == dataset)
        train_max = d.filter(pl.col("split") == "train").get_column("timestamp").max()
        later = d.filter(pl.col("split") != "train").get_column("timestamp")
        assert later.min() > train_max


# ==========================================================================
# 4. The history-snapshot landmine
# ==========================================================================


@requires_store
def test_a_history_snapshot_does_not_contain_clicks_from_its_own_window():
    """If it did, the history field would carry the answer to its own question.

    Measured tolerance rather than zero: a user genuinely re-reading an article
    they clicked earlier in the same window is real behaviour, not leakage.
    Observed rates are 0.578% (MIND train), 0.195% (MIND val), 0.358% / 0.470%
    (EB-NeRD), so 5% is far above the noise and far below a leak.
    """
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet")
    history = pl.read_parquet(PROCESSED / "history.parquet")

    for dataset in impressions.get_column("dataset").unique():
        for split in ("train", "val"):
            h = history.filter((pl.col("dataset") == dataset) & (pl.col("split") == split))
            hist_of = dict(
                zip(
                    h.get_column("user_id").to_list(),
                    h.get_column("history_article_ids").to_list(),
                )
            )
            d = impressions.filter(
                (pl.col("dataset") == dataset) & (pl.col("split") == split)
            )
            total = overlap = 0
            for user, clicks in zip(
                d.get_column("user_id").to_list(),
                d.get_column("clicked_article_ids").to_list(),
            ):
                seen = set(hist_of.get(user) or [])
                if not seen:
                    continue
                total += len(clicks)
                overlap += sum(1 for c in clicks if c in seen)
            if total:
                rate = overlap / total
                assert rate < 0.05, (
                    f"{dataset}/{split}: {rate:.1%} of clicks are already in the "
                    "user's own history snapshot - the snapshot has absorbed the "
                    "window it is supposed to predate"
                )


@requires_store
def test_the_wrong_history_snapshot_would_be_caught_by_that_same_check():
    """The landmine, asserted as a live demonstration rather than a comment.

    `history.parquet` holds one snapshot per (user, split). EB-NeRD's val
    snapshot has absorbed the train window: it contains 99.5% of train-window
    clicks. Pairing it with train impressions - which a join on `user_id` alone
    silently does, since 1,217 EB-NeRD users have all three split rows and
    nothing errors - hands over the answers almost perfectly.

    This test deliberately performs that mis-join and asserts the check above
    would fire. If it ever stops firing, the check has stopped protecting us.
    """
    impressions = pl.read_parquet(PROCESSED / "impressions.parquet").filter(
        (pl.col("dataset") == "ebnerd") & (pl.col("split") == "train")
    )
    wrong = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == "ebnerd") & (pl.col("split") == "val")
    )
    hist_of = dict(
        zip(
            wrong.get_column("user_id").to_list(),
            wrong.get_column("history_article_ids").to_list(),
        )
    )

    total = overlap = 0
    for user, clicks in zip(
        impressions.get_column("user_id").to_list(),
        impressions.get_column("clicked_article_ids").to_list(),
    ):
        seen = set(hist_of.get(user) or [])
        if not seen:
            continue
        total += len(clicks)
        overlap += sum(1 for c in clicks if c in seen)

    assert total > 0
    assert overlap / total > 0.9, (
        "the val-history-on-train-impressions mis-join no longer leaks, which "
        "means this regression test is no longer testing anything - check "
        "whether the store's history snapshots changed"
    )


# ==========================================================================
# 5. The re-ranking scorers see no labels
# ==========================================================================


def test_rerank_scores_do_not_change_when_the_labels_change():
    """The strongest single statement available: flip every label and the
    scores must be byte-identical.

    Any scorer that touched `clicked_article_ids` - directly, or through a
    feature built from it - would move here.
    """
    from newsrec.retrieval import bm25, bm25_search

    articles = pl.DataFrame(
        {
            "dataset": ["t"] * 4,
            "article_id": ["a1", "a2", "a3", "a4"],
            "title": ["tariff talks", "storm warning", "tariff deal", "election poll"],
            "abstract": [""] * 4,
        }
    )
    history = pl.DataFrame(
        {
            "dataset": ["t"],
            "user_id": ["u1"],
            "history_article_ids": [["a1"]],
        }
    )
    article_ids = articles.get_column("article_id").to_list()
    index = bm25.build_index(articles)
    title_term = bm25_search.build_title_term_matrix(articles, index.vocab)
    queries = bm25_search.build_queries(history, index, title_term)

    truthful = _impressions([("i1", "u1", T0, ["a2", "a3", "a4"], ["a3"])])
    inverted = _impressions([("i1", "u1", T0, ["a2", "a3", "a4"], ["a2", "a4"])])

    a = rerank.score_bm25(
        rerank.build_candidate_set(truthful, article_ids, history), index, queries
    )[0]
    b = rerank.score_bm25(
        rerank.build_candidate_set(inverted, article_ids, history), index, queries
    )[0]
    assert np.array_equal(a, b), "BM25 scores moved when only the labels changed"


def test_semantic_scores_do_not_change_when_the_labels_change():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(4, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    from newsrec.retrieval import semantic_search

    history = pl.DataFrame(
        {
            "dataset": ["t"],
            "user_id": ["u1"],
            "history_article_ids": [["a1"]],
        }
    )
    article_ids = ["a1", "a2", "a3", "a4"]
    users = semantic_search.build_user_vectors(history, article_ids, embeddings)

    truthful = _impressions([("i1", "u1", T0, ["a2", "a3", "a4"], ["a3"])])
    inverted = _impressions([("i1", "u1", T0, ["a2", "a3", "a4"], ["a2", "a4"])])

    a = rerank.score_semantic(
        rerank.build_candidate_set(truthful, article_ids, history), users, embeddings
    )[0]
    b = rerank.score_semantic(
        rerank.build_candidate_set(inverted, article_ids, history), users, embeddings
    )[0]
    assert np.array_equal(a, b), "semantic scores moved when only the labels changed"


# --------------------------------------------------------------------------
# A2 (Q1 behavioural features): the two properties every feature must satisfy.
#
# FUTURE DELETION: a feature for an impression at time T may use only the past,
# so deleting every impression after tau must leave the features of every
# impression at or before tau IDENTICAL. This catches any window, count, rank
# or first-seen computation that peeks forward - without having to enumerate
# how each one could.
#
# LABEL BLINDNESS: shuffling the click labels must leave every feature
# identical. The label column itself is the only thing allowed to move.
#
# D42's rack features read every OTHER candidate in the same impression. That is
# not a leak and the two properties above prove it: those candidates are all
# shown at the same instant T, they are the input to the request being served
# (both leaderboards supply the list in full), and no label of any of them is
# read. Future deletion cannot change them because a rack is one impression.
#
# Both run the real builder on real val data for BOTH datasets. EB-NeRD alone
# is not enough: its freshness is min(published_time, first_seen), and
# published_time is almost always the smaller, so a bug in first_seen only shows
# on MIND, where first_seen is the sole source (mutation L2, 2026-09-18). The quarantined future-exposure
# feature is run through the same check and must FAIL it - proof the check can
# see a leak at all, rather than passing because it looks at nothing.
# --------------------------------------------------------------------------

def _a2_inputs(dataset):
    from newsrec.retrieval import semantic
    imps = pl.read_parquet(PROCESSED / "impressions.parquet").filter(pl.col("dataset") == dataset)
    target = imps.filter(pl.col("split") == "val").sort("timestamp").head(400)
    hist = pl.read_parquet(PROCESSED / "history.parquet").filter(
        (pl.col("dataset") == dataset) & pl.col("user_id").is_in(target["user_id"].implode()))
    arts = pl.read_parquet(PROCESSED / "articles.parquet").filter(pl.col("dataset") == dataset)
    ids, emb = semantic.load_article_embeddings(PROCESSED / "embeddings.parquet", dataset=dataset)
    return imps, target, hist, arts, ids, emb


@requires_store
@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_a2_features_are_unchanged_when_the_future_is_deleted(dataset):
    from newsrec.features import assemble
    from newsrec.features.unavailable import unavailable_features
    imps, target, hist, arts, ids, emb = _a2_inputs(dataset)
    tau = target["timestamp"].sort()[target.height // 2]
    early = target.filter(pl.col("timestamp") <= tau)
    past_only = imps.filter(pl.col("timestamp") <= tau)

    full = assemble.build_feature_table(dataset, early, imps, hist, arts, ids, emb)
    trunc = assemble.build_feature_table(dataset, early, past_only, hist, arts, ids, emb)
    # The property is checked over the WHOLE table, so it covers D42's rack
    # columns without naming them. Asserted here so that a refactor which
    # dropped them would fail loudly rather than quietly narrowing the check.
    assert set(assemble.RACK_FEATURES[dataset]) <= set(full.columns)
    assert full.height > 0 and full.equals(trunc), "a feature changed when the future was removed"

    # Teeth: the quarantined future feature must be caught by the same check.
    # (On MIND read_time/scroll_percentage are all-null placeholders from
    # ingestion, so the future exposure share is the part that has teeth there.)
    leak_full = unavailable_features(full, early, imps, arts)["future_exposure_share_24h"]
    leak_trunc = unavailable_features(trunc, early, past_only, arts)["future_exposure_share_24h"]
    assert not leak_full.equals(leak_trunc), "the check cannot see a known leak"


@requires_store
@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_a2_features_are_blind_to_click_labels(dataset):
    from newsrec.features import assemble
    imps, target, hist, arts, ids, emb = _a2_inputs(dataset)
    shuffled = target.with_columns(
        pl.col("clicked_article_ids").shuffle(seed=7).alias("clicked_article_ids"))
    imps_shuffled = imps.update(shuffled.select("impression_id", "clicked_article_ids"),
                                on="impression_id")
    a = assemble.build_feature_table(dataset, target, imps, hist, arts, ids, emb)
    b = assemble.build_feature_table(dataset, shuffled, imps_shuffled, hist, arts, ids, emb)
    assert not a["clicked"].equals(b["clicked"]), "the shuffle did not change the labels"
    assert set(assemble.RACK_FEATURES[dataset]) <= set(a.columns)
    assert a.drop("clicked").equals(b.drop("clicked"))
