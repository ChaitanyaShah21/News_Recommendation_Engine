"""Phase 5 / Q5 - read the unlabeled leaderboard test bundles.

This module is deliberately separate from `build.py` (which builds the
train/val/test feature store) for one reason worth stating plainly: the word
"test" already means something else here. D7's local test split is carved from
MINDsmall_dev's / EB-NeRD-validation's tail and *has labels*. The bundles read
here are the Codabench test sets - millions of impressions with no labels at
all. Folding them into `impressions.parquet` under `split == "test"` would
silently redefine that split for every script written in Phases 1-4.

So: same unified schema (D3), same reader functions where they already work,
but a separate output directory `data/processed/submission/`.

Landmines this module exists to handle, all three verified against the real
bundles on 2026-08-25 before a line of this was written:

  1. EB-NeRD's test behaviors has **no `article_ids_clicked` column** - 14
     columns, and the label is simply absent. `ingest_ebnerd.load_behaviors`
     selects it unconditionally and raises ColumnNotFoundError.
  2. MIND's test behaviors has **no `-0`/`-1` click suffixes** - verified 0
     matches for `N[0-9]+-[01]` across 50,000 rows. `ingest_mind.load_behaviors`
     degrades *correctly but silently* here (the suffix strip becomes a no-op
     and the clicked filter yields empty lists, which is what D3 specifies for
     unlabeled rows). Silent-and-correct today is silent-and-wrong the day the
     format changes, so `assert_unlabeled` checks it instead of trusting it.
  3. EB-NeRD's test behaviors carries an **`is_beyond_accuracy` flag**, true for
     exactly 200,000 of 13,536,710 rows (counted). Carried through rather than
     dropped so the submission writer can see it. What that flag *means* to the
     scorer is **not** verified - the published Submission Guidelines never
     mention it - so we predict every row, which the official example submission
     confirms is correct: it contains all 13,536,710 lines.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import ingest_ebnerd, ingest_mind

# Where the submission-side artifacts live. Kept out of the feature store
# proper so `build_pipeline.py` can never overwrite them and nothing that
# filters on `split` can ever accidentally pick them up.
SUBMISSION_SUBDIR = "submission"

DATASETS = ("mind", "ebnerd")


def _check_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")


def load_submission_articles(dataset: str, test_root: Path) -> pl.DataFrame:
    """Read the test bundle's article corpus into the unified `articles` schema.

    Both existing `load_articles` functions work here **unchanged** - checked
    column by column against the real test schemas rather than assumed. The
    test bundles are supersets of the dev-scale corpora, not a different shape:
    MIND's test news.tsv is the same 8-column headerless TSV, and EB-NeRD's
    test articles.parquet carries all 13 columns `ingest_ebnerd` selects.
    """
    _check_dataset(dataset)

    if dataset == "mind":
        articles = ingest_mind.load_articles(test_root / "news.tsv")
    else:
        articles = ingest_ebnerd.load_articles(test_root / "articles.parquet")

    # build_article_embeddings raises on duplicate ids (a duplicate would burn a
    # top-K slot on the same article twice), so catch it here where the error
    # message can name the file instead of 40 minutes into an embedding run.
    n_dup = articles.height - articles["article_id"].n_unique()
    if n_dup:
        raise ValueError(
            f"{dataset} test corpus has {n_dup} duplicate article_id values "
            f"at {test_root} - the embedding step will reject this"
        )
    if articles.is_empty():
        raise ValueError(f"{dataset} test corpus at {test_root} is empty")

    return articles


def assert_mind_test_unlabeled(behaviors_tsv_path: Path) -> None:
    """Landmine 2, checked rather than trusted.

    `ingest_mind.load_behaviors` handles the unlabeled test file *correctly by
    coincidence*: the `-[01]$` strip becomes a no-op and the `ends_with("-1")`
    filter yields empty lists, which is exactly what D3 specifies for unlabeled
    rows. Nothing about that is wrong - but nothing about it is *checked*
    either. If a future MIND bundle shipped labels in the test file, or a
    partially-labeled one, this code would keep running and quietly produce a
    submission built on a false premise.

    So the submission reader refuses to proceed unless the file really is
    unlabeled. One full pass, a few seconds, once.
    """
    n_labeled = (
        pl.scan_csv(
            behaviors_tsv_path,
            separator="\t",
            has_header=False,
            quote_char=None,
            new_columns=ingest_mind.BEHAVIOR_COLS,
        )
        .select(pl.col("impressions").str.contains(r"-[01](\s|$)").sum().alias("n"))
        .collect(engine="streaming")
        .item()
    )
    if n_labeled:
        raise ValueError(
            f"{behaviors_tsv_path} has {n_labeled:,} rows carrying -0/-1 click "
            f"labels, but this reader assumes the UNLABELED Codabench test "
            f"bundle. Either the wrong file is configured as `test_root`, or "
            f"MIND's test format has changed - do not submit until this is "
            f"understood."
        )


def load_submission_behaviors(dataset: str, test_root: Path) -> pl.LazyFrame:
    """Read the leaderboard test impressions - **lazily**, as a LazyFrame.

    Returns a LazyFrame and not a DataFrame, unlike every other reader in this
    package, because EB-NeRD's test split is 13,536,710 impressions and
    `SCALE_NOTES.md` already measured what materialising it costs: ~4.5 GB of
    NumPy object header overhead against 0.65 GB of actual data. The caller
    slices this frame into chunks and writes predictions incrementally.

    **There is deliberately no `clicked_article_ids` column.** D3 specifies an
    empty list for unlabeled rows, and that is right for the feature store,
    where labeled and unlabeled rows share one table. Here the whole split is
    unlabeled, and an all-empty label column is an invitation to the precise
    mistake Q9 exists to catch: code that computes a metric over it and reports
    a number instead of refusing. An absent column raises immediately.
    """
    _check_dataset(dataset)

    if dataset == "mind":
        # Refuse the file before reading it if it turns out to be labeled.
        assert_mind_test_unlabeled(test_root / "behaviors.tsv")

        behaviors = pl.scan_csv(
            test_root / "behaviors.tsv",
            separator="\t",
            has_header=False,
            quote_char=None,
            new_columns=ingest_mind.BEHAVIOR_COLS,
            schema_overrides={
                "impression_id": pl.Int64,
                "history": pl.Utf8,
                "impressions": pl.Utf8,
            },
        )
        return behaviors.select(
            pl.lit("mind").alias("dataset"),
            (pl.lit("mind:") + pl.col("impression_id").cast(pl.Utf8)).alias(
                "impression_id"
            ),
            (pl.lit("mind:") + pl.col("user_id")).alias("user_id"),
            pl.col("time")
            .str.strptime(pl.Datetime, ingest_mind.MIND_TIME_FORMAT)
            .alias("timestamp"),
            # No suffix strip. The assert above guarantees there is nothing to
            # strip, and leaving the strip in would let this reader keep
            # "working" on a labeled file the assert is there to reject.
            pl.col("impressions")
            .str.split(" ")
            .list.eval(pl.lit("mind:") + pl.element())
            .alias("candidate_article_ids"),
        )

    behaviors = pl.scan_parquet(test_root / "test" / "behaviors.parquet")
    return behaviors.select(
        pl.lit("ebnerd").alias("dataset"),
        (pl.lit("ebnerd:") + pl.col("impression_id").cast(pl.Utf8)).alias(
            "impression_id"
        ),
        (pl.lit("ebnerd:") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
        pl.col("impression_time").alias("timestamp"),
        pl.col("article_ids_inview")
        .list.eval(pl.lit("ebnerd:") + pl.element().cast(pl.Utf8))
        .alias("candidate_article_ids"),
        # Landmine 3. Carried through rather than dropped, so the submission
        # writer can act on it if needed.
        #
        # What is VERIFIED: the column exists and is true for exactly 200,000 of
        # 13,536,710 rows (counted 2026-08-26), and the competition's official
        # example submission contains all 13,536,710 lines - so every row must be
        # predicted regardless of this flag, which is what we do.
        #
        # What is NOT verified: an earlier note here asserted that "the RecSys
        # 2024 rules treat these as a separately-scored subset". The published
        # Submission Guidelines say nothing about the flag, so that claim was
        # inferred, not read. Left as an observation rather than a rule.
        pl.col("is_beyond_accuracy"),
    )


def load_submission_history(
    dataset: str, test_root: Path, n_recent: int = 10
) -> pl.DataFrame:
    """Read the test users' history, **truncated to the last `n_recent` articles**.

    The truncation is not an optimisation detail, it is what makes this run at
    all. EB-NeRD's test history is 807,677 users at a median of 83 articles
    each - roughly 67 million ids. `build_user_vectors` calls `.to_list()`,
    which would materialise those as ~67 million Python strings, several
    gigabytes on a machine with about 2.5 GB free. Truncating first leaves at
    most 8 million.

    It changes no result: `build_user_vectors` already takes `[-n_recent:]`
    from the end of the list, so handing it a pre-truncated tail is the same
    tail. D12's N = 10 is the default, matching Q2/Q3 exactly.

    Ordering is oldest-first, so the tail is the most recent. Re-verified on
    the real test bundle rather than inherited from the demo-scale check:
    0 of 807,677 EB-NeRD test users have out-of-order `impression_time_fixed`
    (the demo check covered 4,714). MIND ships no history timestamps at all,
    so its ordering still rests on the dataset documentation - unchanged, and
    unchangeable, from Phase 2.
    """
    _check_dataset(dataset)

    if dataset == "mind":
        # MIND has no history file: history is field 4 of behaviors.tsv,
        # repeated identically on every one of that user's rows. Re-verified at
        # test scale: 0 of 484,059 multi-row users carry more than one distinct
        # history string (the Phase 1 check covered 33,617).
        behaviors = pl.scan_csv(
            test_root / "behaviors.tsv",
            separator="\t",
            has_header=False,
            quote_char=None,
            new_columns=ingest_mind.BEHAVIOR_COLS,
            schema_overrides={"history": pl.Utf8},
        ).select("user_id", "history")

        # fill_null BEFORE any list operation. `.str.split()` on a cold-start
        # user's null history returns null, not [] - the Phase 1 trap, and
        # 29,108 of these test rows are cold-start.
        history_lists = (
            pl.col("history")
            .str.split(" ")
            .fill_null([])
            # Guard the empty-string case too: a history of "" splits to [""],
            # which would become the id "mind:" - a token that looks like an
            # article id and matches nothing. Not observed in this bundle, but
            # it costs one expression to make impossible.
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
            .list.tail(n_recent)
        )

        return (
            behaviors.unique(subset="user_id", keep="any")
            .select(
                pl.lit("mind").alias("dataset"),
                (pl.lit("mind:") + pl.col("user_id")).alias("user_id"),
                history_lists.list.eval(pl.lit("mind:") + pl.element()).alias(
                    "history_article_ids"
                ),
                pl.lit(None, dtype=pl.List(pl.Datetime)).alias("history_timestamps"),
            )
            .collect(engine="streaming")
        )

    history = pl.scan_parquet(test_root / "test" / "history.parquet")
    return (
        history.select(
            pl.lit("ebnerd").alias("dataset"),
            (pl.lit("ebnerd:") + pl.col("user_id").cast(pl.Utf8)).alias("user_id"),
            pl.col("article_id_fixed")
            .list.tail(n_recent)
            .list.eval(pl.lit("ebnerd:") + pl.element().cast(pl.Utf8))
            .alias("history_article_ids"),
            # Truncated to the same tail, so the two lists stay row-aligned.
            pl.col("impression_time_fixed")
            .list.tail(n_recent)
            .alias("history_timestamps"),
        )
        .collect(engine="streaming")
    )


def count_empty_texts(articles: pl.DataFrame) -> int:
    """How many articles would embed to a vector of nothing.

    Not an error - reported, not raised. An article whose title *and* abstract
    are both blank still gets encoded, and the model returns a well-formed
    384-dimensional unit vector for the empty string. That vector is confident
    nonsense of exactly the kind D20 rejected `distiluse` over, so its
    population size belongs in the run log rather than being discovered later
    as an unexplained cluster of identical neighbours.
    """
    from .retrieval.semantic import build_document_texts

    with_text = build_document_texts(articles)
    return int((with_text["text"].str.strip_chars() == "").sum())
