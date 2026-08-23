"""Build the feature store (Q1.4): run ingestion + the temporal split for
both datasets, then write the three unified-schema tables to data/processed/
as Parquet - both datasets combined per table (D9 in ARCHITECTURE.md), tagged
by the `dataset` and `split` columns already in the schema. This is the
artefact every later phase (Q2 BM25, Q3 embeddings, Q4 eval) reads directly,
instead of re-parsing raw files every time - see GLOSSARY.md's "feature store"
entry for why that matters.
"""

from pathlib import Path

import polars as pl

from newsrec import ingest_ebnerd, ingest_mind, temporal_split


def build_feature_store(mind_root: Path, ebnerd_root: Path, output_dir: Path) -> None:
    """Ingest MIND-small from `mind_root` (containing MINDsmall_train/,
    MINDsmall_dev/) and EB-NeRD from `ebnerd_root` (containing
    articles.parquet, train/, validation/), apply the temporal split, and
    write articles.parquet, impressions.parquet, and history.parquet to
    `output_dir`.
    """
    # --- MIND ---
    mind_train_articles = ingest_mind.load_articles(
        mind_root / "MINDsmall_train" / "news.tsv"
    )
    mind_dev_articles = ingest_mind.load_articles(
        mind_root / "MINDsmall_dev" / "news.tsv"
    )
    mind_train_behaviors = ingest_mind.load_behaviors(
        mind_root / "MINDsmall_train" / "behaviors.tsv"
    )
    mind_dev_behaviors = ingest_mind.load_behaviors(
        mind_root / "MINDsmall_dev" / "behaviors.tsv"
    )
    mind_train_history = ingest_mind.load_history(
        mind_root / "MINDsmall_train" / "behaviors.tsv"
    )
    mind_dev_history = ingest_mind.load_history(
        mind_root / "MINDsmall_dev" / "behaviors.tsv"
    )

    # Train's and dev's news.tsv are separate crawls, not the same file twice -
    # verified 13,956 of dev's 42,416 articles don't appear in train's news.tsv
    # at all, so this concat+dedupe is necessary, not defensive boilerplate.
    # For the 28,460 articles present in both, content is identical (verified:
    # 0 title mismatches sampled), so keep="first" vs "last" doesn't matter.
    mind_articles = pl.concat([mind_train_articles, mind_dev_articles]).unique(
        subset="article_id", keep="first"
    )

    # --- EB-NeRD --- (one shared articles.parquet, no merge needed)
    ebnerd_articles = ingest_ebnerd.load_articles(ebnerd_root / "articles.parquet")
    ebnerd_train_behaviors = ingest_ebnerd.load_behaviors(
        ebnerd_root / "train" / "behaviors.parquet"
    )
    ebnerd_val_behaviors = ingest_ebnerd.load_behaviors(
        ebnerd_root / "validation" / "behaviors.parquet"
    )
    ebnerd_train_history = ingest_ebnerd.load_history(
        ebnerd_root / "train" / "history.parquet"
    )
    ebnerd_val_history = ingest_ebnerd.load_history(
        ebnerd_root / "validation" / "history.parquet"
    )

    # --- Temporal split, per dataset (D7/D8) ---
    mind_impressions = temporal_split.add_impressions_split(
        mind_train_behaviors, mind_dev_behaviors
    )
    ebnerd_impressions = temporal_split.add_impressions_split(
        ebnerd_train_behaviors, ebnerd_val_behaviors
    )
    mind_history = temporal_split.add_history_split(
        mind_train_history, mind_dev_history
    )
    ebnerd_history = temporal_split.add_history_split(
        ebnerd_train_history, ebnerd_val_history
    )

    # --- Combine both datasets per table (D9), write out ---
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.concat([mind_articles, ebnerd_articles]).write_parquet(
        output_dir / "articles.parquet"
    )
    pl.concat([mind_impressions, ebnerd_impressions]).write_parquet(
        output_dir / "impressions.parquet"
    )
    pl.concat([mind_history, ebnerd_history]).write_parquet(
        output_dir / "history.parquet"
    )
