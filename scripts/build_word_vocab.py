"""D43: build the MIND title vocabulary and its GloVe embedding matrix.

    python scripts/build_word_vocab.py --dataset mind

Reads data/processed/articles.parquet and data/raw/glove/glove.6B.300d.txt,
writes data/processed/words_{dataset}.npz (gitignored): the vocabulary, the
embedding matrix, and the per-article title token matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from newsrec.rank import words  # noqa: E402

GLOVE = REPO_ROOT / "data" / "raw" / "glove" / "glove.6B.300d.txt"
PROCESSED = REPO_ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mind")
    ap.add_argument("--dim", type=int, default=300)
    args = ap.parse_args()

    arts = (pl.scan_parquet(PROCESSED / "articles.parquet")
            .filter(pl.col("dataset") == args.dataset)
            .select("article_id", "title").collect())
    titles = [t or "" for t in arts["title"].to_list()]
    print(f"{args.dataset}: {len(titles):,} articles")

    t0 = time.perf_counter()
    vocab = words.build_vocabulary(titles)
    print(f"vocabulary: {len(vocab):,} tokens (incl. <pad>, <unk>) "
          f"in {time.perf_counter()-t0:.0f} s")
    prof = words.title_length_profile(titles, vocab)
    print("title tokens: " + json.dumps(prof))

    t0 = time.perf_counter()
    glove = words.load_glove(GLOVE, set(vocab.words))
    emb = words.embedding_matrix(vocab, glove, args.dim)
    print(f"GloVe: {vocab.n_glove:,}/{len(vocab):,} words covered "
          f"({vocab.n_glove/len(vocab):.1%}) in {time.perf_counter()-t0:.0f} s")

    toks = words.title_token_matrix(arts["article_id"].to_list(), titles, vocab)
    empty = int((toks == words.PAD).all(1).sum())
    print(f"token matrix: {toks.shape}, {empty} articles with an empty title, "
          f"{(toks == words.UNK).mean():.3%} of slots <unk>")

    out = PROCESSED / f"words_{args.dataset}.npz"
    np.savez_compressed(out, words=np.array(vocab.words, dtype=object),
                        emb=emb, tokens=toks,
                        article_ids=np.array(arts["article_id"].to_list(), dtype=object),
                        profile=json.dumps(prof), n_glove=vocab.n_glove)
    print(f"-> {out.relative_to(REPO_ROOT)} ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
