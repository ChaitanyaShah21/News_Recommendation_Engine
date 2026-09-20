"""NRMS: shapes, the empty-history trap, and that it learns a planted preference."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from newsrec.rank import nrms


def _world(n_imp=600, seed=0, rack=6, d=16):
    """Two topics. A user's history is all one topic; they click the one candidate
    of their topic in each rack. Article vectors: topic direction + noise."""
    rng = np.random.default_rng(seed)
    n_art = 400
    topic = rng.integers(2, size=n_art)
    base = np.eye(d, dtype=np.float32)[topic] * 2
    emb = (base + rng.normal(0, 0.5, (n_art, d))).astype(np.float32)
    ids = [f"a{i}" for i in range(n_art)]
    rows, hist = [], {}
    for i in range(n_imp):
        u = f"u{i % 100}"
        ut = i % 100 % 2
        hist.setdefault(u, [ids[a] for a in rng.choice(np.flatnonzero(topic == ut), 8)])
        hit = rng.choice(np.flatnonzero(topic == ut))
        others = rng.choice(np.flatnonzero(topic != ut), rack - 1, replace=False)
        for a in rng.permutation(np.concatenate([[hit], others])):
            rows.append({"impression_id": f"i{i}", "user_id": u, "article_id": ids[a],
                         "clicked": int(a == hit), "freshness_hours": 1.0,
                         "exposure_share_1h": 0.1})
    ft = pl.DataFrame(rows)
    h = pl.DataFrame({"user_id": list(hist), "history_article_ids": list(hist.values())})
    return ft, h, emb, {a: i for i, a in enumerate(ids)}


def test_learns_a_planted_topic_preference():
    ft, h, emb, row = _world()
    b = nrms.prepare(ft, h, row)
    m = nrms.train(nrms.NRMS(emb.shape[1]), b, emb, epochs=3, batch=64, log=lambda *_: None)
    scores = nrms.score(m, b, emb)
    top1 = np.mean([b.labels[b.cand_start[i]:b.cand_start[i + 1]][np.argmax(s)]
                    for i, s in enumerate(scores)])
    assert top1 > 0.9, top1  # random is 1/6


def test_a_user_with_no_history_gets_finite_scores_not_nan():
    ft, h, emb, row = _world(n_imp=20)
    h = h.with_columns(pl.lit([], dtype=pl.List(pl.Utf8)).alias("history_article_ids"))
    b = nrms.prepare(ft, h, row)
    assert (b.hist_rows == -1).all()
    scores = nrms.score(nrms.NRMS(emb.shape[1]), b, emb)
    assert all(np.isfinite(s).all() for s in scores)


def test_history_is_the_most_recent_and_right_aligned():
    ft, h, emb, row = _world(n_imp=2)
    long = [f"a{i}" for i in range(nrms.HIST_LEN + 7)]
    h = h.with_columns(pl.lit(long).alias("history_article_ids"))
    b = nrms.prepare(ft, h, row)
    assert b.hist_rows[0, -1] == row[long[-1]]
    assert b.hist_rows[0, 0] == row[long[7]]


def test_the_time_term_starts_at_exactly_the_baseline():
    ft, h, emb, row = _world(n_imp=10)
    torch.manual_seed(0)
    base = nrms.NRMS(emb.shape[1])
    torch.manual_seed(0)
    timed = nrms.NRMS(emb.shape[1], n_time=2)
    b0 = nrms.prepare(ft, h, row)
    b1 = nrms.prepare(ft, h, row, time_cols=["freshness_hours", "exposure_share_1h"])
    for s0, s1 in zip(nrms.score(base, b0, emb), nrms.score(timed, b1, emb)):
        np.testing.assert_allclose(s0, s1, rtol=1e-6)


def test_every_sample_has_its_positive_first_and_negatives_from_the_same_impression():
    ft, h, emb, row = _world(n_imp=50)
    b = nrms.prepare(ft, h, row)
    S = nrms.training_samples(b, np.random.default_rng(0))
    for imp, pos, *negs in S:
        s, e = b.cand_start[imp], b.cand_start[imp + 1]
        assert s <= pos < e and b.labels[pos] == 1
        assert all(s <= n < e and b.labels[n] == 0 for n in negs)


# --------------------------------------------------------------- D43, word-level

def _word_world(n_imp=600, seed=0, rack=6, vocab=60, title_len=8):
    """The same planted world, but an article's topic is carried by its WORDS.

    Topic-0 articles draw their title tokens from the first half of the
    vocabulary, topic-1 from the second half, with one shared filler token in
    every title so the encoder cannot win by looking at a single position. The
    signal is therefore only recoverable by reading the tokens - which is
    exactly what the sentence-embedding encoder is handed for free and what the
    word-level encoder has to learn.
    """
    rng = np.random.default_rng(seed)
    n_art = 400
    topic = rng.integers(2, size=n_art)
    half = (vocab - 2) // 2
    tokens = np.empty((n_art, title_len), dtype=np.int32)
    for a in range(n_art):
        lo = 2 if topic[a] == 0 else 2 + half
        tokens[a] = rng.integers(lo, lo + half, size=title_len)
        tokens[a, 0] = 2  # a filler token shared by every title
    ids = [f"a{i}" for i in range(n_art)]
    rows, hist = [], {}
    for i in range(n_imp):
        u = f"u{i % 100}"
        ut = i % 100 % 2
        hist.setdefault(u, [ids[a] for a in rng.choice(np.flatnonzero(topic == ut), 8)])
        hit = rng.choice(np.flatnonzero(topic == ut))
        others = rng.choice(np.flatnonzero(topic != ut), rack - 1, replace=False)
        for a in rng.permutation(np.concatenate([[hit], others])):
            rows.append({"impression_id": f"i{i}", "user_id": u, "article_id": ids[a],
                         "clicked": int(a == hit), "freshness_hours": 1.0,
                         "exposure_share_1h": 0.1})
    ft = pl.DataFrame(rows)
    h = pl.DataFrame({"user_id": list(hist), "history_article_ids": list(hist.values())})
    word_emb = rng.normal(0, 0.1, (vocab, 24)).astype(np.float32)
    word_emb[0] = 0.0  # <pad>
    return ft, h, tokens, word_emb, {a: i for i, a in enumerate(ids)}


def test_word_level_encoder_learns_a_planted_preference_from_tokens():
    """D43's licence to spend 14 h of CPU: the word-level path must actually
    learn, not merely produce correctly-shaped finite numbers. The planted
    signal lives only in the token ids, so a model that ignored them - or that
    silently received PAD everywhere - would score at the 1/6 random baseline.
    """
    ft, h, tokens, word_emb, row = _word_world()
    b = nrms.prepare(ft, h, row)
    m = nrms.NRMS(tokens.shape[1], word_embeddings=word_emb)
    assert m.word_level
    m = nrms.train(m, b, tokens, epochs=3, batch=64, log=lambda *_: None)
    scores = nrms.score(m, b, tokens)
    top1 = np.mean([b.labels[b.cand_start[i]:b.cand_start[i + 1]][np.argmax(s)]
                    for i, s in enumerate(scores)])
    assert top1 > 0.9, f"word-level top-1 {top1:.3f}, random is {1/6:.3f}"


def test_word_level_gathers_tokens_not_vectors():
    """The drop-in works because `emb` changes meaning from article VECTORS to
    article TOKEN IDS. If a refactor ever made the gather return floats, the
    embedding lookup would silently round them - so pin the dtype."""
    ft, h, tokens, word_emb, row = _word_world(n_imp=20)
    b = nrms.prepare(ft, h, row)
    hist_t, mask = nrms._hist_tensors(torch.from_numpy(tokens), b.hist_rows)
    assert not torch.is_floating_point(hist_t), "token gather must stay integral"
    assert hist_t.shape[-1] == tokens.shape[1]
    # Padding slots must carry PAD, which the encoder masks out.
    assert int(hist_t[~mask].abs().sum()) == 0
