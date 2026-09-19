"""baseline: NRMS (Wu et al., 2019), on CPU, over frozen article embeddings.

WHAT NRMS IS
Two encoders and a dot product.
  news encoder: article -> vector.
  user encoder: the vectors of the user's clicked articles -> one user vector,
                via multi-head SELF-attention (each clicked article looks at
                the others, so "football" next to "transfer news" is read
                differently from "football" next to "injury report"), then
                ADDITIVE attention (a learned weighted average - some clicks
                count more than others).
  click score:  user . candidate.
Training: for every click, sample `NEG` unclicked candidates from the SAME
impression and apply softmax cross-entropy over the 1 + NEG scores ("which of
these five did they click?") - the paper's negative-sampling objective.

THE ONE DEVIATION FROM THE PAPER (D39)
The paper's news encoder learns word-level self-attention over title tokens
from GloVe vectors. Ours is a learned projection of the 384-dim multilingual
sentence embedding we already have for every article (D20): the word-level half
is replaced by a pretrained sentence encoder. The user encoder, the objective
and the scoring are the paper's. This is what makes it trainable on a CPU in
minutes, and it is stated as a deviation, not hidden.

THE Q3 IMPROVEMENT HOOK
`time_features`: optional per-candidate columns (freshness, exposure share)
added to the click score through a small learned term. NRMS, like A1's
retrievers, has no time input at all; A1 measured that 92.7-93.5% of clicks go
to fresh articles. Off = the reproduced baseline; on = the improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import nn

SEED = 20260919
HIST_LEN = 50
NEG = 4
DIM = 256
HEADS = 16


class AdditiveAttention(nn.Module):
    """A learned weighted average: weight_i = softmax(q . tanh(W x_i + b))."""

    def __init__(self, dim: int, hidden: int = 200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self.query(torch.tanh(self.proj(x))).squeeze(-1)       # (B, L)
        a = a.masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=-1).unsqueeze(-1)                  # (B, L, 1)
        return (w * x).sum(1)                                       # (B, D)


class NRMS(nn.Module):
    def __init__(self, emb_dim: int, n_time: int = 0, dropout: float = 0.2):
        super().__init__()
        self.news = nn.Sequential(nn.Linear(emb_dim, DIM), nn.Tanh(), nn.Dropout(dropout))
        self.self_attn = nn.MultiheadAttention(DIM, HEADS, dropout=dropout, batch_first=True)
        self.pool = AdditiveAttention(DIM)
        self.n_time = n_time
        if n_time:
            # The improvement: a tiny MLP over the candidate's time features,
            # added to the relevance score. Zero-initialised output layer, so
            # training starts from exactly the baseline's scores.
            self.time = nn.Sequential(nn.Linear(n_time, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.time[2].weight)
            nn.init.zeros_(self.time[2].bias)

    def user(self, hist: torch.Tensor, hmask: torch.Tensor) -> torch.Tensor:
        h = self.news(hist)                                          # (B, L, D)
        # A user with NO history has an all-False mask; attention over nothing
        # is undefined (NaN). Let such rows attend to slot 0 (a zero vector).
        safe = hmask.clone()
        safe[~hmask.any(1), 0] = True
        z, _ = self.self_attn(h, h, h, key_padding_mask=~safe)
        return self.pool(z, safe)                                    # (B, D)

    def forward(self, hist, hmask, cand, tfeat=None) -> torch.Tensor:
        u = self.user(hist, hmask)                                   # (B, D)
        c = self.news(cand)                                          # (B, C, D)
        s = (c * u.unsqueeze(1)).sum(-1)                             # (B, C)
        if self.n_time:
            s = s + self.time(tfeat).squeeze(-1)
        return s


# ------------------------------------------------------------------ data

@dataclass
class Batchable:
    """Everything as integer rows into one article matrix, so a batch is a gather."""
    hist_rows: np.ndarray        # (n_users, HIST_LEN) int32, -1 = padding
    user_of_imp: np.ndarray      # (n_imp,) row into hist_rows
    cand_start: np.ndarray       # (n_imp + 1,) offsets into the flat arrays
    cand_rows: np.ndarray        # (n_rows,) article row
    labels: np.ndarray           # (n_rows,) int8
    tfeat: np.ndarray | None     # (n_rows, n_time) float32
    impression_ids: list[str]


def time_matrix(ft: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """log1p-scaled time features; nulls -> 0 with no separate flag (rare: 31 rows)."""
    out = []
    for c in cols:
        x = ft[c].cast(pl.Float64).fill_null(0.0).to_numpy()
        out.append(np.log1p(np.clip(x, 0, None)) if c == "freshness_hours" else x)
    return np.stack(out, 1).astype(np.float32)


def prepare(ft: pl.DataFrame, history: pl.DataFrame, art_row: dict,
            time_cols: list[str] | None = None) -> Batchable:
    """From a cached Q1 feature table (candidate order and labels) and the split's
    history (clicked articles, oldest first; the last HIST_LEN are kept)."""
    runs = ft["impression_id"].rle()
    lens = runs.struct.field("len").to_numpy()
    if runs.struct.field("value").n_unique() != len(lens):
        raise ValueError("impression rows are not contiguous")
    imp_users = ft["user_id"].gather(np.concatenate([[0], np.cumsum(lens)[:-1]]))
    users = imp_users.unique(maintain_order=True).to_list()
    uidx = {u: i for i, u in enumerate(users)}
    hist_of = dict(zip(history["user_id"], history["history_article_ids"].to_list()))
    H = np.full((len(users), HIST_LEN), -1, np.int32)
    for i, u in enumerate(users):
        rows = [art_row[a] for a in (hist_of.get(u) or []) if a in art_row][-HIST_LEN:]
        if rows:
            H[i, HIST_LEN - len(rows):] = rows   # right-aligned, newest last
    cand = np.fromiter((art_row[a] for a in ft["article_id"]), np.int64, ft.height)
    return Batchable(
        hist_rows=H,
        user_of_imp=np.fromiter((uidx[u] for u in imp_users), np.int64, len(lens)),
        cand_start=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        cand_rows=cand,
        labels=ft["clicked"].to_numpy().astype(np.int8),
        tfeat=time_matrix(ft, time_cols) if time_cols else None,
        impression_ids=runs.struct.field("value").to_list(),
    )


def _hist_tensors(emb: torch.Tensor, rows: np.ndarray):
    mask = torch.from_numpy(rows >= 0)
    return emb[torch.from_numpy(np.clip(rows, 0, None)).long()] * mask.unsqueeze(-1), mask


def training_samples(b: Batchable, rng: np.random.Generator) -> np.ndarray:
    """One sample per click: (impression, positive row, NEG negative rows).

    Negatives are drawn from the same impression, with replacement only when it
    has fewer than NEG unclicked candidates. Impressions with no click or no
    non-click give no samples. Re-drawn every epoch."""
    out = []
    for i in range(len(b.cand_start) - 1):
        s, e = b.cand_start[i], b.cand_start[i + 1]
        y = b.labels[s:e]
        pos = np.flatnonzero(y == 1) + s
        neg = np.flatnonzero(y == 0) + s
        if len(pos) == 0 or len(neg) == 0:
            continue
        for p in pos:
            n = rng.choice(neg, NEG, replace=len(neg) < NEG)
            out.append(np.concatenate([[i, p], n]))
    return np.asarray(out, np.int64)


def train(model: NRMS, b: Batchable, emb: np.ndarray, epochs: int = 3, batch: int = 256,
          lr: float = 1e-3, log=print) -> NRMS:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    E = torch.from_numpy(emb)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        S = training_samples(b, rng)
        S = S[rng.permutation(len(S))]
        total = 0.0
        for k in range(0, len(S), batch):
            sb = S[k:k + batch]
            hist, hmask = _hist_tensors(E, b.hist_rows[b.user_of_imp[sb[:, 0]]])
            rows = sb[:, 1:]                                  # positive first
            cand = E[torch.from_numpy(b.cand_rows[rows])]
            tf = torch.from_numpy(b.tfeat[rows]) if b.tfeat is not None else None
            logits = model(hist, hmask, cand, tf)
            loss = lossf(logits, torch.zeros(len(sb), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(sb)
        log(f"  epoch {ep + 1}: {len(S):,} samples, loss {total / len(S):.4f}")
    return model


@torch.no_grad()
def score(model: NRMS, b: Batchable, emb: np.ndarray, batch: int = 512) -> list[np.ndarray]:
    """One score array per impression, candidates in table order."""
    model.eval()
    E = torch.from_numpy(emb)
    out: list[np.ndarray] = []
    n = len(b.cand_start) - 1
    for k in range(0, n, batch):
        idx = np.arange(k, min(k + batch, n))
        hist, hmask = _hist_tensors(E, b.hist_rows[b.user_of_imp[idx]])
        u = model.user(hist, hmask)
        for j, i in enumerate(idx):
            s, e = b.cand_start[i], b.cand_start[i + 1]
            c = model.news(E[torch.from_numpy(b.cand_rows[s:e])])
            sc = c @ u[j]
            if model.n_time:
                sc = sc + model.time(torch.from_numpy(b.tfeat[s:e])).squeeze(-1)
            out.append(sc.numpy().astype(np.float64))
    return out
