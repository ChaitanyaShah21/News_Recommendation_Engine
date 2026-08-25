"""Phase 5 / Q5 - turn scores into the exact file each leaderboard wants.

Both competitions want the same shape, and it is not the obvious one.

    MIND      zip containing `prediction.txt`   (singular)
    EB-NeRD   zip containing `predictions.txt`  (plural)

    line format, both:   <impression_id> [r1,r2,...,rn]

**The trap this module exists to get right.** `r_i` is the rank awarded to the
candidate at position `i` of that impression's own candidate list - NOT the
identity of the article placed at rank `i`. It is the *inverse permutation* of
an argsort, not the argsort.

MIND's own worked example: candidates `N125045 N87192 N73556 N20417`, answer
`[4,1,3,2]`, meaning N125045 came 4th, N87192 came 1st, N73556 3rd, N20417 2nd.
So the ranking, best first, is N87192, N20417, N73556, N125045.

Writing `np.argsort(-scores) + 1` instead produces a file that is perfectly
well-formed - right line count, right row order, ranks 1..n, no error anywhere -
and scores approximately random. Nothing on the leaderboard would tell you
which of the two you submitted. Hence `rank_vector` is four lines with a test
suite attached, pinned against both competitions' own worked examples.

**On the rate limits printed on those pages** ("at most one submission each
day" for MIND, five for EB-NeRD): those are the *live-competition* limits and
both competitions ended years ago. The current limit is 10 per day on both -
corrected by Chaitanya, 2026-08-25, against the page text. Same shape as D20's
stale download-speed figure: documentation that was accurate when written and
silently was not re-checked. We still validate against the official example
file rather than against the leaderboard, but the pressure is scheduling
convenience, not a hard budget of attempts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .eval.rerank import CandidateSet

# The filename INSIDE the zip. They differ by one letter between the two
# competitions, which is exactly the sort of thing that wastes a daily
# submission slot, so both are constants rather than literals at a call site.
PREDICTION_FILENAME = {"mind": "prediction.txt", "ebnerd": "predictions.txt"}

ID_PREFIX = {"mind": "mind:", "ebnerd": "ebnerd:"}


def rank_vector(scores: np.ndarray) -> np.ndarray:
    """Ranks 1..n, where `out[i]` is the rank awarded to candidate `i`.

    Higher score wins. Ties keep the platform's own candidate order, which D23
    verified carries no click signal (mean normalised position of a clicked
    item 0.5017 on MIND, 0.4961 on EB-NeRD, against 0.5 for a uniform shuffle).

    D23's pessimistic tie policy deliberately does **not** apply here. That
    policy exists to stop tie luck inflating a *reported metric*, and it works
    by consulting the labels - which is exactly what we must not do, and here
    cannot do: this split has no labels. For a submission the only honest thing
    is a stable order.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if not np.isfinite(scores).all():
        # NaN sorts unpredictably and would scatter a whole impression's ranking
        # with no error raised. Never seen in practice; catastrophic if it were.
        raise ValueError("scores contain NaN or infinity; refusing to rank")

    # kind="stable" so equal scores keep input order rather than an arbitrary
    # quicksort permutation - reproducibility, and the tie rule above.
    order = np.argsort(-scores, kind="stable")

    # THE inverse permutation. `order[j] = i` says "candidate i finished j-th",
    # so writing j+1 INTO position order[j] gives "candidate i's rank".
    # The bug this prevents is returning `order + 1` directly.
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    return ranks


def build_submission_candidate_set(
    impressions: pl.DataFrame, article_ids: list[str]
) -> CandidateSet:
    """`build_candidate_set` for a split that has no labels (D30).

    The stock builder reads `clicked_article_ids`, which the submission reader
    deliberately does not produce. Rather than passing an empty column - the
    thing D30 rejected - this builds the same structure with `labels=None`, so
    any code that reaches for a label gets a TypeError on the spot instead of
    a metric computed over fabricated zeros.

    `history_len` is likewise not carried: it exists for Q4.3's cold-start
    slice, and there is no slicing to do on an unlabeled split.
    """
    row_of_article = {aid: i for i, aid in enumerate(article_ids)}

    impression_ids = impressions.get_column("impression_id").to_list()
    user_ids = impressions.get_column("user_id").to_list()
    cand_lists = impressions.get_column("candidate_article_ids").to_list()

    candidate_rows: list[np.ndarray] = []
    for imp_id, cands in zip(impression_ids, cand_lists):
        if not cands:
            # A zero-candidate impression has no valid rank vector at all, and
            # writing an empty "[]" would desynchronise nothing but would be
            # scored as a failure. Verified absent from both test bundles.
            raise ValueError(f"impression {imp_id} has no candidates")
        missing = [c for c in cands if c not in row_of_article]
        if missing:
            raise KeyError(
                f"impression {imp_id} has {len(missing)} candidate(s) absent from "
                f"the article matrix, e.g. {missing[0]!r}"
            )
        candidate_rows.append(
            np.fromiter(
                (row_of_article[c] for c in cands), dtype=np.int32, count=len(cands)
            )
        )

    return CandidateSet(
        impression_ids=impression_ids,
        user_ids=user_ids,
        candidate_rows=candidate_rows,
        labels=None,
        history_len=np.zeros(len(impression_ids), dtype=np.int32),
    )


def format_lines(
    impression_ids: list[str], scores: list[np.ndarray], dataset: str
) -> list[str]:
    """One submission line per impression, in the order given.

    Order is load-bearing: both competitions say "the row orders of the results
    should be consistent with those in the original files". Nothing here sorts,
    groups or deduplicates, and the chunked writer concatenates in file order.
    """
    prefix = ID_PREFIX[dataset]
    if len(impression_ids) != len(scores):
        raise ValueError(
            f"{len(impression_ids)} impressions but {len(scores)} score arrays"
        )

    lines = []
    for imp_id, s in zip(impression_ids, scores):
        # Our unified schema prefixes every id ("mind:24481"); the leaderboard
        # wants the platform's own. removeprefix, not lstrip - lstrip("mind:")
        # strips *characters*, so an id starting with one of m/i/n/d/: would be
        # silently eaten. Classic, and it would corrupt only some rows.
        raw = imp_id.removeprefix(prefix)
        ranks = rank_vector(s)
        lines.append(f"{raw} [{','.join(map(str, ranks.tolist()))}]")
    return lines


def zip_submission(txt_path: Path, zip_path: Path, arcname: str) -> Path:
    """Zip a single predictions file, flat, with nothing else inside.

    Both competitions are explicit: "a valid zip submission should contain
    nothing but a text file", "no __macosx file", "do not place the submission
    file within folders before it is compressed". `arcname` forces the entry to
    be a bare filename regardless of where `txt_path` sits on disk - writing the
    file with its full path is the standard way to produce a rejected zip.
    """
    import zipfile

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=arcname)
    return zip_path
