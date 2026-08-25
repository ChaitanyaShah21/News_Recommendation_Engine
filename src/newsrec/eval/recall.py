"""recall@K for retrieval (Q2.4, and reused by Q3).

recall@K is defined per impression - "of the articles this user actually
clicked, what fraction appear in the top-K we retrieved" - so turning many
impressions into one number needs an averaging rule. D18 chose macro (mean of
per-impression recalls, every impression counting equally) for the headline,
because Q4's metrics, its bootstrap resampling, and both leaderboards all work
per impression. Micro (pooled hits over pooled clicks, every *click* counting
equally) is reported alongside since it costs nothing and differs materially on
MIND, where 29% of impressions carry more than one click.

D17 chose to report two coverage variants: over impressions where a query
exists, and over all impressions with query-less users counted as misses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecallResult:
    k: int
    macro: float
    micro: float
    n_impressions: int
    n_clicks: int

    def as_row(self, label: str) -> dict:
        return {
            "slice": label,
            "k": self.k,
            "recall@k (macro)": round(self.macro, 4),
            "recall@k (micro)": round(self.micro, 4),
            "impressions": self.n_impressions,
            "clicks": self.n_clicks,
        }


def recall_at_k(
    clicked: list[list[str]],
    retrieved: list[list[str]],
    k: int,
) -> RecallResult:
    """Macro and micro recall@k over aligned per-impression lists.

    Args:
        clicked: ground-truth clicked article ids, one list per impression.
        retrieved: retrieved article ids, best-first, one list per impression.
            Only the first `k` are considered, so one retrieval at the largest
            k can be sliced for every smaller k - the lists are score-ordered,
            so the top-50 of a top-200 list is exactly the top-50.

    Impressions with no ground-truth clicks are skipped: their recall would be
    0/0. Verified to be zero of them in either dataset's val split, but a
    division by zero on the large bundles would be found the hard way.
    """
    if len(clicked) != len(retrieved):
        raise ValueError(
            f"clicked has {len(clicked)} impressions but retrieved has "
            f"{len(retrieved)}; they must be aligned row for row"
        )

    total_recall = 0.0
    hits_pooled = 0
    clicks_pooled = 0
    evaluated = 0

    for truth, got in zip(clicked, retrieved):
        if not truth:
            continue
        top_k = set(got[:k])
        hits = sum(1 for article in truth if article in top_k)
        total_recall += hits / len(truth)
        hits_pooled += hits
        clicks_pooled += len(truth)
        evaluated += 1

    return RecallResult(
        k=k,
        macro=total_recall / evaluated if evaluated else 0.0,
        micro=hits_pooled / clicks_pooled if clicks_pooled else 0.0,
        n_impressions=evaluated,
        n_clicks=clicks_pooled,
    )
