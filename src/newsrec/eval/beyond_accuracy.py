"""beyond-accuracy metrics: intra-list diversity, novelty, coverage.

AUC, MRR and nDCG all ask one question - did the clicked item land near the top?
A system can answer it perfectly and still be bad. Phase 3's Finding 4 is the
worked example already in hand: MIND user `mind:U13132` read three political
stories and one about a Starbucks latte, and semantic retrieval returned five
Popeyes chicken-sandwich articles and nothing political. Every accuracy metric
would have called that a good list if one of the five was clicked.

READ THESE AGAINST ACCURACY, NEVER ALONE
A purely random recommender scores *best* on all three metrics in this module -
maximally diverse, maximally novel, near-total coverage. So these numbers do not
rank systems; they price what a system gave up. "Semantic wins nDCG@5 by 20% and
surrenders 40% of its diversity" is a finding. "Semantic has low diversity" is
noise. This is why D24's random arm is run through this module too.

WHAT LIST THEY ARE MEASURED ON (fork A, decided 2026-08-25)
Beyond-accuracy describes a system's *own* output. In re-ranking the platform
chose the items and we only reordered them, and only 6.8% of MIND's corpus
(19.2% of EB-NeRD's) ever appears in any val candidate list - so coverage
measured there is hard-capped by someone else's recommender, not ours. The
headline is therefore measured on the Q2/Q3 retrieval top-K, where our system
chose from the whole corpus. The re-ranking numbers are computed too, precisely
to show that cap rather than assert it.

WHAT `distance` MEANS IN DIVERSITY (fork B, decided 2026-08-25)
Both bases are reported. The conventional one is cosine distance between article
embeddings - but semantic retrieval finds articles by *maximising* cosine
similarity, so grading it in that space measures it by the quantity it exists to
minimise, and it loses by construction rather than by finding. Category distance
(18 categories on MIND, 25 on EB-NeRD, zero nulls in both) is independent of the
embedding space, so no method gets an automatic penalty.

The disagreement between the two bases is the actual diagnosis: a list that
scores badly on embedding distance but well on category distance spans topics
and repeats *within* them - which is exactly what the Popeyes example looks like,
and is a sharper statement than either number alone.
"""

from __future__ import annotations

import numpy as np

MIN_LIST_FOR_DIVERSITY = 2


def intra_list_diversity_embedding(rows: np.ndarray, embeddings: np.ndarray) -> float:
    """Mean pairwise cosine *distance* among a list's articles, in [0, 2].

    The obvious implementation forms all k(k-1)/2 pairs. This uses a closed form
    that is O(k*d) rather than O(k^2*d), valid because every stored vector is
    unit length (semantic.py asserts that at load):

        sum over ALL ordered pairs (i, j) of  v_i . v_j   =   || sum_i v_i ||^2

    The i == j terms contribute exactly k (each is v_i . v_i = 1), so

        sum over unordered i != j pairs  =  ( ||sum||^2 - k ) / 2
        mean pairwise cosine             =  ( ||sum||^2 - k ) / ( k(k-1) )
        ILD                              =  1 - mean pairwise cosine

    A brute-force pairwise implementation is kept in the tests and compared
    against this on random data, so an error in the algebra cannot hide.

    Returns NaN for a list of fewer than two articles - there are no pairs, and
    0.0 would read as "perfectly uniform" rather than "not measurable".
    """
    if len(rows) < MIN_LIST_FOR_DIVERSITY:
        return float("nan")
    vectors = embeddings[rows]
    k = len(rows)
    total = vectors.sum(axis=0, dtype=np.float64)
    mean_cos = (float(total @ total) - k) / (k * (k - 1))
    return 1.0 - mean_cos


def intra_list_diversity_category(rows: np.ndarray, categories: np.ndarray) -> float:
    """Fraction of pairs in the list that fall in *different* categories, [0, 1].

    Distance is 0 for a same-category pair and 1 for a cross-category pair, so
    the mean is just that fraction. Counted from group sizes rather than by
    enumerating pairs: a category holding n of the list's k articles contributes
    n(n-1)/2 same-category pairs, so

        ILD = 1 - ( sum_c n_c(n_c - 1) / 2 ) / ( k(k-1) / 2 )

    1.0 means every article is in a different category; 0.0 means all one
    category. Coarse by design - it cannot separate two different
    chicken-sandwich stories - which is why the embedding basis is reported
    beside it rather than instead of it.
    """
    if len(rows) < MIN_LIST_FOR_DIVERSITY:
        return float("nan")
    k = len(rows)
    _, counts = np.unique(categories[rows], return_counts=True)
    same_pairs = float((counts * (counts - 1)).sum())
    return 1.0 - same_pairs / (k * (k - 1))


def item_novelty(train_counts: np.ndarray) -> np.ndarray:
    """Per-article self-information, -log2(p), from train-window click counts.

    p is the article's share of all training clicks, so a frequently-clicked
    article is unsurprising and scores low.

    **Laplace smoothing is not optional here.** 88.2% of MIND's corpus and a
    comparable share of EB-NeRD's were never clicked during training, and
    -log2(0) is +inf - which would propagate through every mean and make the
    metric useless without raising anything. Adding one pseudo-click to every
    article gives an unclicked one the largest *finite* novelty in the corpus,
    which is the right answer rather than a patch:

        MIND     floor 6.13 (most-clicked)   ceiling 18.20 (never clicked)
        EB-NeRD  floor 8.01                  ceiling 15.16

    (Those floors are *after* smoothing; unsmoothed they would be 5.78 and 7.46.
    Smoothing shifts every article slightly, so the range quoted here is the one
    the code actually produces rather than the textbook one.)

    Those ranges are worth carrying. Unlike diversity and coverage, novelty is
    **not bounded in [0, 1]** and its scale is a property of the corpus, so a
    raw novelty figure means nothing on its own and MIND's numbers are not
    comparable to EB-NeRD's. Same trap as AUC's sensitivity to candidate-list
    length in Q4.1: a metric that looks absolute and silently carries the
    dataset's scale.

    Train-window counts only, for the same reason `train_click_counts` refuses
    any other split: novelty computed over val would be measured against the
    clicks being evaluated.
    """
    smoothed = train_counts.astype(np.float64) + 1.0
    return -np.log2(smoothed / smoothed.sum())


def list_novelty(rows: np.ndarray, novelty_per_article: np.ndarray) -> float:
    """Mean self-information over a list's articles."""
    if len(rows) == 0:
        return float("nan")
    return float(novelty_per_article[rows].mean())


def coverage(lists: list[np.ndarray], catalogue_size: int) -> float:
    """Fraction of the catalogue that appears in at least one recommended list.

    Unlike diversity and novelty this is a property of the *whole run*, not of
    one list - a single list has no coverage. That distinction matters for
    Q4.4's bootstrap, which resamples impressions: coverage has to be
    recomputed inside each resample rather than averaged across them, because
    the mean of per-list coverages is not the coverage of the union.
    """
    if catalogue_size <= 0:
        raise ValueError(f"catalogue_size must be positive, got {catalogue_size}")
    seen: set[int] = set()
    for rows in lists:
        seen.update(rows.tolist())
    return len(seen) / catalogue_size


def evaluate_lists(
    lists: list[np.ndarray],
    embeddings: np.ndarray,
    categories: np.ndarray,
    novelty_per_article: np.ndarray,
    catalogue_size: int,
) -> dict[str, object]:
    """All four figures for one method's output.

    Per-list values are returned as arrays alongside their means, for the same
    reason `evaluate_impressions` does it: Q4.4's bootstrap needs the individual
    values, and a mean has already discarded them.
    """
    ild_embed = np.array(
        [intra_list_diversity_embedding(r, embeddings) for r in lists], dtype=np.float64
    )
    ild_cat = np.array(
        [intra_list_diversity_category(r, categories) for r in lists], dtype=np.float64
    )
    nov = np.array(
        [list_novelty(r, novelty_per_article) for r in lists], dtype=np.float64
    )
    return {
        "ild_embedding": ild_embed,
        "ild_category": ild_cat,
        "novelty": nov,
        "coverage": coverage(lists, catalogue_size),
        "n_lists": len(lists),
    }
