"""Adversarial tests for the Q5 submission writer (R10).

The centrepiece is `test_rank_vector_matches_MINDs_own_worked_example` and its
EB-NeRD twin. Both competitions publish a fully worked example on their
submission-guidelines page, and those examples are the only external ground
truth available for the one thing in this module that can be silently wrong:
whether the emitted vector is the ranking, or its inverse.

Both spellings produce a file that passes every structural check - correct line
count, correct row order, integers 1..n, one bracketed list per line. The only
difference is the score. So structural tests are necessary and nowhere near
sufficient, and these two tests carry the weight.
"""

import zipfile

import numpy as np
import polars as pl
import pytest

from newsrec import predict


# --------------------------------------------------------------------------
# The two official worked examples - the tests that actually matter
# --------------------------------------------------------------------------


def test_rank_vector_matches_MINDs_own_worked_example():
    """From MIND's Submission Guidelines page, quoted verbatim:

        given the impression   24481   N125045 N87192 N73556 N20417
        the prediction can be  24481 [4,1,3,2]
        "which means that the ranking orders of the candidate news articles in
         this impression are N87192, N20417, N73556 and N125045"

    So N87192 (position 2) is 1st, N20417 (position 4) is 2nd, N73556
    (position 3) is 3rd, N125045 (position 1) is 4th. Build scores that
    produce exactly that ranking and assert the published vector comes out.
    """
    # position:            N125045  N87192  N73556  N20417
    scores = np.array([0.10, 0.90, 0.30, 0.50])
    #   descending order -> N87192(.90), N20417(.50), N73556(.30), N125045(.10)

    assert predict.rank_vector(scores).tolist() == [4, 1, 3, 2]


def test_rank_vector_matches_EBNeRDs_own_worked_example():
    """From the Ekstra Bladet Submission Guidelines page:

        impression 139350, article_ids_inview [9798759, 9798604, 9777339, 9798829]
        prediction: 139350 [3,2,4,1]
        "the ranking orders ... are 9798829 (first), 9798604 (second),
         9798759 (third), and 9777339 (fourth)"
    """
    # position:       9798759  9798604  9777339  9798829
    scores = np.array([0.30, 0.60, 0.10, 0.99])
    assert predict.rank_vector(scores).tolist() == [3, 2, 4, 1]


def test_rank_vector_is_NOT_the_argsort():
    """The bug, stated positively, so it cannot be reintroduced by 'tidying'.

    For these scores the argsort-plus-one spelling gives [2,4,3,1] and the
    correct answer is [4,1,3,2]. Both are permutations of 1..4; both would sail
    through every structural check in this file.
    """
    scores = np.array([0.10, 0.90, 0.30, 0.50])
    correct = predict.rank_vector(scores)
    argsort_spelling = np.argsort(-scores, kind="stable") + 1

    assert correct.tolist() == [4, 1, 3, 2]
    assert argsort_spelling.tolist() == [2, 4, 3, 1]
    assert not np.array_equal(correct, argsort_spelling)


def test_rank_vector_round_trips_through_the_readers_definition():
    """Independent check of the same property, derived rather than quoted.

    Reconstruct the ranking from the rank vector the way a scorer would - place
    each candidate at its stated position - and assert it equals the ranking
    implied by sorting on score. This catches an inversion even if both worked
    examples above were transcribed wrongly.
    """
    rng = np.random.default_rng(7)
    for n in (1, 2, 5, 40):
        scores = rng.standard_normal(n)
        ranks = predict.rank_vector(scores)

        # "the article with rank 1" ... "the article with rank n"
        reconstructed = np.argsort(ranks, kind="stable")
        expected = np.argsort(-scores, kind="stable")
        assert np.array_equal(reconstructed, expected)


# --------------------------------------------------------------------------
# rank_vector: structural properties and hostile inputs
# --------------------------------------------------------------------------


def test_ranks_are_a_permutation_of_1_to_n():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(200)
    ranks = predict.rank_vector(scores)
    assert sorted(ranks.tolist()) == list(range(1, 201))


def test_single_candidate_impression():
    """MIND's val split has 2,744 impressions with only 2 candidates; a
    one-candidate rack is the boundary below that."""
    assert predict.rank_vector(np.array([0.5])).tolist() == [1]


def test_all_tied_scores_keep_platform_order():
    """A cold-start user scores a flat zero on every candidate. The result must
    still be a valid 1..n permutation in the platform's own order, not an
    arbitrary one - 29,108 MIND test impressions are cold-start."""
    assert predict.rank_vector(np.zeros(5)).tolist() == [1, 2, 3, 4, 5]


def test_negative_scores_rank_correctly():
    """Cosine runs [-1, 1]. A scorer emitting all-negative scores must still
    rank best-first - this is the -inf-versus-0.0 masking bug from Phase 3
    wearing different clothes."""
    assert predict.rank_vector(np.array([-0.9, -0.1, -0.5])).tolist() == [3, 1, 2]


def test_nan_scores_are_refused_loudly():
    """NaN sorts unpredictably and would scatter one impression's ranking with
    no error at all - the worst possible failure: silent and localised."""
    with pytest.raises(ValueError, match="NaN"):
        predict.rank_vector(np.array([0.5, np.nan, 0.1]))


def test_two_dimensional_input_is_refused():
    with pytest.raises(ValueError, match="1-D"):
        predict.rank_vector(np.zeros((3, 3)))


# --------------------------------------------------------------------------
# Line formatting
# --------------------------------------------------------------------------


def test_line_format_is_exactly_what_the_guidelines_show():
    lines = predict.format_lines(
        ["mind:24481"], [np.array([0.10, 0.90, 0.30, 0.50])], "mind"
    )
    assert lines == ["24481 [4,1,3,2]"]


def test_no_spaces_inside_the_bracket():
    """Both published examples are comma-separated with no spaces. Cheap to get
    wrong via a stray ", ".join and not obviously wrong on inspection."""
    line = predict.format_lines(["ebnerd:1"], [np.arange(5.0)], "ebnerd")[0]
    assert " " not in line.split(" ", 1)[1]


def test_prefix_is_removed_without_eating_id_characters():
    """removeprefix, not lstrip. `"mind:12345".lstrip("mind:")` returns "2345" -
    it strips *characters in the set*, so any id whose digits begin with one of
    m/i/n/d/: loses them. It would corrupt only some rows, which is worse than
    corrupting all of them.
    """
    lines = predict.format_lines(
        ["mind:1", "mind:24481", "mind:1234"], [np.array([1.0])] * 3, "mind"
    )
    assert [ln.split(" ")[0] for ln in lines] == ["1", "24481", "1234"]


def test_ebnerd_prefix_removed():
    lines = predict.format_lines(["ebnerd:139350"], [np.array([1.0])], "ebnerd")
    assert lines[0].startswith("139350 [")


def test_format_lines_refuses_a_length_mismatch():
    """Two parallel lists able to drift out of sync (R10). A mismatch here
    would silently pair impression i's id with impression i+1's ranking for
    every subsequent row."""
    with pytest.raises(ValueError, match="score arrays"):
        predict.format_lines(["mind:1", "mind:2"], [np.array([1.0])], "mind")


def test_format_lines_preserves_input_order():
    """'The row orders of the results should be consistent with those in the
    original files' - both competitions say so explicitly."""
    ids = [f"mind:{i}" for i in [50, 10, 30, 20]]
    lines = predict.format_lines(ids, [np.array([1.0])] * 4, "mind")
    assert [ln.split(" ")[0] for ln in lines] == ["50", "10", "30", "20"]


# --------------------------------------------------------------------------
# Candidate set without labels (D30)
# --------------------------------------------------------------------------


def _impressions(rows):
    return pl.DataFrame(
        {
            "impression_id": [r[0] for r in rows],
            "user_id": [r[1] for r in rows],
            "candidate_article_ids": [r[2] for r in rows],
        }
    )


def test_submission_candidate_set_has_no_labels():
    cs = predict.build_submission_candidate_set(
        _impressions([("mind:1", "mind:U1", ["mind:A", "mind:B"])]),
        ["mind:A", "mind:B", "mind:C"],
    )
    assert cs.labels is None
    assert cs.candidate_rows[0].tolist() == [0, 1]


def test_submission_candidate_set_rejects_an_unknown_candidate():
    """Measured 0 missing across both full test bundles - so this is a guard on
    an invariant, not a live case. A missing candidate would otherwise need an
    invented score, and inventing one silently is how a fake number ships."""
    with pytest.raises(KeyError, match="absent from"):
        predict.build_submission_candidate_set(
            _impressions([("mind:1", "mind:U1", ["mind:A", "mind:ZZZ"])]),
            ["mind:A"],
        )


def test_submission_candidate_set_rejects_an_empty_impression():
    with pytest.raises(ValueError, match="no candidates"):
        predict.build_submission_candidate_set(
            _impressions([("mind:1", "mind:U1", [])]), ["mind:A"]
        )


def test_candidate_rows_keep_platform_order_not_sorted_order():
    """The rank vector is positional, so any reordering of candidates here
    silently permutes every rank we emit."""
    cs = predict.build_submission_candidate_set(
        _impressions([("mind:1", "mind:U1", ["mind:C", "mind:A", "mind:B"])]),
        ["mind:A", "mind:B", "mind:C"],
    )
    assert cs.candidate_rows[0].tolist() == [2, 0, 1]


# --------------------------------------------------------------------------
# The zip
# --------------------------------------------------------------------------


def test_zip_contains_exactly_one_flat_file(tmp_path):
    """'A valid zip submission should contain nothing but a text file', 'no
    __macosx file', 'do not place the submission file within folders'."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    txt = nested / "prediction.txt"
    txt.write_text("1 [1]\n")

    zp = predict.zip_submission(txt, tmp_path / "sub.zip", "prediction.txt")

    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
    # Written from a nested directory - the arcname must flatten it, or the
    # scorer looks for prediction.txt and finds a/b/prediction.txt.
    assert names == ["prediction.txt"]


def test_the_two_competitions_expect_different_filenames():
    """One letter apart. Pinned so a copy-paste between the two submission
    scripts cannot silently use the wrong one."""
    assert predict.PREDICTION_FILENAME["mind"] == "prediction.txt"
    assert predict.PREDICTION_FILENAME["ebnerd"] == "predictions.txt"
    assert predict.PREDICTION_FILENAME["mind"] != predict.PREDICTION_FILENAME["ebnerd"]
