"""The leaderboard score cache must never serve one model's scores as another's.

`run_submission_lgbm.py` resumes by skipping any user-group whose score file
exists. Before 2026-09-25 that cache was keyed by dataset only, so scoring with
the D42 rack model would have skipped all 64 groups and re-emitted the shipped
model's scores under the new name. These tests pin both halves of the fix:
different model names get different paths, and a same-named cache filled by
different model bytes is refused rather than resumed.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_submission_lgbm.py"
_spec = importlib.util.spec_from_file_location("run_submission_lgbm", SCRIPT)
sub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sub)


def test_default_model_keeps_the_submitted_paths():
    # The 20 September submission was built at suffix "" - it must not move.
    assert sub.model_suffix(sub.REPO_ROOT / "data" / "models" / "lgbm_mind.txt", "mind") == ""


def test_named_models_get_distinct_suffixes(tmp_path):
    assert sub.model_suffix(Path("data/models/lgbm_mind_rack.txt"), "mind") == "_rack"
    assert sub.model_suffix(Path("data/models/lgbm_ebnerd_rack.txt"), "ebnerd") == "_rack"
    # A default-NAMED file elsewhere is not the default model.
    assert sub.model_suffix(tmp_path / "lgbm_mind.txt", "mind") == "_lgbm_mind"
    # The other dataset's model must not collapse to "" or share a suffix.
    assert sub.model_suffix(Path("data/models/lgbm_ebnerd.txt"), "mind") == "_lgbm_ebnerd"


def test_cache_is_claimed_then_resumed_by_the_same_model(tmp_path):
    model = tmp_path / "m.txt"
    model.write_text("tree v1")
    parts = tmp_path / "cache"
    parts.mkdir()
    sub.claim_cache(parts, model)
    (parts / "000.parquet").write_bytes(b"x")
    sub.claim_cache(parts, model)  # same bytes: resuming is correct


def test_cache_filled_by_different_model_bytes_is_refused(tmp_path):
    model = tmp_path / "m.txt"
    model.write_text("tree v1")
    parts = tmp_path / "cache"
    parts.mkdir()
    sub.claim_cache(parts, model)
    (parts / "000.parquet").write_bytes(b"x")
    model.write_text("tree v2")  # retrained under the same name
    with pytest.raises(SystemExit, match="different model"):
        sub.claim_cache(parts, model)


def test_unmarked_cache_with_scores_is_refused(tmp_path):
    # Exactly the pre-fix state on disk: score files, no record of their model.
    model = tmp_path / "m.txt"
    model.write_text("tree v1")
    parts = tmp_path / "cache"
    parts.mkdir()
    (parts / "000.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit, match="unrecorded"):
        sub.claim_cache(parts, model)
