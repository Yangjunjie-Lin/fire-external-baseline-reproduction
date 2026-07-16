from __future__ import annotations

import json
from pathlib import Path

from external_baselines.interop.deepeval_handoff.validator import validate_handoff


def test_exported_bundle_validates(exported_handoff: Path, fixture_main_repo: Path) -> None:
    report = validate_handoff(exported_handoff, main_repo=fixture_main_repo)
    assert report["ok"] is True
    assert report["coverage_report"]["identical_case_sets"] is True


def test_tampered_prediction_fails_hash_and_safety(exported_handoff: Path, fixture_main_repo: Path) -> None:
    path = exported_handoff / "predictions/bm25_rag.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["structured_prediction"]["real_world_execution_allowed"] = True
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = validate_handoff(exported_handoff, main_repo=fixture_main_repo)
    assert report["ok"] is False
    assert any("prediction_sha256_mismatch" in error for error in report["errors"])
    assert any("real_world_execution_allowed" in error for error in report["errors"])
