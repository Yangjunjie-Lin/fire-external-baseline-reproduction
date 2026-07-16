from __future__ import annotations

import json
import shutil
from pathlib import Path

from external_baselines.interop.deepeval_handoff.contracts import contract_report

ROOT = Path(__file__).resolve().parents[3]


def test_fixture_contract_matches_local_snapshot(fixture_main_repo: Path) -> None:
    report = contract_report(repository_root=ROOT, main_repo=fixture_main_repo)
    assert report["ok"] is True
    assert report["source_sha256"] == report["local_snapshot_sha256"]
    assert len(report["source_commit"]) == 40


def test_contract_mismatch_fails(tmp_path: Path, fixture_main_repo: Path) -> None:
    target = tmp_path / "main"
    shutil.copytree(fixture_main_repo, target)
    schema = target / "schemas/deepeval_external_v1/external_prediction.schema.json"
    payload = json.loads(schema.read_text(encoding="utf-8"))
    payload["title"] = "mismatch"
    schema.write_text(json.dumps(payload), encoding="utf-8")
    report = contract_report(repository_root=ROOT, main_repo=target)
    assert report["ok"] is False
    assert "source_snapshot_sha256_mismatch" in report["errors"]


def test_missing_explicit_main_repository_fails(tmp_path: Path) -> None:
    report = contract_report(repository_root=ROOT, main_repo=tmp_path / "missing")
    assert report["ok"] is False
    assert "main_repository_missing" in report["errors"]
