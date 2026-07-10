"""Validation evidence and readiness status separation tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_JSON = ROOT / "outputs/diagnostics/validation_evidence.json"
CONTRACT_JSON = ROOT / "outputs/diagnostics/cross_repo_contract.json"


def test_contract_tool_ready_not_equal_verified():
    text = (ROOT / "docs/status/readiness_summary.md").read_text(encoding="utf-8")
    assert "cross_repository_contract_tool_ready" in text
    assert "cross_repository_contract_verified" in text
    assert "true*" not in text


def test_verified_requires_evidence_file():
    if not EVIDENCE_JSON.is_file():
        return
    evidence = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    contract = evidence.get("cross_repository_contract") or {}
    if contract.get("verified"):
        assert contract.get("executed") is True
        assert evidence.get("commands")


def test_verified_requires_main_commit():
    if not CONTRACT_JSON.is_file():
        return
    data = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    if data.get("cross_repository_contract_verified"):
        assert data.get("main_project_commit")
        assert data.get("baseline_commit")


def test_verified_requires_runner_checksum():
    if not CONTRACT_JSON.is_file():
        return
    data = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    if data.get("cross_repository_contract_verified"):
        assert data.get("runner_bundle_checksum")


def test_local_ci_status_separate_from_remote_ci():
    if not EVIDENCE_JSON.is_file():
        return
    evidence = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
    assert "local_ci_equivalent_passed" in evidence
    assert evidence.get("ci_remote_status_known") is False or "ci_remote_status_known" in evidence
