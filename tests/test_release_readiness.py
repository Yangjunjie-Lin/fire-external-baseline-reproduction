"""Release readiness audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_release_readiness_fails_when_engineering_gate_missing(monkeypatch, tmp_path):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "OUT_JSON", tmp_path / "release_readiness.json")
    monkeypatch.setattr(audit, "OUT_MD", tmp_path / "release_readiness.md")
    monkeypatch.setattr(
        audit,
        "_engineering_gate_values",
        lambda: {"method_registry_converged": False},
    )
    monkeypatch.setattr(audit, "_empirical_gate_values", lambda: {"cross_repo_contract_verified": False})
    with pytest.raises(SystemExit) as excinfo:
        audit.main([])
    assert excinfo.value.code == 1


def test_release_readiness_passes_when_engineering_gates_complete(monkeypatch, tmp_path):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "OUT_JSON", tmp_path / "release_readiness.json")
    monkeypatch.setattr(audit, "OUT_MD", tmp_path / "release_readiness.md")
    gates = {key: True for key in audit.ENGINEERING_GATE_KEYS}
    monkeypatch.setattr(audit, "_engineering_gate_values", lambda: gates)
    monkeypatch.setattr(audit, "_empirical_gate_values", lambda: {"cross_repo_contract_verified": False})
    with pytest.raises(SystemExit) as excinfo:
        audit.main([])
    assert excinfo.value.code == 0


def test_empirical_false_does_not_fail_engineering_readiness(monkeypatch, tmp_path):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "OUT_JSON", tmp_path / "release_readiness.json")
    monkeypatch.setattr(audit, "OUT_MD", tmp_path / "release_readiness.md")
    gates = {key: True for key in audit.ENGINEERING_GATE_KEYS}
    monkeypatch.setattr(audit, "_engineering_gate_values", lambda: gates)
    monkeypatch.setattr(
        audit,
        "_empirical_gate_values",
        lambda: {key: False for key in audit.EMPIRICAL_GATE_KEYS},
    )
    with pytest.raises(SystemExit) as excinfo:
        audit.main([])
    assert excinfo.value.code == 0


def test_ci_status_is_not_hardcoded_true(monkeypatch, tmp_path):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github/workflows/ci.yml").write_text("name: ci\n", encoding="utf-8")
    report = audit.build_report()
    assert "ci_status_known" not in report
    assert report["engineering"]["gates"]["ci_workflow_config_present"] is True
    assert report["ci_result_verified_externally"] is False


def test_engineering_gate_total_is_dynamic(monkeypatch, tmp_path):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    gates = {key: True for key in audit.ENGINEERING_GATE_KEYS}
    monkeypatch.setattr(audit, "_engineering_gate_values", lambda: gates)
    monkeypatch.setattr(audit, "_empirical_gate_values", lambda: {"cross_repo_contract_verified": False})
    report = audit.build_report()
    assert report["engineering"]["total"] == len(audit.ENGINEERING_GATE_KEYS)
    assert report["engineering"]["passed"] == len(audit.ENGINEERING_GATE_KEYS)


def test_report_only_mode_exits_zero(monkeypatch, tmp_path):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "OUT_JSON", tmp_path / "release_readiness.json")
    monkeypatch.setattr(audit, "OUT_MD", tmp_path / "release_readiness.md")
    monkeypatch.setattr(
        audit,
        "_engineering_gate_values",
        lambda: {"method_registry_converged": False},
    )
    monkeypatch.setattr(audit, "_empirical_gate_values", lambda: {"cross_repo_contract_verified": False})
    with pytest.raises(SystemExit) as excinfo:
        audit.main(["--report-only"])
    assert excinfo.value.code == 0
    payload = json.loads((tmp_path / "release_readiness.json").read_text(encoding="utf-8"))
    assert payload["engineering"]["ready"] is False
