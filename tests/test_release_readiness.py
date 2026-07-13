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


def test_readiness_external_schema_gate_reads_manifest_flag(monkeypatch):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "_manifest_template_flag_exact_true", lambda flag: flag == "require_external_schema")
    monkeypatch.setattr(audit, "_source_contains", lambda rel, needle: True)
    monkeypatch.setattr(audit, "_exists", lambda rel: True)
    assert audit._external_schema_required() is True


def test_readiness_external_schema_gate_fails_when_flag_false(monkeypatch):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "_manifest_template_flag_exact_true", lambda flag: False)
    monkeypatch.setattr(audit, "_source_contains", lambda rel, needle: True)
    monkeypatch.setattr(audit, "_exists", lambda rel: True)
    assert audit._external_schema_required() is False


def test_readiness_checksum_gate_reads_manifest_flag(monkeypatch):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "_manifest_template_flag_exact_true", lambda flag: flag == "require_bundle_checksum")
    monkeypatch.setattr(audit, "_source_contains", lambda rel, needle: True)
    monkeypatch.setattr(audit, "_exists", lambda rel: True)
    assert audit._checksum_policy_enabled() is True


def test_readiness_checksum_gate_fails_when_tamper_tests_missing(monkeypatch):
    import scripts.audit_release_readiness as audit

    monkeypatch.setattr(audit, "_manifest_template_flag_exact_true", lambda flag: True)
    monkeypatch.setattr(audit, "_source_contains", lambda rel, needle: True)
    monkeypatch.setattr(
        audit,
        "_exists",
        lambda rel: rel != "tests/test_bundle_integrity.py",
    )
    assert audit._checksum_policy_enabled() is False


def test_readiness_has_no_unconditional_security_true_gate():
    import scripts.audit_release_readiness as audit

    source = (ROOT / "scripts/audit_release_readiness.py").read_text(encoding="utf-8")
    assert '"external_schema_required": True' not in source
    assert '"checksum_policy_enabled": True' not in source
    gates = audit._engineering_gate_values()
    assert isinstance(gates["external_schema_required"], bool)
    assert isinstance(gates["checksum_policy_enabled"], bool)
