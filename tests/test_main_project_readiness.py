"""Tests for main-project readiness assessment."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from external_baselines.common.main_project_readiness import assess_main_project_readiness


def _write_resources(path: Path, **overrides) -> None:
    base = {
        "main_project": {
            "repository_path": ".",
            "expected_branch": "main",
            "runner_bundle_path": None,
        },
        "execution": {
            "allow_real_model_calls": False,
            "allow_cross_repo_test": False,
            "allow_formal_evaluation": False,
        },
        "status": {"main_project_v1_ready": False},
    }
    base.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(base), encoding="utf-8")


def test_missing_main_project_repo(tmp_path: Path) -> None:
    resources = tmp_path / "resources.yaml"
    _write_resources(
        resources,
        main_project={
            "repository_path": str(tmp_path / "missing"),
            "expected_branch": "main",
            "runner_bundle_path": None,
        },
    )
    report = assess_main_project_readiness(resources)
    assert report["main_project_v1_ready"] is False
    assert "main_project_repository_missing" in report["reasons"]


def test_branch_mismatch(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "main"
    repo.mkdir()
    resources = tmp_path / "resources.yaml"
    _write_resources(
        resources,
        main_project={
            "repository_path": str(repo),
            "expected_branch": "evaluation/benchmark-v1",
            "runner_bundle_path": None,
        },
    )

    def fake_git(cmd, cwd):
        if cmd[:2] == ["branch", "--show-current"]:
            return "main"
        if cmd[:2] == ["rev-parse", "HEAD"]:
            return "abc123"
        return ""

    monkeypatch.setattr(
        "external_baselines.common.main_project_readiness._git",
        fake_git,
    )
    report = assess_main_project_readiness(resources)
    assert report["main_project_v1_ready"] is False
    assert "main_project_branch_mismatch" in report["reasons"]


def test_runner_bundle_not_configured(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "main"
    repo.mkdir()
    resources = tmp_path / "resources.yaml"
    _write_resources(
        resources,
        main_project={
            "repository_path": str(repo),
            "expected_branch": "main",
            "runner_bundle_path": None,
        },
    )

    monkeypatch.setattr(
        "external_baselines.common.main_project_readiness._git",
        lambda cmd, cwd: "main" if cmd[:2] == ["branch", "--show-current"] else "deadbeef",
    )
    report = assess_main_project_readiness(resources)
    assert report["main_project_v1_ready"] is False
    assert "runner_bundle_path_not_configured" in report["reasons"]


def test_runner_bundle_missing_when_configured(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "main"
    repo.mkdir()
    resources = tmp_path / "resources.yaml"
    _write_resources(
        resources,
        main_project={
            "repository_path": str(repo),
            "expected_branch": "main",
            "runner_bundle_path": str(tmp_path / "no_such_bundle"),
        },
    )
    monkeypatch.setattr(
        "external_baselines.common.main_project_readiness._git",
        lambda cmd, cwd: "main" if cmd[:2] == ["branch", "--show-current"] else "deadbeef",
    )
    report = assess_main_project_readiness(resources)
    assert "runner_bundle_path_missing" in report["reasons"]


def test_readiness_report_has_no_api_key_fields(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "main"
    repo.mkdir()
    resources = tmp_path / "resources.yaml"
    _write_resources(
        resources,
        main_project={
            "repository_path": str(repo),
            "expected_branch": "main",
            "runner_bundle_path": None,
        },
    )
    monkeypatch.setattr(
        "external_baselines.common.main_project_readiness._git",
        lambda cmd, cwd: "main" if cmd[:2] == ["branch", "--show-current"] else "deadbeef",
    )
    report = assess_main_project_readiness(resources)
    blob = json.dumps(report)
    assert "sk-" not in blob
    assert "api_key" not in blob.lower()


def test_v1_marker_file(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "main"
    marker = repo / "artifacts" / "status" / "first_model_v1_ready.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"ready": True}), encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "input_cases.jsonl").write_text("{}\n", encoding="utf-8")
    (bundle / "prediction_schema.json").write_text("{}", encoding="utf-8")
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")

    resources = tmp_path / "resources.yaml"
    _write_resources(
        resources,
        main_project={
            "repository_path": str(repo),
            "expected_branch": "main",
            "runner_bundle_path": str(bundle),
        },
        execution={
            "allow_real_model_calls": True,
            "allow_cross_repo_test": True,
            "allow_formal_evaluation": False,
        },
        status={"main_project_v1_ready": False, "configs_frozen": False, "real_dry_run_completed": False},
    )

    monkeypatch.setattr(
        "external_baselines.common.main_project_readiness._git",
        lambda cmd, cwd: "main" if cmd[:2] == ["branch", "--show-current"] else "deadbeef",
    )
    monkeypatch.setattr(
        "external_baselines.common.main_project_readiness._validate_bundle_checksum",
        lambda root: {"ok": True, "runner_bundle_checksum": "checksum", "schema_checksum": "schema"},
    )
    report = assess_main_project_readiness(resources)
    assert report["main_project_v1_marker_present"] is True
    assert report["safe_to_run_real_dry_run"] is True
    assert report["safe_to_run_formal_experiment"] is False
