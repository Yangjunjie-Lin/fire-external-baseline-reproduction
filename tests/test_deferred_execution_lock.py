"""Deferred execution lock tests (dry_run vs formal)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from external_baselines.common.execution_lock import (
    LOCK_MESSAGE,
    ExecutionLockError,
    assert_execution_allowed,
    assert_formal_execution_allowed,
)

ROOT = Path(__file__).resolve().parents[1]


def _resources(
    tmp_path: Path,
    *,
    execution: dict | None = None,
    status: dict | None = None,
    ready: bool = False,
) -> Path:
    path = tmp_path / "experiment_resources.yaml"
    repo = tmp_path / "main"
    repo.mkdir(exist_ok=True)
    bundle = tmp_path / "bundle"
    if ready:
        bundle.mkdir(exist_ok=True)
        (bundle / "manifest.json").write_text("{}", encoding="utf-8")
        (bundle / "input_cases.jsonl").write_text("{}\n", encoding="utf-8")
        (bundle / "prediction_schema.json").write_text("{}", encoding="utf-8")
        marker = repo / "artifacts" / "status" / "first_model_v1_ready.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{"ready": true}', encoding="utf-8")

    payload = {
        "main_project": {
            "repository_path": str(repo),
            "expected_branch": "main",
            "runner_bundle_path": str(bundle) if ready else None,
        },
        "execution": {
            "allow_real_model_calls": False,
            "allow_cross_repo_test": False,
            "allow_formal_evaluation": False,
            **(execution or {}),
        },
        "status": {
            "main_project_v1_ready": False,
            "configs_frozen": False,
            "real_dry_run_completed": False,
            **(status or {}),
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _patch_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        "external_baselines.common.main_project_readiness._git",
        lambda cmd, cwd: "main" if cmd[:2] == ["branch", "--show-current"] else "deadbeef",
    )
    monkeypatch.setattr(
        "external_baselines.common.main_project_readiness._validate_bundle_checksum",
        lambda root: {"ok": True, "runner_bundle_checksum": "c", "schema_checksum": "s"},
    )


def test_lock_blocks_placeholder_bundle(tmp_path: Path) -> None:
    resources = _resources(tmp_path)
    with pytest.raises(ExecutionLockError) as exc:
        assert_formal_execution_allowed(
            experiment_manifest={"bundle": "REQUIRED_AFTER_MAIN_PROJECT_V1", "paper_final": True},
            resources_path=resources,
        )
    assert "Formal execution is currently locked" in str(exc.value)


def test_override_allows_bypass(tmp_path: Path) -> None:
    resources = _resources(tmp_path)
    audit = assert_formal_execution_allowed(
        experiment_manifest={"bundle": "REQUIRED_AFTER_MAIN_PROJECT_V1", "paper_final": True},
        resources_path=resources,
        override_readiness_lock=True,
    )
    assert audit["execution_lock_overridden"] is True
    assert audit["paper_valid"] is False


def test_override_is_recorded(tmp_path: Path) -> None:
    resources = _resources(tmp_path)
    audit = assert_execution_allowed(
        experiment_manifest={"bundle": "REQUIRED_AFTER_MAIN_PROJECT_V1"},
        resources_path=resources,
        execution_stage="dry_run",
        limit=3,
        output_path="outputs/dry_run/x/predictions.jsonl",
        override_readiness_lock=True,
    )
    assert audit["execution_lock_overridden"] is True


def test_run_interop_blocks_without_override() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_interop_baselines.py",
            "--experiment-manifest",
            "configs/experiments/controlled_main_table_v1.yaml.example",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Formal execution is currently locked" in proc.stderr + proc.stdout


def test_default_execution_stage_is_formal() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-stage", choices=["dry_run", "formal"], default="formal")
    args = parser.parse_args([])
    assert args.execution_stage == "formal"


def test_lock_message_constant() -> None:
    assert "main project v1 Runner Bundle" in LOCK_MESSAGE


def test_dry_run_does_not_require_formal_evaluation(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={
            "allow_real_model_calls": True,
            "allow_cross_repo_test": True,
            "allow_formal_evaluation": False,
        },
        status={"main_project_v1_ready": True},
    )
    audit = assert_execution_allowed(
        experiment_manifest={"bundle": str(tmp_path / "bundle")},
        resources_path=resources,
        execution_stage="dry_run",
        limit=3,
        output_path="outputs/dry_run/controlled_v1/predictions.jsonl",
    )
    assert audit["execution_stage"] == "dry_run"
    assert audit["execution_lock_overridden"] is False


def test_dry_run_requires_limit(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={"allow_real_model_calls": True, "allow_cross_repo_test": True},
        status={"main_project_v1_ready": True},
    )
    with pytest.raises(ExecutionLockError) as exc:
        assert_execution_allowed(
            experiment_manifest={"bundle": str(tmp_path / "bundle")},
            resources_path=resources,
            execution_stage="dry_run",
            limit=None,
            output_path="outputs/dry_run/x/predictions.jsonl",
        )
    assert "dry_run_limit_required" in str(exc.value)


def test_dry_run_rejects_limit_above_maximum(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={"allow_real_model_calls": True, "allow_cross_repo_test": True},
        status={"main_project_v1_ready": True},
    )
    with pytest.raises(ExecutionLockError) as exc:
        assert_execution_allowed(
            experiment_manifest={"bundle": str(tmp_path / "bundle")},
            resources_path=resources,
            execution_stage="dry_run",
            limit=50,
            output_path="outputs/dry_run/x/predictions.jsonl",
        )
    assert "dry_run_limit_out_of_range" in str(exc.value)


def test_dry_run_requires_dry_run_output_path(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={"allow_real_model_calls": True, "allow_cross_repo_test": True},
        status={"main_project_v1_ready": True},
    )
    with pytest.raises(ExecutionLockError) as exc:
        assert_execution_allowed(
            experiment_manifest={"bundle": str(tmp_path / "bundle")},
            resources_path=resources,
            execution_stage="dry_run",
            limit=3,
            output_path="outputs/interop/predictions.jsonl",
        )
    assert "dry_run_output" in str(exc.value)


def test_formal_requires_allow_formal_evaluation(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={
            "allow_real_model_calls": True,
            "allow_cross_repo_test": True,
            "allow_formal_evaluation": False,
        },
        status={
            "main_project_v1_ready": True,
            "configs_frozen": True,
            "real_dry_run_completed": True,
        },
    )
    with pytest.raises(ExecutionLockError) as exc:
        assert_execution_allowed(
            experiment_manifest={"bundle": str(tmp_path / "bundle"), "paper_final": True},
            resources_path=resources,
            execution_stage="formal",
            output_path="outputs/interop/predictions.jsonl",
        )
    assert "allow_formal_evaluation_false" in str(exc.value)


def test_formal_requires_configs_frozen(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={
            "allow_real_model_calls": True,
            "allow_cross_repo_test": True,
            "allow_formal_evaluation": True,
        },
        status={
            "main_project_v1_ready": True,
            "configs_frozen": False,
            "real_dry_run_completed": True,
        },
    )
    with pytest.raises(ExecutionLockError) as exc:
        assert_execution_allowed(
            experiment_manifest={"bundle": str(tmp_path / "bundle")},
            resources_path=resources,
            execution_stage="formal",
            output_path="outputs/interop/predictions.jsonl",
        )
    assert "configs_not_frozen" in str(exc.value)


def test_formal_requires_completed_real_dry_run(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={
            "allow_real_model_calls": True,
            "allow_cross_repo_test": True,
            "allow_formal_evaluation": True,
        },
        status={
            "main_project_v1_ready": True,
            "configs_frozen": True,
            "real_dry_run_completed": False,
        },
    )
    with pytest.raises(ExecutionLockError) as exc:
        assert_execution_allowed(
            experiment_manifest={"bundle": str(tmp_path / "bundle")},
            resources_path=resources,
            execution_stage="formal",
            output_path="outputs/interop/predictions.jsonl",
        )
    assert "real_dry_run_not_completed" in str(exc.value)


def test_formal_rejects_limit(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={
            "allow_real_model_calls": True,
            "allow_cross_repo_test": True,
            "allow_formal_evaluation": True,
        },
        status={
            "main_project_v1_ready": True,
            "configs_frozen": True,
            "real_dry_run_completed": True,
        },
    )
    with pytest.raises(ExecutionLockError) as exc:
        assert_execution_allowed(
            experiment_manifest={"bundle": str(tmp_path / "bundle")},
            resources_path=resources,
            execution_stage="formal",
            limit=3,
            output_path="outputs/interop/predictions.jsonl",
        )
    assert "formal_limit_forbidden" in str(exc.value)


def test_formal_rejects_allow_partial(tmp_path: Path, monkeypatch) -> None:
    _patch_ready(monkeypatch)
    resources = _resources(
        tmp_path,
        ready=True,
        execution={
            "allow_real_model_calls": True,
            "allow_cross_repo_test": True,
            "allow_formal_evaluation": True,
        },
        status={
            "main_project_v1_ready": True,
            "configs_frozen": True,
            "real_dry_run_completed": True,
        },
    )
    with pytest.raises(ExecutionLockError) as exc:
        assert_execution_allowed(
            experiment_manifest={"bundle": str(tmp_path / "bundle")},
            resources_path=resources,
            execution_stage="formal",
            allow_partial=True,
            output_path="outputs/interop/predictions.jsonl",
        )
    assert "formal_allow_partial_forbidden" in str(exc.value)
