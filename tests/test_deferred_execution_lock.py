"""Deferred execution lock tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from external_baselines.common.execution_lock import (
    LOCK_MESSAGE,
    ExecutionLockError,
    assert_formal_execution_allowed,
)

ROOT = Path(__file__).resolve().parents[1]


def _resources(tmp_path: Path, **execution) -> Path:
    path = tmp_path / "experiment_resources.yaml"
    payload = {
        "main_project": {"runner_bundle_path": None},
        "execution": {
            "allow_real_model_calls": False,
            "allow_cross_repo_test": False,
            "allow_formal_evaluation": False,
            **execution,
        },
        "status": {"main_project_v1_ready": False},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


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


def test_lock_message_constant() -> None:
    assert "main project v1 Runner Bundle" in LOCK_MESSAGE
