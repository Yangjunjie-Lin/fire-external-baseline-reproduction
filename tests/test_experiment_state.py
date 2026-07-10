"""Environment loading and experiment-state presence tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from external_baselines.common.environment import (
    environment_variable_presence,
    load_local_environment,
)

ROOT = Path(__file__).resolve().parents[1]


def test_environment_loader_preserves_existing_process_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "process-key")
    env_file = tmp_path / ".env"
    env_file.write_text("SILICONFLOW_API_KEY=file-key\n", encoding="utf-8")
    load_local_environment(repo_root=tmp_path)
    assert os.getenv("SILICONFLOW_API_KEY") == "process-key"


def test_environment_loader_reads_baseline_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("FIRE_AGENT_DEMO_ENV_PATH", raising=False)
    (tmp_path / ".env").write_text("SILICONFLOW_API_KEY=baseline-key\n", encoding="utf-8")
    meta = load_local_environment(repo_root=tmp_path)
    assert "baseline_repo_env" in meta["loaded_sources"]
    assert os.getenv("SILICONFLOW_API_KEY") == "baseline-key"
    # Clean for other tests
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)


def test_environment_loader_reads_explicit_main_env_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SILICONFLOW_BASE_URL", raising=False)
    main_env = tmp_path / "main.env"
    main_env.write_text("SILICONFLOW_BASE_URL=https://example.test/v1\n", encoding="utf-8")
    monkeypatch.setenv("FIRE_AGENT_DEMO_ENV_PATH", str(main_env))
    meta = load_local_environment(repo_root=tmp_path)
    assert "explicit_main_env_path" in meta["loaded_sources"]
    assert os.getenv("SILICONFLOW_BASE_URL") == "https://example.test/v1"
    monkeypatch.delenv("SILICONFLOW_BASE_URL", raising=False)
    monkeypatch.delenv("FIRE_AGENT_DEMO_ENV_PATH", raising=False)


def test_environment_loader_reads_sibling_main_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("FIRE_AGENT_DEMO_ENV_PATH", raising=False)
    workspace = tmp_path / "workspace"
    repo = workspace / "baseline"
    demo = workspace / "fire-agent-demo"
    repo.mkdir(parents=True)
    demo.mkdir(parents=True)
    (demo / ".env").write_text("LLM_API_KEY=sibling-key\n", encoding="utf-8")
    meta = load_local_environment(repo_root=repo)
    assert "sibling_main_project_env" in meta["loaded_sources"]
    assert os.getenv("LLM_API_KEY") == "sibling-key"
    monkeypatch.delenv("LLM_API_KEY", raising=False)


def test_env_presence_only_returns_present_or_missing(monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "secret-should-not-leak")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    presence = environment_variable_presence(["SILICONFLOW_API_KEY", "OPENAI_API_KEY"])
    assert presence == {"SILICONFLOW_API_KEY": "present", "OPENAI_API_KEY": "missing"}
    blob = json.dumps(presence)
    assert "secret-should-not-leak" not in blob


def test_environment_report_does_not_contain_secret_value(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "super-secret-value-xyz")
    meta = load_local_environment(repo_root=tmp_path)
    blob = json.dumps(meta)
    assert "super-secret-value-xyz" not in blob
    presence = environment_variable_presence(["SILICONFLOW_API_KEY"])
    assert "super-secret-value-xyz" not in json.dumps(presence)


def test_show_experiment_state_matches_llm_env_loading(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("FIRE_AGENT_DEMO_ENV_PATH", raising=False)
    env_file = tmp_path / "demo.env"
    env_file.write_text("SILICONFLOW_API_KEY=from-demo-env\n", encoding="utf-8")
    monkeypatch.setenv("FIRE_AGENT_DEMO_ENV_PATH", str(env_file))

    # Ensure script uses same loader; invoke via import path by setting env before run.
    state = tmp_path / "state.yaml"
    state.write_text(
        "stage: configuration_prepared_waiting_for_main_v1\n"
        "gates: {real_llm_config_ready: true}\n"
        "allowed_actions: {api_call: false, formal_experiment: false}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/show_experiment_state.py",
            "--state",
            str(state),
            "--resources",
            str(tmp_path / "missing_resources.yaml"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "FIRE_AGENT_DEMO_ENV_PATH": str(env_file)},
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["api_env_presence"]["SILICONFLOW_API_KEY"] == "present"
    assert "from-demo-env" not in proc.stdout
    assert data["allowed_actions"]["api_call"] is False
