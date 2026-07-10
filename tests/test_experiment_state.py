"""Tests for experiment state display script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_show_experiment_state_defaults_block_api_calls() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/show_experiment_state.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    assert data["allowed_actions"]["api_call"] is False
    assert data["allowed_actions"]["formal_experiment"] is False
    assert data["gates"]["real_llm_config_ready"] is True


def test_show_experiment_state_env_presence_labels_only() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/show_experiment_state.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    presence = data["api_env_presence"]
    for value in presence.values():
        assert value in {"present", "missing"}
