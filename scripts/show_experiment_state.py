#!/usr/bin/env python3
"""Display local experiment stage and gate status (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.environment import (  # noqa: E402
    DEFAULT_PRESENCE_VARS,
    environment_variable_presence,
    load_local_environment,
)
from external_baselines.common.io import read_yaml  # noqa: E402
from external_baselines.common.main_project_readiness import assess_main_project_readiness  # noqa: E402


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = read_yaml(path)
    return dict(data) if isinstance(data, dict) else {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show experiment preparation state (no mutations).")
    parser.add_argument("--state", default="configs/local/experiment_state.yaml")
    parser.add_argument("--resources", default="configs/local/experiment_resources.yaml")
    args = parser.parse_args(argv)

    env_meta = load_local_environment()
    state = _load_yaml(ROOT / args.state)
    resources = _load_yaml(ROOT / args.resources)
    readiness = assess_main_project_readiness(ROOT / args.resources) if (ROOT / args.resources).is_file() else {}

    report = {
        "stage": state.get("stage"),
        "gates": state.get("gates"),
        "allowed_actions": state.get("allowed_actions"),
        "resource_status": resources.get("resource_status"),
        "execution_flags": resources.get("execution"),
        "main_project_readiness": readiness,
        "api_env_presence": environment_variable_presence(DEFAULT_PRESENCE_VARS),
        "env_sources": {
            "loaded_sources": env_meta.get("loaded_sources"),
            "discovered_sources": env_meta.get("discovered_sources"),
            "override": False,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
