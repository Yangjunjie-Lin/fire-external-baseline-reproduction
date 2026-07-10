#!/usr/bin/env python3
"""Display local experiment stage and gate status (read-only)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.io import read_yaml  # noqa: E402
from external_baselines.common.main_project_readiness import assess_main_project_readiness  # noqa: E402

ENV_VARS = (
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "SILICONFLOW_MODEL",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
)


def _env_presence() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ENV_VARS:
        out[name] = "present" if os.getenv(name) else "missing"
    return out


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
        "api_env_presence": _env_presence(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
