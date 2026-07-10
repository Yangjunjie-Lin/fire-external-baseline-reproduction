#!/usr/bin/env python3
"""Validate formal experiment / method configs (stage-aware).

Stages:
  template  — .example + placeholders allowed; freeze_status must be provisional
  dry_run   — no .example/placeholders; freeze_status provisional|frozen
  formal    — freeze_status=frozen + freeze_manifest required
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.formal_config_validator import (  # noqa: E402
    FormalConfigError,
    _is_example_path,
    validate_experiment_manifest,
    validate_method_config,
)
from external_baselines.common.io import read_yaml  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate formal configs with stage-aware freeze rules.")
    parser.add_argument("--config", required=True, help="Experiment manifest or method config YAML")
    parser.add_argument(
        "--validation-stage",
        choices=["template", "dry_run", "formal"],
        default=None,
        help="Validation stage (default: formal, or template when --allow-placeholders).",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Deprecated alias for --validation-stage template.",
    )
    parser.add_argument("--method-config", action="store_true", help="Treat --config as a method config")
    args = parser.parse_args(argv)

    stage = args.validation_stage
    if args.allow_placeholders:
        warnings.warn(
            "--allow-placeholders is deprecated; use --validation-stage template",
            DeprecationWarning,
            stacklevel=1,
        )
        stage = stage or "template"
    stage = stage or "formal"

    path = Path(args.config)
    if stage != "template" and _is_example_path(str(path)):
        print(
            json.dumps(
                {
                    "valid": False,
                    "validation_stage": stage,
                    "error": (
                        f"{stage} validation rejects .example config paths. "
                        "Copy the template to a non-.example file first, or use --validation-stage template."
                    ),
                },
                indent=2,
            )
        )
        raise SystemExit(1)

    try:
        if args.method_config:
            cfg = read_yaml(path)
            validate_method_config(
                cfg,
                method_id=str(cfg.get("method_id") or ""),
                allow_placeholders=(stage == "template"),
                require_formal=True,
            )
            result = {
                "path": str(path),
                "type": "method_config",
                "valid": True,
                "validation_stage": stage,
                "mode": stage,
            }
        else:
            result = validate_experiment_manifest(
                path,
                allow_placeholders=(stage == "template"),
                validation_stage=stage,
            )
        print(json.dumps(result, indent=2))
    except FormalConfigError as exc:
        print(json.dumps({"valid": False, "validation_stage": stage, "error": str(exc)}, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
