#!/usr/bin/env python3
"""Validate formal experiment / method configs (paper-facing guards).

Modes:
  Template validation (structure check on .example files):
    python scripts/validate_formal_config.py --config path/to/manifest.yaml.example --allow-placeholders

  Formal validation (real run prep; rejects .example and placeholders):
    python scripts/validate_formal_config.py --config path/to/manifest.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.formal_config_validator import (  # noqa: E402
    FormalConfigError,
    _is_example_path,
    validate_experiment_manifest,
    validate_method_config,
)
from external_baselines.common.io import read_yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate formal configs. Default = formal run mode (no .example, no placeholders). "
            "Use --allow-placeholders for template/.example structure checks only."
        )
    )
    parser.add_argument("--config", required=True, help="Experiment manifest or method config YAML")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Template validation mode: allow .example paths and placeholder values",
    )
    parser.add_argument("--method-config", action="store_true", help="Treat --config as a method config")
    args = parser.parse_args(argv)

    path = Path(args.config)
    if not args.allow_placeholders and _is_example_path(str(path)):
        print(
            json.dumps(
                {
                    "valid": False,
                    "error": (
                        "Formal validation rejects .example config paths. "
                        "Copy the template to a non-.example file first, or use --allow-placeholders."
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
                allow_placeholders=args.allow_placeholders,
                require_formal=True,
            )
            result = {
                "path": str(path),
                "type": "method_config",
                "valid": True,
                "mode": "template" if args.allow_placeholders else "formal",
            }
        else:
            result = validate_experiment_manifest(path, allow_placeholders=args.allow_placeholders)
        print(json.dumps(result, indent=2))
    except FormalConfigError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
