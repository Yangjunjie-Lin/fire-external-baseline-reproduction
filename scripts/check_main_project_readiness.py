#!/usr/bin/env python3
"""Check main-project v1 readiness without modifying fire-agent-demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.main_project_readiness import assess_main_project_readiness  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assess main-project v1 readiness for cross-repo runs.")
    parser.add_argument(
        "--resources",
        default="configs/local/experiment_resources.yaml",
        help="Local experiment resources file (gitignored).",
    )
    parser.add_argument("--require-v1-marker", action="store_true")
    args = parser.parse_args(argv)

    report = assess_main_project_readiness(args.resources, require_v1_marker=args.require_v1_marker)
    print(json.dumps(report, indent=2))
    # Not ready is expected during preparation; exit 0 unless file missing.
    if not Path(args.resources).is_file():
        raise SystemExit(f"Resources file not found: {args.resources}")


if __name__ == "__main__":
    main()
