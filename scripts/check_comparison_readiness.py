#!/usr/bin/env python3
"""Per-method comparison-suite readiness diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.comparison_readiness import assess_comparison_readiness  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check comparison-suite resource readiness.")
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--resources", default="configs/local/experiment_resources.yaml")
    parser.add_argument("--bundle", default=None, help="Runner Bundle path (overrides manifest/resources).")
    parser.add_argument("--method-set", choices=["main_table", "comparison_suite"], default="comparison_suite")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit 0 after printing the report (for status display).",
    )
    args = parser.parse_args(argv)
    report = assess_comparison_readiness(
        experiment_manifest=args.experiment_manifest,
        resources_path=args.resources,
        method_set=args.method_set,
        bundle_path=args.bundle,
    )
    print(json.dumps(report, indent=2))
    if args.report_only:
        return
    if not report.get("comparison_ready"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
