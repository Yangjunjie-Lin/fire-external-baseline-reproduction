#!/usr/bin/env python3
"""Evaluate previously generated predictions with proxy diagnostics only.

Paper-facing scores should use the main project's neutral shared evaluator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.runner import evaluate_predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline predictions (proxy metrics only).")
    parser.add_argument("--predictions", default="outputs/predictions.jsonl")
    parser.add_argument("--dataset", default="data/scenarios/scenario_matrix_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", default="outputs/baseline_metrics.csv")
    parser.add_argument("--report", default="outputs/baseline_report.md")
    parser.add_argument("--manifest", default="outputs/run_manifest.json")
    args = parser.parse_args(argv)
    result = evaluate_predictions(
        predictions_path=args.predictions,
        dataset=args.dataset,
        metrics_path=args.metrics,
        report_path=args.report,
        manifest_path=args.manifest,
        limit=args.limit,
    )
    print(f"Evaluated {result['n']} predictions (proxy diagnostics only).")
    print(f"Wrote metrics to {args.metrics}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
