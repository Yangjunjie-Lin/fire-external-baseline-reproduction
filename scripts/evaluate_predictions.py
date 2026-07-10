#!/usr/bin/env python3
"""LOCAL PROXY DIAGNOSTIC — NOT SHARED PAPER EVALUATOR.

Evaluate previously generated predictions with proxy diagnostics only.
Paper-facing scores must use fire-agent-demo's shared evaluator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.runner import evaluate_predictions

_BANNER = "LOCAL PROXY DIAGNOSTIC — NOT SHARED PAPER EVALUATOR"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=f"{_BANNER}. Proxy metrics only; not the shared paper evaluator."
    )
    parser.add_argument("--predictions", default="outputs/predictions.jsonl")
    parser.add_argument("--dataset", default="data/scenarios/scenario_matrix_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", default="outputs/baseline_metrics.csv")
    parser.add_argument("--report", default="outputs/baseline_report.md")
    parser.add_argument("--manifest", default="outputs/run_manifest.json")
    args = parser.parse_args(argv)
    print(_BANNER)
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
