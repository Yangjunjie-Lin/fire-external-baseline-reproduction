#!/usr/bin/env python3
"""Build a markdown report from predictions + optional metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.io import load_expected_by_id, read_json, read_jsonl
from external_baselines.evaluation.metrics import aggregate_metrics, score_output
from external_baselines.evaluation.report import build_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build baseline report from predictions.")
    parser.add_argument("--predictions", default="outputs/predictions.jsonl")
    parser.add_argument("--dataset", default=None, help="Optional; enables proxy scoring if gold present.")
    parser.add_argument("--manifest", default="outputs/run_manifest.json")
    parser.add_argument("--report", default="outputs/baseline_report.md")
    args = parser.parse_args(argv)

    outputs = read_jsonl(args.predictions)
    manifest = read_json(args.manifest, default={}) if Path(args.manifest).exists() else {}
    aggregated = {}
    if args.dataset:
        expected = load_expected_by_id(args.dataset)
        scored = [score_output(out, expected.get(str(out.get("scenario_id")), {})) for out in outputs]
        aggregated = aggregate_metrics(scored)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(build_report(outputs, aggregated, manifest=manifest), encoding="utf-8")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
