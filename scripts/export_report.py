#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.evaluation.report import export_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export markdown report from baseline output JSONL.")
    parser.add_argument("--input", default="outputs/baseline_outputs.jsonl")
    parser.add_argument("--output", default="outputs/baseline_report.md")
    args = parser.parse_args()
    export_report(args.input, args.output)
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
