#!/usr/bin/env python3
"""Validate a DeepEval handoff without invoking an evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.interop.deepeval_handoff.validator import validate_handoff  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a DeepEval-compatible handoff bundle")
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--main-repo", required=True, type=Path)
    args = parser.parse_args(argv)
    report = validate_handoff(args.handoff.resolve(), main_repo=args.main_repo.resolve(), write_report=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
