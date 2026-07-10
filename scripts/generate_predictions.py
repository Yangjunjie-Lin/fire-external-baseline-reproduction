#!/usr/bin/env python3
"""Generate baseline predictions with gold isolation (no evaluation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.runner import generate_predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate external baseline predictions (gold-isolated).")
    parser.add_argument(
        "--methods",
        default="direct_llm,bm25_rag,ekell_style_controlled_shared_llm",
    )
    parser.add_argument("--method", default=None)
    parser.add_argument("--dataset", default="data/scenarios/scenario_matrix_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--manifest", default="outputs/run_manifest.json")
    args = parser.parse_args(argv)
    methods = [args.method] if args.method else [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = generate_predictions(
        methods=methods,
        dataset=args.dataset,
        config_paths=args.config,
        limit=args.limit,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
