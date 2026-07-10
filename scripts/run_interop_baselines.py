#!/usr/bin/env python3
"""Run baselines against a firebench-interop-v1 Runner Bundle.

Baselines may only read the Runner Bundle. Do not pass Evaluator Bundles.
Does not call paid APIs unless the user supplies a real LLM config and runs manually.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.guards import assert_paper_final_allowed
from external_baselines.common.io import deep_merge, load_config, write_json, write_jsonl
from external_baselines.interop.bundle import assert_no_evaluator_bundle_access, load_runner_bundle, validate_bundle_checksum
from external_baselines.interop.schema import baseline_row_to_interop, validate_interop_record
from external_baselines.runner import generate_predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run firebench-interop-v1 baseline predictions from a Runner Bundle.")
    parser.add_argument("--bundle", required=True, help="Path to Runner Bundle directory or manifest.json")
    parser.add_argument(
        "--methods",
        default="direct_llm,bm25_rag,dense_rag,hybrid_rag,ekell_style_faithful",
    )
    parser.add_argument("--config", action="append", default=[], help="Additional YAML configs (e.g. shared model).")
    parser.add_argument("--output", default="outputs/firebench_interop_v1_predictions.jsonl")
    parser.add_argument("--legacy-output", default="outputs/baseline_outputs_legacy.jsonl")
    parser.add_argument("--manifest", default="outputs/interop_run_manifest.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-bundle-checksum", default=None)
    args = parser.parse_args(argv)

    assert_no_evaluator_bundle_access(args.bundle)
    bundle = load_runner_bundle(args.bundle)
    checksum_report = validate_bundle_checksum(bundle, expected=args.expected_bundle_checksum)
    if args.expected_bundle_checksum and not checksum_report["ok"]:
        raise SystemExit(f"Bundle checksum mismatch: {checksum_report}")

    if not bundle.get("scenarios_path"):
        raise SystemExit("Runner Bundle does not contain a scenarios file.")

    base = load_config("configs/default.yaml", *(args.config or []))
    # Bundle experiment config overlays local config; paths from bundle win when present.
    config = deep_merge(base, bundle.get("experiment_config") or {})
    if bundle.get("corpus_dir"):
        config.setdefault("paths", {})["corpus_dir"] = bundle["corpus_dir"]
    config["bundle_checksum"] = bundle.get("bundle_checksum")
    assert_paper_final_allowed(config)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    legacy_rows = generate_predictions(
        methods=methods,
        dataset=bundle["scenarios_path"],
        limit=args.limit,
        output_path=args.legacy_output,
        manifest_path=args.manifest,
        config=config,
    )

    interop_rows = [
        baseline_row_to_interop(row, bundle_checksum=bundle.get("bundle_checksum"))
        for row in legacy_rows
    ]
    errors = []
    for i, row in enumerate(interop_rows):
        errs = validate_interop_record(row)
        if errs:
            errors.append({"index": i, "case_id": row.get("case_id"), "errors": errs})
    if errors:
        write_json(Path(args.output).with_suffix(".validation_errors.json"), errors)
        raise SystemExit(f"Interop schema validation failed for {len(errors)} records. See validation_errors.json")

    write_jsonl(args.output, interop_rows)
    write_json(
        Path(args.manifest).with_name("interop_bundle_report.json"),
        {
            "bundle": {
                "root": bundle.get("bundle_root"),
                "checksum": bundle.get("bundle_checksum"),
                "scenarios_path": bundle.get("scenarios_path"),
                "corpus_dir": bundle.get("corpus_dir"),
                "corpus_manifest": bundle.get("corpus_manifest"),
            },
            "checksum_validation": checksum_report,
            "n_predictions": len(interop_rows),
            "methods": methods,
            "output": args.output,
        },
    )
    print(f"Wrote {len(interop_rows)} firebench-interop-v1 predictions to {args.output}")
    print(f"Bundle checksum: {bundle.get('bundle_checksum')}")


if __name__ == "__main__":
    main()
