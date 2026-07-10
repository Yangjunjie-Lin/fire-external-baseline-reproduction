#!/usr/bin/env python3
"""Run baselines against a firebench-interop-v1 Runner Bundle.

Formal entrypoint uses a single --experiment-manifest that references:
  - one shared model config
  - per-method configs
  - main-table vs supplemental method roles

Do not pass Evaluator Bundles. Do not auto-call paid APIs from CI/agents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.checksums import sha256_file
from external_baselines.common.experiment_manifest import (
    MAIN_TABLE_METHODS,
    build_method_config,
    enabled_methods,
    load_experiment_manifest,
)
from external_baselines.common.guards import assert_paper_final_allowed
from external_baselines.common.fairness import validate_cross_method_fairness
from external_baselines.common.io import write_json, write_jsonl
from external_baselines.interop.bundle import (
    assert_no_evaluator_bundle_access,
    load_runner_bundle,
    recompute_bundle_checksum,
    validate_bundle_checksum,
)
from external_baselines.interop.schema import (
    SCHEMA_PATH,
    baseline_row_to_interop,
    validate_interop_record,
)
from external_baselines.runner import generate_predictions


def _verify_bundle_hashes(bundle: dict) -> dict:
    """Record schema/scenario/corpus hashes for cross-repo verification checklist."""
    schema_path = bundle.get("prediction_schema_path")
    scenarios_path = bundle.get("scenarios_path")
    corpus_dir = bundle.get("corpus_dir")
    return {
        "schema_sha256": sha256_file(schema_path) if schema_path else None,
        "scenarios_sha256": sha256_file(scenarios_path) if scenarios_path else None,
        "corpus_aggregate_sha256": (bundle.get("corpus_manifest") or {}).get("aggregate_sha256"),
        "bundle_checksum": bundle.get("bundle_checksum"),
        "corpus_dir": corpus_dir,
        "cross_repository_interop_verified": False,
        "note": "Set cross_repository_interop_verified=true only after main-project Runner Bundle + neutral evaluator are actually run.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run firebench-interop-v1 baselines from an experiment manifest + Runner Bundle.")
    parser.add_argument(
        "--experiment-manifest",
        required=True,
        help="Single experiment manifest YAML/JSON (shared model + per-method configs).",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Runner Bundle path; overrides manifest.bundle when provided.",
    )
    parser.add_argument(
        "--include-supplemental",
        action="store_true",
        help="Also run supplemental/extended methods (dense/hybrid/ekell_enhanced). Default: main-table only.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--legacy-output", default=None)
    parser.add_argument("--manifest", default=None, help="Run manifest output path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-bundle-checksum", default=None)
    # Deprecated: multiple --config overlays are ambiguous for paper runs.
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.config:
        raise SystemExit(
            "Multiple --config overlays are no longer supported for formal interop runs. "
            "Use a single --experiment-manifest that references shared_model_config and per-method configs."
        )

    experiment = load_experiment_manifest(args.experiment_manifest)
    bundle_path = args.bundle or experiment.get("bundle")
    if not bundle_path:
        raise SystemExit("Runner Bundle path required via --bundle or experiment manifest.bundle")

    assert_no_evaluator_bundle_access(bundle_path)
    bundle = load_runner_bundle(bundle_path)
    expected_checksum = (
        args.expected_bundle_checksum or experiment.get("expected_bundle_checksum")
    )
    checksum_report = validate_bundle_checksum(bundle, expected=expected_checksum)
    if not checksum_report["ok"]:
        raise SystemExit(f"Bundle checksum mismatch: {checksum_report}")
    if not bundle.get("scenarios_path"):
        raise SystemExit("Runner Bundle does not contain a scenarios file.")

    method_entries = enabled_methods(experiment, include_supplemental=args.include_supplemental)
    if not method_entries:
        raise SystemExit("No enabled methods in experiment manifest (main-table empty?).")

    method_configs: dict[str, dict] = {}
    methods: list[str] = []
    for entry in method_entries:
        mid = entry["method_id"]
        cfg = build_method_config(experiment, entry)
        if bundle.get("corpus_dir"):
            cfg.setdefault("paths", {})["corpus_dir"] = bundle["corpus_dir"]
        cfg["bundle_checksum"] = recompute_bundle_checksum(bundle["bundle_root"])
        assert_paper_final_allowed(cfg)
        method_configs[mid] = cfg
        methods.append(mid)
    fairness_report = validate_cross_method_fairness(method_configs)

    output = args.output or experiment.get("output")
    legacy_output = args.legacy_output or experiment.get("legacy_output")
    run_manifest = args.manifest or experiment.get("run_manifest")
    limit = args.limit if args.limit is not None else experiment.get("limit")

    # Use first method config as default/shared snapshot for run manifest metadata.
    shared_snapshot = next(iter(method_configs.values()))
    legacy_rows = generate_predictions(
        methods=methods,
        dataset=bundle["scenarios_path"],
        limit=limit,
        output_path=legacy_output,
        manifest_path=run_manifest,
        config=shared_snapshot,
        method_configs=method_configs,
    )

    interop_rows = [
        baseline_row_to_interop(row, bundle_checksum=bundle.get("bundle_checksum"))
        for row in legacy_rows
    ]
    errors = []
    bundle_schema = bundle.get("prediction_schema") or bundle.get("schemas", {}).get("prediction")
    schema_path = bundle.get("prediction_schema_path")
    expected_schema_sha = bundle.get("prediction_schema_sha256") or experiment.get("expected_prediction_schema_sha256")
    for i, row in enumerate(interop_rows):
        errs = validate_interop_record(
            row,
            schema=bundle_schema if isinstance(bundle_schema, dict) else None,
            schema_path=schema_path or SCHEMA_PATH,
            expected_schema_sha256=expected_schema_sha,
            require_external_schema=bool(experiment.get("paper_final")),
        )
        if errs:
            errors.append({"index": i, "case_id": row.get("case_id"), "errors": errs})
    if errors:
        write_json(Path(output).with_suffix(".validation_errors.json"), errors)
        raise SystemExit(f"Interop schema validation failed for {len(errors)} records.")

    write_jsonl(output, interop_rows)
    hash_report = _verify_bundle_hashes(bundle)
    write_json(
        Path(run_manifest).with_name("interop_bundle_report.json"),
        {
            "experiment_id": experiment.get("experiment_id"),
            "experiment_manifest": experiment.get("manifest_path"),
            "freeze_status": experiment.get("freeze_status"),
            "main_table_methods": list(MAIN_TABLE_METHODS),
            "methods_run": methods,
            "include_supplemental": bool(args.include_supplemental),
            "bundle": {
                "root": bundle.get("bundle_root"),
                "checksum": bundle.get("bundle_checksum"),
                "scenarios_path": bundle.get("scenarios_path"),
                "corpus_dir": bundle.get("corpus_dir"),
            },
            "checksum_validation": checksum_report,
            "cross_method_fairness": fairness_report,
            "hash_verification": hash_report,
            "n_predictions": len(interop_rows),
            "output": output,
            "cross_repository_interop_verified": False,
        },
    )
    print(f"Wrote {len(interop_rows)} firebench-interop-v1 predictions to {output}")
    print(f"Methods: {methods}")
    print(f"Bundle checksum: {bundle.get('bundle_checksum')}")
    print("cross_repository_interop_verified=false (await formal main-project Runner Bundle run)")


if __name__ == "__main__":
    main()
