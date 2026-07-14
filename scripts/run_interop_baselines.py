#!/usr/bin/env python3
"""Run baselines against a firebench-interop-v1 Runner Bundle.

Formal entrypoint uses a single --experiment-manifest that references:
  - one shared model config
  - per-method configs
  - main-table vs supplemental method roles

Do not pass Evaluator Bundles. Do not auto-call paid APIs from CI/agents.

Order: CLI → stage-aware validation → execution lock → bundle → fairness →
prepare runtime (via generate_predictions) → LLM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.checksums import sha256_file  # noqa: E402
from external_baselines.common.experiment_manifest import (  # noqa: E402
    MAIN_TABLE_METHODS,
    build_method_config,
    enabled_methods,
    load_experiment_manifest,
)
from external_baselines.common.fairness import validate_cross_method_fairness  # noqa: E402
from external_baselines.common.formal_config_validator import (  # noqa: E402
    FormalConfigError,
    validate_experiment_manifest,
)
from external_baselines.common.guards import assert_paper_final_allowed  # noqa: E402
from external_baselines.common.io import load_scenarios, read_json, write_json, write_jsonl  # noqa: E402
from external_baselines.common.llm_client import llm_config_summary  # noqa: E402
from external_baselines.common.strict_config_types import require_exact_bool  # noqa: E402
from external_baselines.interop.bundle import (  # noqa: E402
    assert_no_evaluator_bundle_access,
    load_runner_bundle,
    validate_bundle_checksum,
)
from external_baselines.interop.schema import (  # noqa: E402
    SCHEMA_PATH,
    baseline_row_to_interop,
    validate_interop_record,
)
from external_baselines.retrieval.embedding_backends import resolve_dimension  # noqa: E402
from external_baselines.runner import generate_predictions  # noqa: E402


def _verify_bundle_hashes(bundle: dict) -> dict:
    schema_path = bundle.get("prediction_schema_path")
    scenarios_path = bundle.get("scenarios_path")
    corpus_dir = bundle.get("corpus_dir")
    return {
        "schema_sha256": sha256_file(schema_path) if schema_path else None,
        "prediction_schema_path": schema_path,
        "prediction_schema_source": bundle.get("prediction_schema_source"),
        "prediction_schema_inside_bundle": bundle.get("prediction_schema_inside_bundle"),
        "prediction_schema_declared_sha256": bundle.get("prediction_schema_declared_sha256"),
        "prediction_schema_checksum_match": bundle.get("prediction_schema_checksum_match"),
        "prediction_schema_authoritative": bundle.get("prediction_schema_authoritative"),
        "prediction_schema_formal_eligible": bundle.get("prediction_schema_formal_eligible"),
        "scenarios_sha256": sha256_file(scenarios_path) if scenarios_path else None,
        "corpus_aggregate_sha256": (bundle.get("corpus_manifest") or {}).get("aggregate_sha256")
        if isinstance(bundle.get("corpus_manifest"), dict)
        else None,
        "producer_declared_checksum": bundle.get("producer_declared_checksum"),
        "consumer_computed_bundle_hash": bundle.get("consumer_computed_bundle_hash"),
        "file_checksum_report": bundle.get("file_checksum_report"),
        "corpus_dir": corpus_dir,
        "cross_repository_interop_verified": False,
        "note": (
            "Set cross_repository_interop_verified=true only after main-project "
            "Runner Bundle + neutral evaluator are actually run."
        ),
    }


def _shared_embedding_snapshot(method_configs: dict[str, dict]) -> dict[str, Any]:
    for mid in ("dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"):
        cfg = method_configs.get(mid) or {}
        if mid == "dense_rag":
            block = cfg.get("dense_rag") or {}
        elif mid == "hybrid_rag":
            block = dict(cfg.get("dense_rag") or {})
            hybrid = cfg.get("hybrid_rag") or {}
            block.setdefault("backend", hybrid.get("dense_method"))
            block.setdefault("model_name", hybrid.get("dense_model_name"))
            block.setdefault("model_version", hybrid.get("dense_model_version"))
            block.setdefault("dimension", hybrid.get("dimension"))
            block.setdefault("normalize_embeddings", hybrid.get("normalize_embeddings"))
        else:
            block = cfg.get("ekell_vector") or {}
        if block:
            normalize = (
                require_exact_bool(block.get("normalize_embeddings"), field=f"{mid}.normalize_embeddings")
                if block.get("normalize_embeddings") is not None
                else None
            )
            return {
                "backend": block.get("backend"),
                "model_name": block.get("model_name"),
                "model_version": block.get("model_version"),
                "dimension": resolve_dimension(block),
                "normalize_embeddings": normalize,
            }
    return {
        "backend": None,
        "model_name": None,
        "model_version": None,
        "dimension": 0,
        "normalize_embeddings": None,
    }


def _index_checksum_snapshot(
    method_configs: dict[str, dict],
    *,
    runtime_reuse_by_method: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dense = method_configs.get("dense_rag") or {}
    hybrid = method_configs.get("hybrid_rag") or {}
    ekell = method_configs.get("ekell_style_controlled_shared_llm") or {}
    dense_checksum = dense.get("dense_index_checksum") or (dense.get("dense_rag") or {}).get("index_checksum")
    hybrid_checksum = (
        hybrid.get("dense_index_checksum")
        or (hybrid.get("dense_rag") or {}).get("index_checksum")
        or (hybrid.get("hybrid_rag") or {}).get("dense_index_checksum")
    )
    ekell_checksum = ekell.get("ekell_index_checksum") or (ekell.get("ekell_vector") or {}).get("index_checksum")

    runtime = runtime_reuse_by_method or {}
    dense_rt = runtime.get("dense_rag") or {}
    hybrid_rt = runtime.get("hybrid_rag") or {}
    ekell_rt = runtime.get("ekell_style_controlled_shared_llm") or {}
    if dense_rt.get("index_checksum"):
        dense_checksum = dense_rt.get("index_checksum")
    if hybrid_rt.get("index_checksum"):
        hybrid_checksum = hybrid_rt.get("index_checksum")
    if ekell_rt.get("index_checksum"):
        ekell_checksum = ekell_rt.get("index_checksum")

    return {
        "dense": {"checksum": dense_checksum},
        "hybrid_dense_dependency": {"checksum": hybrid_checksum or dense_checksum},
        "ekell": {"checksum": ekell_checksum},
    }


def _assert_prediction_coverage(
    *,
    case_ids: list[str],
    methods: list[str],
    interop_rows: list[dict],
    allow_partial: bool,
) -> dict[str, Any]:
    expected_pairs = {(cid, mid) for cid in case_ids for mid in methods}
    observed: dict[tuple[str, str], int] = {}
    for row in interop_rows:
        key = (str(row.get("case_id")), str(row.get("method_id")))
        observed[key] = observed.get(key, 0) + 1
    duplicates = [f"{c}|{m}" for (c, m), n in observed.items() if n > 1]
    missing = sorted(f"{c}|{m}" for (c, m) in expected_pairs if (c, m) not in observed)
    extra = sorted(f"{c}|{m}" for (c, m) in observed if (c, m) not in expected_pairs)
    report = {
        "expected_case_count": len(case_ids),
        "methods_run": methods,
        "expected_prediction_count": len(expected_pairs),
        "prediction_count": len(interop_rows),
        "unique_case_method_count": len(observed),
        "duplicate_case_ids": duplicates,
        "missing_case_ids": missing,
        "extra_case_ids": extra,
    }
    if not allow_partial and (duplicates or missing or extra):
        raise SystemExit(f"Prediction coverage failure: {report}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run firebench-interop-v1 baselines from an experiment manifest + Runner Bundle."
    )
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--include-supplemental", action="store_true", help="Deprecated; prefer --method-set comparison_suite")
    parser.add_argument(
        "--method-set",
        choices=["main_table", "comparison_suite"],
        default="main_table",
        help="Which method set to run (default: main_table).",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--legacy-output", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-bundle-checksum", default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Debugging only: allow missing/duplicate/extra case×method predictions.",
    )
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--execution-stage",
        choices=["dry_run", "formal"],
        default="formal",
        help="dry_run: limited real API check; formal: frozen TEST run (default).",
    )
    parser.add_argument(
        "--override-readiness-lock",
        action="store_true",
        help="Manual bypass only; never use in CI or automation.",
    )
    args = parser.parse_args(argv)

    if args.config:
        raise SystemExit(
            "Multiple --config overlays are no longer supported for formal interop runs. "
            "Use a single --experiment-manifest."
        )

    if args.include_supplemental and args.method_set == "main_table":
        print(
            "WARNING: --include-supplemental is deprecated; using method_set=comparison_suite",
            file=sys.stderr,
        )
        args.method_set = "comparison_suite"

    experiment = load_experiment_manifest(args.experiment_manifest)
    bundle_path = args.bundle or experiment.get("bundle")
    output = args.output or experiment.get("output")
    limit = args.limit if args.limit is not None else experiment.get("limit")

    validation_stage = "dry_run" if args.execution_stage == "dry_run" else "formal"
    try:
        validate_experiment_manifest(
            args.experiment_manifest,
            validation_stage=validation_stage,
            method_set=args.method_set,
            runtime_bundle_path=bundle_path,
        )
    except FormalConfigError as exc:
        raise SystemExit(f"Stage-aware config validation failed ({validation_stage}): {exc}") from exc

    from external_baselines.common.execution_lock import assert_execution_allowed  # noqa: E402

    lock_audit = assert_execution_allowed(
        experiment_manifest=experiment,
        bundle_path=bundle_path,
        execution_stage=args.execution_stage,
        limit=limit,
        output_path=output,
        allow_partial=bool(args.allow_partial),
        override_readiness_lock=bool(args.override_readiness_lock),
    )

    if not bundle_path:
        raise SystemExit("Runner Bundle path required via --bundle or experiment manifest.bundle")

    assert_no_evaluator_bundle_access(bundle_path)
    formal = args.execution_stage == "formal"
    bundle = load_runner_bundle(bundle_path, formal=formal)
    expected_checksum = args.expected_bundle_checksum or experiment.get("expected_bundle_checksum")
    checksum_report = validate_bundle_checksum(bundle, expected=expected_checksum)
    if not checksum_report["ok"]:
        raise SystemExit(f"Bundle checksum mismatch: {checksum_report}")
    if not bundle.get("scenarios_path"):
        raise SystemExit("Runner Bundle does not contain input_cases / scenarios file.")

    local_schema_sha = sha256_file(SCHEMA_PATH) if SCHEMA_PATH.exists() else None
    bundle_schema_sha = bundle.get("prediction_schema_sha256")
    local_schema_snapshot_match = (
        bundle_schema_sha == local_schema_sha
        if bundle_schema_sha and local_schema_sha
        else None
    )
    schema_authority = "runner_bundle"

    method_entries = enabled_methods(
        experiment,
        include_supplemental=bool(args.include_supplemental),
        method_set=args.method_set,
    )
    if not method_entries:
        raise SystemExit("No enabled methods in experiment manifest (main-table empty?).")

    method_configs: dict[str, dict] = {}
    methods: list[str] = []
    for entry in method_entries:
        mid = entry["method_id"]
        cfg = build_method_config(experiment, entry)
        if bundle.get("corpus_dir"):
            cfg.setdefault("paths", {})["corpus_dir"] = bundle["corpus_dir"]
        cfg["bundle_checksum"] = bundle.get("producer_declared_checksum") or bundle.get(
            "consumer_computed_bundle_hash"
        )
        cfg["schema_version"] = "firebench-interop-v1"
        assert_paper_final_allowed(cfg)
        method_configs[mid] = cfg
        methods.append(mid)
    fairness_report = validate_cross_method_fairness(method_configs)

    freeze_path = experiment.get("freeze_manifest")
    if args.execution_stage == "formal" and freeze_path:
        from external_baselines.common.freeze_manifest import (  # noqa: E402
            validate_freeze_manifest,
            validate_frozen_runtime_inputs,
        )

        validate_freeze_manifest(
            freeze_path,
            experiment_manifest_path=args.experiment_manifest,
            experiment_raw=experiment.get("raw") or {},
            require_complete=(args.method_set == "comparison_suite"),
            expected_runner_bundle_checksum=bundle.get("producer_declared_checksum")
            or bundle.get("consumer_computed_bundle_hash"),
            expected_corpus_checksum=(bundle.get("corpus_manifest") or {}).get("aggregate_sha256")
            if isinstance(bundle.get("corpus_manifest"), dict)
            else None,
            expected_prediction_schema_checksum=bundle.get("prediction_schema_sha256"),
        )
        if args.method_set == "comparison_suite":
            from external_baselines.common.decision_suite_preflight import (  # noqa: E402
                preflight_decision_suite,
            )

            preflight = preflight_decision_suite(
                method_ids=methods,
                method_configs=method_configs,
                runner_bundle=Path(bundle_path),
                execution_stage="formal",
                experiment_manifest=Path(args.experiment_manifest),
            )
            if preflight.get("ok") is not True:
                raise FormalConfigError(
                    "; ".join(preflight.get("shared_errors") or [])
                    or "legacy_formal_comparison_suite_preflight_failed"
                )
        else:
            validate_frozen_runtime_inputs(
                freeze_path,
                bundle=bundle,
                method_configs=method_configs,
            )

    legacy_output = args.legacy_output or experiment.get("legacy_output")
    run_manifest = args.manifest or experiment.get("run_manifest")

    cases = load_scenarios(bundle["scenarios_path"], limit=limit)
    case_ids = [str(c.get("case_id") or c.get("scenario_id")) for c in cases]

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

    runtime_reuse: dict[str, Any] = {}
    if run_manifest and Path(run_manifest).is_file():
        existing_run = read_json(run_manifest)
        if isinstance(existing_run, dict):
            runtime_reuse = existing_run.get("runtime_reuse_by_method") or {}

    interop_rows = [
        baseline_row_to_interop(
            row,
            bundle_checksum=bundle.get("producer_declared_checksum")
            or bundle.get("consumer_computed_bundle_hash"),
        )
        for row in legacy_rows
    ]
    coverage = _assert_prediction_coverage(
        case_ids=case_ids,
        methods=methods,
        interop_rows=interop_rows,
        allow_partial=bool(args.allow_partial),
    )

    errors = []
    schema_path = bundle.get("prediction_schema_path") or SCHEMA_PATH
    expected_schema_sha = bundle.get("prediction_schema_sha256") or experiment.get(
        "expected_prediction_schema_sha256"
    )
    for i, row in enumerate(interop_rows):
        errs = validate_interop_record(
            row,
            schema_path=schema_path,
            expected_schema_sha256=expected_schema_sha if experiment.get("paper_final") else None,
            require_external_schema=True,
        )
        if errs:
            errors.append({"index": i, "case_id": row.get("case_id"), "errors": errs})
    if errors:
        write_json(Path(output).with_suffix(".validation_errors.json"), errors)
        raise SystemExit(f"Interop schema validation failed for {len(errors)} records.")

    write_jsonl(output, interop_rows)
    hash_report = _verify_bundle_hashes(bundle)
    llm_summary = llm_config_summary(shared_snapshot)
    indexes_snapshot = _index_checksum_snapshot(method_configs, runtime_reuse_by_method=runtime_reuse)
    dense_cs = (indexes_snapshot.get("dense") or {}).get("checksum")
    hybrid_cs = (indexes_snapshot.get("hybrid_dense_dependency") or {}).get("checksum")
    if dense_cs and hybrid_cs and str(dense_cs) != str(hybrid_cs):
        raise SystemExit(
            f"Hybrid dense dependency checksum must match dense: dense={dense_cs} hybrid={hybrid_cs}"
        )
    paper_valid = (
        args.execution_stage == "formal"
        and bool(lock_audit.get("paper_valid"))
        and not lock_audit.get("execution_lock_overridden")
        and not errors
        and not coverage.get("duplicate_case_ids")
        and not coverage.get("missing_case_ids")
        and not coverage.get("extra_case_ids")
    )
    enriched_manifest = {
        "execution_stage": args.execution_stage,
        "method_set": args.method_set,
        "methods_run": methods,
        "experiment_manifest_checksum": sha256_file(experiment.get("manifest_path")),
        "runner_bundle_checksum": bundle.get("producer_declared_checksum")
        or bundle.get("consumer_computed_bundle_hash"),
        "corpus_checksum": (bundle.get("corpus_manifest") or {}).get("aggregate_sha256")
        if isinstance(bundle.get("corpus_manifest"), dict)
        else None,
        "prediction_schema_checksum": bundle.get("prediction_schema_sha256") or sha256_file(SCHEMA_PATH),
        "prediction_schema_provenance": {
            "prediction_schema_path": bundle.get("prediction_schema_path"),
            "prediction_schema_source": bundle.get("prediction_schema_source"),
            "prediction_schema_inside_bundle": bundle.get("prediction_schema_inside_bundle"),
            "prediction_schema_declared_sha256": bundle.get("prediction_schema_declared_sha256"),
            "prediction_schema_sha256": bundle.get("prediction_schema_sha256"),
            "prediction_schema_checksum_match": bundle.get("prediction_schema_checksum_match"),
            "prediction_schema_authoritative": bundle.get("prediction_schema_authoritative"),
            "prediction_schema_formal_eligible": bundle.get("prediction_schema_formal_eligible"),
        },
        "schema_authority": schema_authority,
        "local_schema_snapshot_match": local_schema_snapshot_match,
        "local_schema_snapshot_authoritative": False,
        "freeze_status": experiment.get("freeze_status"),
        "freeze_manifest": experiment.get("freeze_manifest"),
        "llm": {
            "provider": llm_summary.get("provider"),
            "model": llm_summary.get("model"),
            "model_version": llm_summary.get("model_version"),
            "model_source": llm_summary.get("model_source") or "yaml_config",
        },
        "embedding": _shared_embedding_snapshot(method_configs),
        "indexes": indexes_snapshot,
        "fairness": fairness_report,
        "coverage": coverage,
        "execution_lock_overridden": bool(lock_audit.get("execution_lock_overridden")),
        "paper_valid": paper_valid,
        "cross_method_fairness": fairness_report,
        "runtime_reuse_by_method": runtime_reuse,
    }
    if run_manifest and Path(run_manifest).is_file():
        existing = read_json(run_manifest)
        if isinstance(existing, dict):
            existing.update(enriched_manifest)
            write_json(run_manifest, existing)
        else:
            write_json(run_manifest, enriched_manifest)
    elif run_manifest:
        write_json(run_manifest, enriched_manifest)

    write_json(
        Path(run_manifest).with_name("interop_bundle_report.json"),
        {
            "experiment_id": experiment.get("experiment_id"),
            "experiment_manifest": experiment.get("manifest_path"),
            "freeze_status": experiment.get("freeze_status"),
            "main_table_methods": list(MAIN_TABLE_METHODS),
            "methods_run": methods,
            "method_set": args.method_set,
            "include_supplemental": bool(args.include_supplemental),
            "execution_stage": args.execution_stage,
            "execution_lock_overridden": bool(lock_audit.get("execution_lock_overridden")),
            "paper_valid": paper_valid,
            "cross_method_fairness": fairness_report,
            "llm": enriched_manifest["llm"],
            "embedding": enriched_manifest["embedding"],
            "indexes": enriched_manifest["indexes"],
            "fairness": fairness_report,
            "coverage": coverage,
            "bundle": {
                "root": bundle.get("bundle_root"),
                "producer_declared_checksum": bundle.get("producer_declared_checksum"),
                "consumer_computed_bundle_hash": bundle.get("consumer_computed_bundle_hash"),
                "scenarios_path": bundle.get("scenarios_path"),
                "corpus_dir": bundle.get("corpus_dir"),
                "formal_manifest_files_used": bundle.get("formal_manifest_files_used"),
                "prediction_schema_sha256": bundle.get("prediction_schema_sha256"),
                "prediction_schema_provenance": enriched_manifest["prediction_schema_provenance"],
                "schema_authority": schema_authority,
                "local_schema_snapshot_sha256": local_schema_sha,
                "local_schema_snapshot_match": local_schema_snapshot_match,
                "local_schema_snapshot_authoritative": False,
            },
            "checksum_validation": checksum_report,
            "hash_verification": hash_report,
            "n_predictions": len(interop_rows),
            "output": output,
            "cross_repository_interop_verified": False,
        },
    )
    print(f"Wrote {len(interop_rows)} firebench-interop-v1 predictions to {output}")
    print(f"Methods: {methods}")
    print(f"Producer checksum: {bundle.get('producer_declared_checksum')}")
    print(f"Consumer hash: {bundle.get('consumer_computed_bundle_hash')}")
    print("cross_repository_interop_verified=false")


if __name__ == "__main__":
    main()
