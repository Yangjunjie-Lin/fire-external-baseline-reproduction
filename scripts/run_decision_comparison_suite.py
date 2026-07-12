#!/usr/bin/env python3
"""Run the five-method decision comparison suite against a FireBench Runner Bundle.

Emits per-method firebench-interop-v1 prediction JSONL plus human-readable
decision/response artifacts. Does not score against gold; formal evaluation
remains owned by fire-agent-demo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.checksums import sha256_file  # noqa: E402
from external_baselines.common.decision_output import (  # noqa: E402
    DecisionParseError,
    unified_row_to_interop,
)
from external_baselines.common.decision_suite_guard import (  # noqa: E402
    FORMAL_LIMIT_FORBIDDEN_MESSAGE,
    FormalCoverageError,
    FormalRunFailed,
    FormalSuiteExecutionError,
    FormalTaxonomyError,
    assert_formal_smoke_config_forbidden,
    sanitize_error_message,
    validate_decision_suite_execution,
    validate_formal_method_configs,
)
from external_baselines.common.decision_suite_preflight import preflight_decision_suite  # noqa: E402
from external_baselines.common.formal_config_validator import FormalConfigError  # noqa: E402
from external_baselines.common.io import (  # noqa: E402
    assert_no_gold_in_prediction_input,
    ensure_dir,
    load_scenarios,
    to_prediction_input,
    write_json,
    write_jsonl,
)
from external_baselines.common.llm_client import (  # noqa: E402
    TokenUsage,
    UsageTrackingLLMClient,
    build_llm_client,
)
from external_baselines.common.method_runtime import (  # noqa: E402
    close_method_runtime,
    pipeline_accepts_runtime,
    prepare_method_runtime,
)
from external_baselines.common.runtime_evidence import (  # noqa: E402
    collect_method_runtime_evidence,
    compute_suite_formal_compliance,
    evidence_to_summary_sections,
    method_formal_compliance,
)
from external_baselines.common.taxonomy_normalizer import assert_canonical_interop_record  # noqa: E402
from external_baselines.interop.bundle import (  # noqa: E402
    assert_no_evaluator_bundle_access,
    inspect_runner_bundle_case_coverage,
    load_runner_bundle,
    validate_formal_runner_bundle_coverage,
)
from external_baselines.interop.schema import SCHEMA_PATH, validate_interop_record  # noqa: E402
from external_baselines.method_registry import (  # noqa: E402
    canonicalize_method_id,
    comparison_suite_methods,
    resolve_pipeline,
)

COMPARISON_METHODS = list(comparison_suite_methods())


def _base_smoke_config(corpus_dir: str | Path | None, *, execution_stage: str) -> dict[str, Any]:
    assert_formal_smoke_config_forbidden(execution_stage=execution_stage)
    return {
        "execution_stage": execution_stage,
        "unified_decision_output": True,
        "strict_decision_parse": execution_stage == "formal",
        "dev_aliases_enabled": False,
        "paper_final": False,
        "llm": {
            "provider": "heuristic",
            "model": "local-deterministic-heuristic-smoke-test",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "seed": 20260710,
        },
        "paths": {"corpus_dir": str(corpus_dir or "data/corpus")},
        "retrieval": {"top_k": 5, "max_chunk_chars": 1000},
        "dense_rag": {
            "backend": "smoke",
            "model_name": "smoke-hash-embedding",
            "model_version": "v0-smoke",
            "dimension": 64,
            "top_k": 5,
            "reject_smoke": False,
            "allow_index_rebuild": True,
        },
        "hybrid_rag": {
            "top_k": 5,
            "rrf_k": 60,
            "lexical_weight": 1.0,
            "dense_weight": 1.0,
            "candidate_pool": 20,
            "reject_smoke": False,
        },
        "ekell_style": {
            "prompt_dir": "configs/prompts/controlled",
            "neighborhood_k_hop": 1,
            "neighborhood_max_nodes": 50,
            "neighborhood_max_triples": 80,
        },
        "ekell_vector": {
            "backend": "smoke",
            "dimension": 32,
            "top_k": 8,
            "reject_smoke": False,
        },
        "scenario_parser": {"use_llm": False},
        "normalization": {"infer_structured_safety_fields": False},
    }


def _method_config(
    method_id: str,
    *,
    base: dict[str, Any],
    experiment_manifest: Path | None,
) -> dict[str, Any]:
    cfg = dict(base)
    if experiment_manifest and experiment_manifest.is_file():
        from external_baselines.common.experiment_manifest import (
            build_method_config,
            get_method_entry,
            load_experiment_manifest,
        )

        experiment = load_experiment_manifest(experiment_manifest)
        method_entry = get_method_entry(experiment, method_id)
        cfg = build_method_config(experiment, method_entry)
        cfg["execution_stage"] = base.get("execution_stage", "dry_run")
        cfg["unified_decision_output"] = True
        cfg["strict_decision_parse"] = base.get("execution_stage") == "formal"
        cfg.setdefault("normalization", {})["infer_structured_safety_fields"] = False
    return cfg


def _assert_method_coverage(
    *,
    case_ids: list[str],
    method_id: str,
    rows: list[dict[str, Any]],
    formal: bool,
) -> dict[str, Any]:
    observed = [str(r.get("case_id")) for r in rows]
    method_ids = {str(r.get("method_id")) for r in rows}
    duplicates = sorted({cid for cid in observed if observed.count(cid) > 1})
    missing = sorted(set(case_ids) - set(observed))
    extra = sorted(set(observed) - set(case_ids))
    report = {
        "method_id": method_id,
        "expected_case_count": len(case_ids),
        "prediction_count": len(rows),
        "duplicate_case_ids": duplicates,
        "missing_case_ids": missing,
        "extra_case_ids": extra,
        "method_ids_in_file": sorted(method_ids),
    }
    errors: list[str] = []
    if len(rows) != len(case_ids):
        errors.append("prediction_count_mismatch")
    if duplicates:
        errors.append("duplicate_case_ids")
    if missing:
        errors.append("missing_case_ids")
    if extra:
        errors.append("extra_case_ids")
    if method_ids != {method_id}:
        errors.append("mixed_or_wrong_method_id")
    report["errors"] = errors
    if formal and errors:
        raise FormalCoverageError(f"Coverage failure for {method_id}: {report}")
    return report


def _interop_taxonomy_valid(interop_rows: list[dict[str, Any]]) -> bool:
    for row in interop_rows:
        meta = row.get("method_metadata") or {}
        if meta.get("parsing_failure"):
            return False
        if meta.get("taxonomy_unmapped"):
            return False
    return True


def _integrity_flags_from_preflight(preflight: dict[str, Any]) -> dict[str, bool]:
    integrity = preflight.get("runner_bundle_integrity") or {}
    return {
        "runner_bundle_integrity_ok": integrity.get("ok") is True,
        "input_cases_integrity_ok": integrity.get("input_cases_integrity") is True,
        "prediction_schema_integrity_ok": integrity.get("prediction_schema_integrity") is True,
        "corpus_integrity_ok": integrity.get("corpus_integrity") is True,
    }


@dataclass
class FormalRunLayout:
    run_root: Path
    prediction_dir: Path
    decision_dir: Path


def resolve_formal_run_layout(
    *,
    formal_run_root: Path | None,
    prediction_dir: Path,
    decision_dir: Path,
) -> FormalRunLayout:
    pred_res = prediction_dir.resolve()
    dec_res = decision_dir.resolve()
    if formal_run_root is not None:
        root = formal_run_root.resolve()
        expected_pred = root / "predictions"
        expected_dec = root / "decisions"
        if pred_res != expected_pred.resolve() or dec_res != expected_dec.resolve():
            raise FormalSuiteExecutionError("formal_output_paths_must_share_run_root")
    else:
        if pred_res.parent != dec_res.parent:
            raise FormalSuiteExecutionError("formal_output_paths_must_share_run_root")
        if pred_res.name != "predictions" or dec_res.name != "decisions":
            raise FormalSuiteExecutionError("formal_output_paths_must_share_run_root")
        root = dec_res.parent
    return FormalRunLayout(
        run_root=root,
        prediction_dir=root / "predictions",
        decision_dir=root / "decisions",
    )


def paths_same_filesystem(path_a: Path, path_b: Path) -> bool:
    a = path_a.resolve()
    b = path_b.resolve()
    if os.name == "nt":
        return a.drive.lower() == b.drive.lower()
    return a.stat().st_dev == b.stat().st_dev


def assert_same_filesystem(path_a: Path, path_b: Path) -> None:
    if not paths_same_filesystem(path_a, path_b):
        raise FormalSuiteExecutionError("formal_atomic_publish_cross_filesystem_forbidden")


def create_formal_temp_run_root(final_run_root: Path, run_id: str) -> Path:
    return final_run_root.parent / f".{final_run_root.name}.tmp_{run_id}"


@dataclass
class RunRootPublishState:
    temp_run_root: Path
    final_run_root: Path
    backup_path: Path | None = None
    original_existed: bool = False
    backup_prepared: bool = False
    committed: bool = False
    restored: bool = False
    rollback_error: str | None = None


@dataclass
class PublishTargetState:
    name: str
    temp_path: Path
    final_path: Path
    backup_path: Path | None = None
    original_existed: bool = False
    backup_prepared: bool = False
    published: bool = False
    restored: bool = False
    rollback_error: str | None = None


@dataclass
class FormalPublishResult:
    success: bool
    committed: bool
    cleanup_complete: bool
    cleanup_warnings: list[str]
    rollback_attempted: bool
    rollback_succeeded: bool
    run_root_state: RunRootPublishState | None = None
    targets: list[PublishTargetState] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class FormalPublishError(RuntimeError):
    """Raised when transactional formal publish fails after rollback is attempted."""

    def __init__(self, message: str, *, publish_result: FormalPublishResult) -> None:
        super().__init__(message)
        self.publish_result = publish_result


def _prepare_target_backup(state: PublishTargetState) -> None:
    state.original_existed = state.final_path.exists()
    if not state.original_existed:
        state.backup_path = None
        state.backup_prepared = True
        return
    backup = state.final_path.with_name(f"{state.final_path.name}.bak")
    if backup.exists():
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    state.final_path.rename(backup)
    state.backup_path = backup
    state.backup_prepared = True


def _publish_temp_directory(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Formal temp directory missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _rollback_target(state: PublishTargetState) -> None:
    if state.final_path.exists():
        _remove_path(state.final_path)
    if state.original_existed:
        if state.backup_path is None or not state.backup_path.exists():
            raise FileNotFoundError(f"backup missing for {state.name}")
        state.backup_path.rename(state.final_path)
    state.restored = True
    state.rollback_error = None


def _remove_directory_backup(backup: Path | None) -> None:
    if backup is None or not backup.exists():
        return
    _remove_path(backup)


def rollback_all_targets(states: list[PublishTargetState]) -> FormalPublishResult:
    errors: list[str] = []
    rollback_attempted = any(state.backup_prepared or state.published for state in states)
    for state in reversed(states):
        if not (state.backup_prepared or state.published):
            continue
        try:
            _rollback_target(state)
        except Exception as exc:  # noqa: BLE001
            state.restored = False
            state.rollback_error = sanitize_error_message(str(exc))
            errors.append(f"restore_{state.name}_failed: {state.rollback_error}")
    rollback_succeeded = rollback_attempted and not errors
    return FormalPublishResult(
        success=False,
        committed=False,
        cleanup_complete=False,
        cleanup_warnings=[],
        rollback_attempted=rollback_attempted,
        rollback_succeeded=rollback_succeeded,
        targets=list(states),
        errors=errors,
    )


def _rollback_run_root(state: RunRootPublishState) -> FormalPublishResult:
    errors: list[str] = []
    rollback_attempted = state.backup_prepared or state.committed
    try:
        if state.committed and state.final_run_root.exists():
            _remove_path(state.final_run_root)
        if state.original_existed:
            if state.backup_path is None or not state.backup_path.exists():
                raise FileNotFoundError("backup missing for formal run root")
            state.backup_path.rename(state.final_run_root)
        state.restored = True
        state.rollback_error = None
    except Exception as exc:  # noqa: BLE001
        state.restored = False
        state.rollback_error = sanitize_error_message(str(exc))
        errors.append(f"restore_run_root_failed: {state.rollback_error}")
    rollback_succeeded = rollback_attempted and not errors
    return FormalPublishResult(
        success=False,
        committed=False,
        cleanup_complete=False,
        cleanup_warnings=[],
        rollback_attempted=rollback_attempted,
        rollback_succeeded=rollback_succeeded,
        run_root_state=state,
        errors=errors,
    )


def publish_formal_run_root_transactionally(
    *,
    temp_run_root: Path,
    final_run_root: Path,
) -> FormalPublishResult:
    """Publish a formal run root with PREPARE / COMMIT / CLEANUP phases."""
    assert_same_filesystem(temp_run_root.parent, final_run_root.parent)
    state = RunRootPublishState(temp_run_root=temp_run_root, final_run_root=final_run_root)
    cleanup_warnings: list[str] = []

    # PREPARE
    try:
        state.original_existed = final_run_root.exists()
        if state.original_existed:
            backup = final_run_root.with_name(f"{final_run_root.name}.bak")
            if backup.exists():
                _remove_path(backup)
            final_run_root.rename(backup)
            state.backup_path = backup
        state.backup_prepared = True
    except Exception as exc:
        result = _rollback_run_root(state)
        raise FormalPublishError(sanitize_error_message(str(exc)), publish_result=result) from exc

    # COMMIT
    try:
        if not temp_run_root.exists():
            raise FileNotFoundError(f"Formal temp run root missing: {temp_run_root}")
        temp_run_root.rename(final_run_root)
        state.committed = True
    except Exception as exc:
        result = _rollback_run_root(state)
        raise FormalPublishError(sanitize_error_message(str(exc)), publish_result=result) from exc

    # CLEANUP
    cleanup_complete = True
    try:
        _remove_directory_backup(state.backup_path)
        state.backup_path = None
    except Exception as exc:  # noqa: BLE001
        cleanup_complete = False
        cleanup_warnings.append(f"backup_cleanup_failed: {sanitize_error_message(str(exc))}")

    return FormalPublishResult(
        success=True,
        committed=True,
        cleanup_complete=cleanup_complete,
        cleanup_warnings=cleanup_warnings,
        rollback_attempted=False,
        rollback_succeeded=True,
        run_root_state=state,
        errors=[],
    )


def publish_formal_artifacts_transactionally(
    *,
    temp_prediction_dir: Path,
    temp_decision_dir: Path,
    final_prediction_dir: Path,
    final_decision_dir: Path,
) -> FormalPublishResult:
    """Backward-compatible wrapper that publishes via a shared formal run root."""
    temp_run_root = temp_prediction_dir.parent
    final_run_root = final_prediction_dir.parent
    if temp_decision_dir.parent != temp_run_root or final_decision_dir.parent != final_run_root:
        raise FormalSuiteExecutionError("formal_output_paths_must_share_run_root")
    return publish_formal_run_root_transactionally(
        temp_run_root=temp_run_root,
        final_run_root=final_run_root,
    )


def _cleanup_formal_temp(temp_root: Path) -> None:
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


def write_formal_failure_diagnostics(
    *,
    diagnostics_dir: Path,
    run_id: str,
    suite_summary: dict[str, Any],
) -> Path:
    path = diagnostics_dir / f"formal_failure_summary_{run_id}.json"
    write_json(path, suite_summary)
    return path


def _publish_targets_to_marker(states: list[PublishTargetState]) -> dict[str, Any]:
    return {
        state.name: {
            "original_existed": state.original_existed,
            "published": state.published,
            "restored": state.restored,
            "rollback_error": state.rollback_error,
        }
        for state in states
    }


def _write_formal_failed_marker(
    *,
    decision_parent: Path,
    run_id: str,
    stage: str,
    method_id: str | None,
    error_type: str,
    error_message: str,
    temporary_artifacts_path: str | None,
    rollback_attempted: bool = False,
    rollback_succeeded: bool = False,
    rollback_errors: list[str] | None = None,
    published_targets: list[str] | None = None,
    targets: dict[str, Any] | None = None,
    failure_summary_path: str | None = None,
) -> None:
    marker = {
        "execution_stage": "formal",
        "run_id": run_id,
        "stage": stage,
        "method_id": method_id,
        "error_type": error_type,
        "error_message": sanitize_error_message(error_message),
        "temporary_artifacts_path": temporary_artifacts_path,
        "formal_outputs_published": False,
        "rollback_attempted": rollback_attempted,
        "rollback_succeeded": rollback_succeeded,
        "rollback_errors": list(rollback_errors or []),
        "published_targets": list(published_targets or []),
        "targets": dict(targets or {}),
        "failure_summary_path": failure_summary_path,
    }
    write_json(decision_parent / "FORMAL_RUN_FAILED.json", marker)


def _classify_formal_failure_stage(exc: Exception) -> str:
    if isinstance(exc, FormalSuiteExecutionError):
        message = str(exc)
        if "manifest" in message or "freeze" in message:
            return "manifest_validation"
        if "coverage" in message or "bundle" in message:
            return "coverage_validation"
        if "run_root" in message or "filesystem" in message:
            return "formal_run_root_validation"
        return "manifest_validation"
    if isinstance(exc, FormalConfigError):
        return "method_config_validation"
    if isinstance(exc, FormalCoverageError):
        return "coverage_validation"
    if isinstance(exc, FormalTaxonomyError):
        return "taxonomy_validation"
    return "formal"


def _raise_formal_failure(
    *,
    message: str,
    stage: str,
    run_id: str,
    marker_dir: Path,
    suite_summary: dict[str, Any],
    temp_root: Path | None,
    keep_failed_temp_artifacts: bool,
    method_id: str | None = None,
    error_type: str = "FormalRunFailed",
    rollback_attempted: bool = False,
    rollback_succeeded: bool = False,
    rollback_errors: list[str] | None = None,
    published_targets: list[str] | None = None,
    targets: dict[str, Any] | None = None,
) -> None:
    diagnostics_dir = ensure_dir(marker_dir / "diagnostics")
    failure_summary_path = write_formal_failure_diagnostics(
        diagnostics_dir=diagnostics_dir,
        run_id=run_id,
        suite_summary=suite_summary,
    )
    _write_formal_failed_marker(
        decision_parent=marker_dir,
        run_id=run_id,
        stage=stage,
        method_id=method_id,
        error_type=error_type,
        error_message=message,
        temporary_artifacts_path=str(temp_root) if temp_root and keep_failed_temp_artifacts else None,
        rollback_attempted=rollback_attempted,
        rollback_succeeded=rollback_succeeded,
        rollback_errors=rollback_errors,
        published_targets=published_targets,
        targets=targets,
        failure_summary_path=str(failure_summary_path.relative_to(marker_dir)),
    )
    if temp_root and not keep_failed_temp_artifacts:
        _cleanup_formal_temp(temp_root)
    raise FormalRunFailed(message, stage=stage, summary=suite_summary)


def _raise_formal_early_failure(
    *,
    exc: Exception,
    run_id: str,
    marker_dir: Path,
    temp_root: Path | None,
    keep_failed_temp_artifacts: bool,
) -> None:
    suite_summary = {
        "execution_stage": "formal",
        "formal": True,
        "formal_compliance": {"pre_publish_compliance_passed": False, "formal_result": False},
    }
    _raise_formal_failure(
        message=str(exc),
        stage=_classify_formal_failure_stage(exc),
        run_id=run_id,
        marker_dir=marker_dir,
        suite_summary=suite_summary,
        temp_root=temp_root,
        keep_failed_temp_artifacts=keep_failed_temp_artifacts,
        error_type=type(exc).__name__,
    )


def _write_decision_artifacts(
    decision_dir: Path,
    method_id: str,
    interop_rows: list[dict[str, Any]],
    *,
    input_cases_sha256: str | None,
    prediction_schema_sha256: str | None,
    prediction_file: Path,
    formal: bool = False,
    coverage_ok: bool = True,
    runtime_evidence: Any | None = None,
    method_compliance: dict[str, Any] | None = None,
    execution_stage: str = "dry_run",
) -> dict[str, Any]:
    from external_baselines.common.firebench_taxonomy import alias_sha256, taxonomy_provenance, taxonomy_sha256

    method_dir = ensure_dir(decision_dir / method_id)
    decisions = []
    responses = []
    unmapped_rows: list[dict[str, Any]] = []
    parsing_failures = 0
    schema_failures = 0
    alias_applied_count = 0
    affected_cases: set[str] = set()
    latencies: list[float] = []
    llm_calls = 0
    for row in interop_rows:
        pred = row.get("prediction") or {}
        fr = pred.get("final_response") or {}
        meta = row.get("method_metadata") or {}
        case_id = str(row.get("case_id") or "")
        if meta.get("parsing_failure"):
            parsing_failures += 1
        aliases = list(meta.get("taxonomy_aliases_applied") or [])
        alias_applied_count += len(aliases)
        for item in meta.get("taxonomy_unmapped") or []:
            affected_cases.add(case_id)
            unmapped_rows.append(
                {
                    "case_id": case_id,
                    "method_id": method_id,
                    "field": item.get("field"),
                    "raw_value": item.get("raw_value"),
                    "normalized_value": item.get("normalized_value"),
                    "reason": item.get("reason") or "not_in_firebench_taxonomy",
                }
            )
        decisions.append(
            {
                "case_id": row.get("case_id"),
                "method_id": method_id,
                "decision": {
                    "risk_signals": pred.get("risk_signals") or [],
                    "risk_level": pred.get("risk_level"),
                    "recommended_actions": pred.get("recommended_actions") or [],
                    "blocked_actions": pred.get("blocked_actions") or [],
                    "missing_confirmations": pred.get("missing_confirmations") or [],
                    "human_review_required": pred.get("human_review_required"),
                    "final_decision_gate": pred.get("final_decision_gate"),
                },
            }
        )
        responses.append(
            {
                "case_id": row.get("case_id"),
                "method_id": method_id,
                "natural_language_response": fr.get("text") or "",
                "citations": fr.get("citations") or [],
            }
        )
        rt = row.get("runtime") or {}
        if rt.get("latency_ms") is not None:
            latencies.append(float(rt["latency_ms"]))
        if rt.get("llm_calls") is not None:
            llm_calls += int(rt["llm_calls"] or 0)
    write_jsonl(method_dir / "decisions.jsonl", decisions)
    write_jsonl(method_dir / "responses.jsonl", responses)
    write_jsonl(method_dir / "unmapped_taxonomy.jsonl", unmapped_rows)
    taxonomy_valid = len(unmapped_rows) == 0 and parsing_failures == 0
    if formal and not taxonomy_valid:
        raise FormalTaxonomyError(
            f"Formal taxonomy validation failed for {method_id}: "
            f"unmapped={len(unmapped_rows)} parsing_failures={parsing_failures}"
        )
    summary = {
        "method_id": method_id,
        "execution_stage": execution_stage,
        "case_count": len(interop_rows),
        "successful_count": len(interop_rows) - parsing_failures - schema_failures,
        "parsing_failure_count": parsing_failures,
        "schema_failure_count": schema_failures,
        "average_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "llm_call_count": llm_calls,
        "input_cases_sha256": input_cases_sha256 or "",
        "prediction_schema_sha256": prediction_schema_sha256 or "",
        "prediction_file_sha256": sha256_file(prediction_file) if prediction_file.is_file() else "",
        "formal_result": bool((method_compliance or {}).get("formal_result", False)),
        "runtime_evidence": evidence_to_summary_sections(runtime_evidence)
        if runtime_evidence is not None
        else None,
        "formal_compliance": method_compliance or {},
        "taxonomy_validation": {
            "valid": taxonomy_valid,
            "invalid_id_count": len(unmapped_rows),
            "alias_applied_count": alias_applied_count,
            "affected_case_count": len(affected_cases),
            "taxonomy_sha256": taxonomy_sha256(),
            "alias_sha256": alias_sha256(),
            "provenance": taxonomy_provenance(),
        },
    }
    write_json(method_dir / "run_summary.json", summary)
    return summary


def run_decision_suite(
    *,
    runner_bundle: Path,
    prediction_dir: Path,
    decision_dir: Path,
    execution_stage: str = "dry_run",
    limit: int | None = None,
    experiment_manifest: Path | None = None,
    methods: list[str] | None = None,
    dev_aliases_enabled: bool = False,
    keep_failed_temp_artifacts: bool = False,
    formal_run_root: Path | None = None,
) -> dict[str, Any]:
    formal = execution_stage == "formal"
    run_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    temp_root: Path | None = None
    write_prediction_dir = prediction_dir
    write_decision_dir = decision_dir
    layout: FormalRunLayout | None = None
    marker_dir = decision_dir.parent
    try:

        assert_no_evaluator_bundle_access(runner_bundle)
        method_ids = [canonicalize_method_id(m) for m in (methods or COMPARISON_METHODS)]

        if formal:
            layout = resolve_formal_run_layout(
                formal_run_root=formal_run_root,
                prediction_dir=prediction_dir,
                decision_dir=decision_dir,
            )
            prediction_dir = layout.prediction_dir
            decision_dir = layout.decision_dir
            marker_dir = layout.run_root

        validate_decision_suite_execution(
            execution_stage=execution_stage,
            experiment_manifest=experiment_manifest,
            method_ids=method_ids,
            runner_bundle=runner_bundle,
            limit=limit,
        )

        load_limit = None if formal else limit
        coverage = inspect_runner_bundle_case_coverage(runner_bundle, limit=load_limit)
        coverage_warning: str | None = None
        if formal:
            validate_formal_runner_bundle_coverage(coverage)
        elif (
            coverage.manifest_case_count is not None
            and coverage.manifest_case_count != coverage.input_file_case_count
        ):
            coverage_warning = (
                "runner_bundle_case_count_mismatch: "
                f"manifest={coverage.manifest_case_count} input_cases={coverage.input_file_case_count}"
            )
        else:
            coverage_warning = None

        if formal:
            validate_formal_method_configs(
                method_ids=method_ids,
                experiment_manifest=experiment_manifest,  # type: ignore[arg-type]
                runner_bundle=runner_bundle,
            )

        bundle = load_runner_bundle(runner_bundle)
        scenarios_path = Path(bundle["scenarios_path"])
        schema_path = Path(bundle.get("prediction_schema_path") or SCHEMA_PATH)
        corpus_dir = bundle.get("corpus_dir")
        scenarios = load_scenarios(scenarios_path, limit=load_limit)
        case_ids = coverage.input_case_ids if formal else coverage.loaded_case_ids
        input_cases_sha = sha256_file(scenarios_path)
        schema_sha = sha256_file(schema_path) if schema_path.is_file() else None

        if formal:
            base: dict[str, Any] = {
                "execution_stage": "formal",
                "unified_decision_output": True,
                "strict_decision_parse": True,
                "dev_aliases_enabled": False,
                "paper_final": True,
                "normalization": {"infer_structured_safety_fields": False},
            }
        else:
            base = _base_smoke_config(corpus_dir, execution_stage=execution_stage)
            if dev_aliases_enabled:
                base["dev_aliases_enabled"] = True
        if corpus_dir:
            base.setdefault("paths", {})["corpus_dir"] = str(corpus_dir)

        method_configs: dict[str, dict[str, Any]] = {}
        for method_id in method_ids:
            method_config = _method_config(
                method_id,
                base=base,
                experiment_manifest=experiment_manifest,
            )
            if corpus_dir:
                method_config.setdefault("paths", {})["corpus_dir"] = str(corpus_dir)
            method_config["unified_decision_output"] = True
            method_config["strict_decision_parse"] = formal
            method_config["execution_stage"] = execution_stage
            method_config["dev_aliases_enabled"] = bool(dev_aliases_enabled and not formal)
            method_configs[method_id] = method_config

        preflight = preflight_decision_suite(
            method_ids=method_ids,
            method_configs=method_configs,
            runner_bundle=runner_bundle,
            execution_stage=execution_stage,
            experiment_manifest=experiment_manifest,
        )
        diagnostics_dir = ensure_dir(marker_dir / "diagnostics")
        write_json(diagnostics_dir / "decision_suite_preflight.json", preflight)
        if formal and not preflight.get("ok"):
            suite_summary = {
                "execution_stage": execution_stage,
                "formal": formal,
                "preflight_ok": False,
                "formal_compliance": {"pre_publish_compliance_passed": False, "formal_result": False},
            }
            _raise_formal_failure(
                message="Formal decision suite preflight failed.",
                stage="preflight",
                run_id=run_id,
                marker_dir=marker_dir,
                suite_summary=suite_summary,
                temp_root=temp_root,
                keep_failed_temp_artifacts=keep_failed_temp_artifacts,
                error_type="PreflightError",
            )

        if formal and layout is not None:
            temp_root = create_formal_temp_run_root(layout.run_root, run_id)
            write_prediction_dir = ensure_dir(temp_root / "predictions")
            write_decision_dir = ensure_dir(temp_root / "decisions")

        ensure_dir(write_prediction_dir)
        ensure_dir(write_decision_dir)
        from external_baselines.common.firebench_taxonomy import (
            alias_sha256,
            formal_alias_sha256,
            taxonomy_sha256,
        )

        integrity_flags = _integrity_flags_from_preflight(preflight)
        suite_summary: dict[str, Any] = {
            "execution_stage": execution_stage,
            "formal": formal,
            "runner_bundle": str(runner_bundle),
            "case_count": len(case_ids),
            "methods": method_ids,
            "method_summaries": {},
            "coverage": {},
            "runner_bundle_coverage": coverage.to_dict(),
            "warnings": [coverage_warning] if coverage_warning else [],
            "preflight_ok": bool(preflight.get("ok")),
            "taxonomy_contract": {
                "taxonomy_version": "firebench-taxonomy-v1",
                "taxonomy_sha256": taxonomy_sha256(),
                "alias_sha256": alias_sha256(),
                "formal_alias_sha256": formal_alias_sha256(),
                "all_methods_valid": True,
                "dev_aliases_enabled": bool(dev_aliases_enabled and not formal),
            },
            "formal_compliance": compute_suite_formal_compliance(
                formal=formal,
                experiment_manifest_provided=experiment_manifest is not None,
                limit_used=limit is not None,
                preflight_ok=bool(preflight.get("ok")),
                coverage_ok=False,
                method_evidences={},
                method_compliance={},
                dev_aliases_enabled=bool(dev_aliases_enabled and not formal),
                shared_generation_identity_match=bool(
                    (preflight.get("shared_generation_identity") or {}).get("ok")
                ),
                ekell_prompt_bundle_valid=bool(preflight.get("ekell_prompt_bundle_valid")),
                method_ids=method_ids,
                phase="pre_publish",
                **integrity_flags,
            ),
        }

        method_evidences: dict[str, Any] = {}
        method_compliance_reports: dict[str, dict[str, Any]] = {}
        all_coverage_ok = True

        try:
            for method_id in method_ids:
                method_config = method_configs[method_id]
                llm = build_llm_client(method_config)
                pipeline = resolve_pipeline(method_id)
                runtime = prepare_method_runtime(method_id, method_config)
                evidence = collect_method_runtime_evidence(
                    method_id=method_id,
                    config=method_config,
                    llm=llm,
                    runtime=runtime,
                )
                method_evidences[method_id] = evidence
                accepts_runtime = pipeline_accepts_runtime(pipeline)
                interop_rows: list[dict[str, Any]] = []
                parsing_failures = 0
                schema_failures = 0
                t0 = time.perf_counter()
                try:
                    for scenario in scenarios:
                        usage_before = (
                            llm.usage_snapshot()
                            if isinstance(llm, UsageTrackingLLMClient)
                            else TokenUsage()
                        )
                        prediction_input = to_prediction_input(scenario, config=method_config)
                        assert_no_gold_in_prediction_input(prediction_input)
                        for forbidden in ("category", "severity", "gold", "expected", "annotation"):
                            if forbidden in prediction_input:
                                raise AssertionError(f"{forbidden} leaked into prediction input")
                        try:
                            if accepts_runtime and runtime is not None:
                                out = pipeline(
                                    prediction_input, config=method_config, llm=llm, runtime=runtime
                                )
                            else:
                                out = pipeline(prediction_input, config=method_config, llm=llm)
                        except DecisionParseError:
                            parsing_failures += 1
                            if formal:
                                raise
                            continue
                        case_usage = (
                            llm.usage_delta(usage_before)
                            if isinstance(llm, UsageTrackingLLMClient)
                            else TokenUsage()
                        )
                        ms = out.setdefault("method_specific", {})
                        runtime_block = ms.setdefault("runtime", {})
                        runtime_block.update(
                            {
                                "llm_calls": case_usage.llm_calls,
                                "token_usage": case_usage.to_dict(),
                            }
                        )
                        if ms.get("parsing_failure"):
                            parsing_failures += 1
                            if formal:
                                raise RuntimeError(
                                    f"Formal parsing failure for {method_id} case "
                                    f"{prediction_input['case_id']}: {ms.get('parsing_errors')}"
                                )
                        interop = unified_row_to_interop(out)
                        interop["case_id"] = prediction_input["case_id"]
                        interop["method_id"] = method_id
                        interop["prediction"]["final_response"]["real_world_execution_allowed"] = False
                        assert_canonical_interop_record(
                            interop,
                            dev_aliases_enabled=bool(method_config.get("dev_aliases_enabled", False)),
                        )
                        errors = validate_interop_record(
                            interop,
                            schema_path=schema_path,
                            expected_schema_sha256=schema_sha,
                        )
                        if errors:
                            schema_failures += 1
                            if formal:
                                raise RuntimeError(
                                    f"Schema failure for {method_id} case "
                                    f"{prediction_input['case_id']}: {errors}"
                                )
                        interop_rows.append(interop)
                finally:
                    close_method_runtime(runtime)

                pred_path = write_prediction_dir / f"{method_id}.jsonl"
                write_jsonl(pred_path, interop_rows)
                coverage_report = _assert_method_coverage(
                    case_ids=case_ids,
                    method_id=method_id,
                    rows=interop_rows,
                    formal=formal,
                )
                if coverage_report.get("errors"):
                    all_coverage_ok = False
                taxonomy_valid = _interop_taxonomy_valid(interop_rows)
                compliance = method_formal_compliance(
                    evidence,
                    formal=formal,
                    method_id=method_id,
                    coverage_ok=not coverage_report.get("errors"),
                    parsing_failures=parsing_failures,
                    schema_failures=schema_failures,
                    taxonomy_valid=taxonomy_valid,
                )
                method_compliance_reports[method_id] = compliance
                summary = _write_decision_artifacts(
                    write_decision_dir,
                    method_id,
                    interop_rows,
                    input_cases_sha256=input_cases_sha,
                    prediction_schema_sha256=schema_sha,
                    prediction_file=pred_path,
                    formal=formal,
                    coverage_ok=not coverage_report.get("errors"),
                    runtime_evidence=evidence,
                    method_compliance=compliance,
                    execution_stage=execution_stage,
                )
                summary["execution_contract"] = {
                    "execution_stage": execution_stage,
                    "experiment_manifest_provided": bool(experiment_manifest),
                    "heuristic_llm_used": bool(evidence.llm_is_smoke),
                    "smoke_embedding_used": bool(evidence.smoke_fallback_used),
                    "dev_aliases_enabled": bool(method_config.get("dev_aliases_enabled", False)),
                    "strict_required_fields": formal,
                    "canonical_output_only": True,
                }
                summary["parsing_failure_count"] = parsing_failures
                summary["schema_failure_count"] = schema_failures
                summary["wall_time_sec"] = round(time.perf_counter() - t0, 4)
                write_json(write_decision_dir / method_id / "run_summary.json", summary)
                suite_summary["method_summaries"][method_id] = summary
                suite_summary["coverage"][method_id] = coverage_report
                if not (summary.get("taxonomy_validation") or {}).get("valid", False):
                    suite_summary["taxonomy_contract"]["all_methods_valid"] = False
                if formal and (parsing_failures or schema_failures):
                    raise RuntimeError(
                        f"Formal run failed for {method_id}: "
                        f"parsing_failures={parsing_failures} schema_failures={schema_failures}"
                    )
        except Exception as exc:
            if formal:
                suite_summary["formal_compliance"] = compute_suite_formal_compliance(
                    formal=formal,
                    experiment_manifest_provided=experiment_manifest is not None,
                    limit_used=limit is not None,
                    preflight_ok=bool(preflight.get("ok")),
                    coverage_ok=False,
                    method_evidences=method_evidences,
                    method_compliance=method_compliance_reports,
                    dev_aliases_enabled=bool(dev_aliases_enabled and not formal),
                    shared_generation_identity_match=bool(
                        (preflight.get("shared_generation_identity") or {}).get("ok")
                    ),
                    ekell_prompt_bundle_valid=bool(preflight.get("ekell_prompt_bundle_valid")),
                    method_ids=method_ids,
                    phase="pre_publish",
                    **integrity_flags,
                )
                _raise_formal_failure(
                    message=str(exc),
                    stage="method_execution",
                    run_id=run_id,
                    marker_dir=marker_dir,
                    suite_summary=suite_summary,
                    temp_root=temp_root,
                    keep_failed_temp_artifacts=keep_failed_temp_artifacts,
                    method_id=method_id if "method_id" in locals() else None,
                    error_type=type(exc).__name__,
                )
            raise

        suite_summary["formal_compliance"] = compute_suite_formal_compliance(
            formal=formal,
            experiment_manifest_provided=experiment_manifest is not None,
            limit_used=limit is not None,
            preflight_ok=bool(preflight.get("ok")),
            coverage_ok=all_coverage_ok,
            method_evidences=method_evidences,
            method_compliance=method_compliance_reports,
            dev_aliases_enabled=bool(dev_aliases_enabled and not formal),
            shared_generation_identity_match=bool(
                (preflight.get("shared_generation_identity") or {}).get("ok")
            ),
            ekell_prompt_bundle_valid=bool(preflight.get("ekell_prompt_bundle_valid")),
            method_ids=method_ids,
            phase="pre_publish",
            **integrity_flags,
        )
        suite_summary["runtime_evidence"] = {
            mid: evidence_to_summary_sections(ev) for mid, ev in method_evidences.items()
        }

        pre_publish_passed = bool(
            suite_summary["formal_compliance"].get("pre_publish_compliance_passed")
        )

        if formal and pre_publish_passed and layout is not None and temp_root is not None:
            suite_summary_path = temp_root / "suite_summary.json"
            write_json(suite_summary_path, suite_summary)
            try:
                publish_result = publish_formal_run_root_transactionally(
                    temp_run_root=temp_root,
                    final_run_root=layout.run_root,
                )
            except FormalPublishError as exc:
                publish_result = exc.publish_result
                _raise_formal_failure(
                    message=str(exc),
                    stage="transactional_publish",
                    run_id=run_id,
                    marker_dir=marker_dir,
                    suite_summary=suite_summary,
                    temp_root=temp_root,
                    keep_failed_temp_artifacts=keep_failed_temp_artifacts,
                    error_type=type(exc).__name__,
                    rollback_attempted=publish_result.rollback_attempted,
                    rollback_succeeded=publish_result.rollback_succeeded,
                    rollback_errors=publish_result.errors,
                )

            suite_summary["transactional_publish"] = {
                "committed": publish_result.committed,
                "cleanup_complete": publish_result.cleanup_complete,
                "cleanup_warnings": list(publish_result.cleanup_warnings),
            }
            if publish_result.cleanup_warnings:
                write_json(
                    marker_dir / "FORMAL_PUBLISH_CLEANUP_WARNING.json",
                    {
                        "committed": publish_result.committed,
                        "cleanup_complete": publish_result.cleanup_complete,
                        "cleanup_warnings": publish_result.cleanup_warnings,
                    },
                )

            suite_summary["formal_compliance"] = compute_suite_formal_compliance(
                formal=formal,
                experiment_manifest_provided=experiment_manifest is not None,
                limit_used=limit is not None,
                preflight_ok=bool(preflight.get("ok")),
                coverage_ok=all_coverage_ok,
                method_evidences=method_evidences,
                method_compliance=method_compliance_reports,
                dev_aliases_enabled=bool(dev_aliases_enabled and not formal),
                shared_generation_identity_match=bool(
                    (preflight.get("shared_generation_identity") or {}).get("ok")
                ),
                ekell_prompt_bundle_valid=bool(preflight.get("ekell_prompt_bundle_valid")),
                method_ids=method_ids,
                phase="final",
                transactional_publish_committed=publish_result.committed,
                transactional_cleanup_complete=publish_result.cleanup_complete,
                **integrity_flags,
            )
            write_json(layout.run_root / "suite_summary.json", suite_summary)

            failed_marker = marker_dir / "FORMAL_RUN_FAILED.json"
            if failed_marker.is_file():
                failed_marker.unlink()

            final_compliance = suite_summary["formal_compliance"]
            if not (
                final_compliance.get("formal_result") is True
                and final_compliance.get("transactional_publish_committed") is True
                and final_compliance.get("pre_publish_compliance_passed") is True
            ):
                if publish_result.run_root_state is not None:
                    rollback_result = _rollback_run_root(publish_result.run_root_state)
                else:
                    rollback_result = FormalPublishResult(
                        success=False,
                        committed=False,
                        cleanup_complete=False,
                        cleanup_warnings=[],
                        rollback_attempted=False,
                        rollback_succeeded=False,
                        errors=[],
                    )
                _raise_formal_failure(
                    message="Final formal compliance checks did not pass after publish.",
                    stage="final_compliance",
                    run_id=run_id,
                    marker_dir=marker_dir,
                    suite_summary=suite_summary,
                    temp_root=None,
                    keep_failed_temp_artifacts=keep_failed_temp_artifacts,
                    error_type="FormalComplianceError",
                    rollback_attempted=rollback_result.rollback_attempted,
                    rollback_succeeded=rollback_result.rollback_succeeded,
                    rollback_errors=rollback_result.errors,
                )
        elif formal:
            suite_summary["formal_compliance"]["formal_result"] = False
            suite_summary["formal_compliance"]["transactional_publish_complete"] = False
            suite_summary["formal_compliance"]["transactional_publish_committed"] = False
            _raise_formal_failure(
                message="Pre-publish compliance checks did not pass.",
                stage="pre_publish_compliance",
                run_id=run_id,
                marker_dir=marker_dir,
                suite_summary=suite_summary,
                temp_root=temp_root,
                keep_failed_temp_artifacts=keep_failed_temp_artifacts,
                error_type="FormalComplianceError",
            )
        else:
            write_json(write_decision_dir / "suite_summary.json", suite_summary)
        return suite_summary


    except FormalRunFailed:
        raise
    except Exception as exc:
        if formal:
            _raise_formal_early_failure(
                exc=exc,
                run_id=run_id,
                marker_dir=marker_dir,
                temp_root=temp_root,
                keep_failed_temp_artifacts=keep_failed_temp_artifacts,
            )
        raise

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Five-method decision comparison suite")
    parser.add_argument("--runner-bundle", required=True)
    parser.add_argument("--method-set", choices=["comparison_suite"], default="comparison_suite")
    parser.add_argument("--execution-stage", choices=["dry_run", "formal"], default="dry_run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument(
        "--formal-run-root",
        default=None,
        help="Formal run root containing predictions/ and decisions/ (recommended for formal).",
    )
    parser.add_argument("--experiment-manifest", default=None)
    parser.add_argument(
        "--enable-dev-aliases",
        action="store_true",
        help="Enable development-only taxonomy aliases (dry_run only).",
    )
    parser.add_argument(
        "--keep-failed-temp-artifacts",
        action="store_true",
        help="Retain formal temporary directories when a formal run fails (debug only).",
    )
    args = parser.parse_args(argv)

    if args.execution_stage == "formal" and args.enable_dev_aliases:
        raise SystemExit("Formal execution forbids --enable-dev-aliases.")

    if args.execution_stage == "formal" and args.limit is not None:
        raise SystemExit(FORMAL_LIMIT_FORBIDDEN_MESSAGE)

    if args.method_set != "comparison_suite":
        raise SystemExit("Only comparison_suite is supported for the decision suite.")

    prediction_dir = Path(args.prediction_dir)
    decision_dir = Path(args.decision_dir)
    formal_run_root = Path(args.formal_run_root) if args.formal_run_root else None
    if args.execution_stage == "formal" and formal_run_root is not None:
        prediction_dir = formal_run_root / "predictions"
        decision_dir = formal_run_root / "decisions"

    summary = run_decision_suite(
        runner_bundle=Path(args.runner_bundle),
        prediction_dir=prediction_dir,
        decision_dir=decision_dir,
        execution_stage=args.execution_stage,
        limit=args.limit,
        experiment_manifest=Path(args.experiment_manifest) if args.experiment_manifest else None,
        dev_aliases_enabled=bool(args.enable_dev_aliases),
        keep_failed_temp_artifacts=bool(args.keep_failed_temp_artifacts),
        formal_run_root=formal_run_root,
    )
    formal = args.execution_stage == "formal"
    formal_result = bool((summary.get("formal_compliance") or {}).get("formal_result"))
    payload = {
        "ok": formal_result if formal else True,
        "execution_stage": args.execution_stage,
        "formal_result": formal_result if formal else False,
        "case_count": summary.get("case_count"),
        "methods": summary.get("methods"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if formal and not formal_result:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except FormalRunFailed as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "execution_stage": "formal",
                    "stage": exc.stage,
                    "formal_result": False,
                    "error": sanitize_error_message(str(exc)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
