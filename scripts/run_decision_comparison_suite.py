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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.checksums import sha256_file  # noqa: E402
from external_baselines.common.decision_output import (  # noqa: E402
    DecisionParseError,
    unified_row_to_interop,
)
from external_baselines.common.decision_suite_guard import (  # noqa: E402
    FORMAL_LIMIT_FORBIDDEN_MESSAGE,
    CLIValidationError,
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
    read_json,
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
    RuntimeCleanupError,
    close_method_runtime_safely,
    pipeline_accepts_runtime,
    prepare_method_runtime,
    runtime_cache_scope,
    runtime_is_cached,
)
from external_baselines.common.runtime_evidence import (  # noqa: E402
    collect_method_runtime_evidence,
    compute_suite_formal_compliance,
    evidence_to_summary_sections,
    method_formal_compliance,
)
from external_baselines.common.safe_paths import (  # noqa: E402
    ManifestArtifactPathError,
    resolve_manifest_artifact_path,
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

EMBEDDING_METHODS = frozenset(
    {
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    }
)

DECISION_SUPPLEMENTAL_ARTIFACTS = (
    "decisions.jsonl",
    "responses.jsonl",
    "unmapped_taxonomy.jsonl",
)


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
    control_root: Path
    failure_marker_path: Path
    cleanup_warning_path: Path


def resolve_formal_control_root(run_root: Path) -> Path:
    return run_root.parent / f".{run_root.name}.control"


def resolve_formal_control_root_from_paths(
    *,
    formal_run_root: Path | None,
    prediction_dir: Path,
    decision_dir: Path,
) -> Path | None:
    try:
        layout = resolve_formal_run_layout(
            formal_run_root=formal_run_root,
            prediction_dir=prediction_dir,
            decision_dir=decision_dir,
        )
    except FormalSuiteExecutionError:
        return None
    return layout.control_root


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
    control_root = resolve_formal_control_root(root)
    return FormalRunLayout(
        run_root=root,
        prediction_dir=root / "predictions",
        decision_dir=root / "decisions",
        control_root=control_root,
        failure_marker_path=control_root / "FORMAL_RUN_FAILED.json",
        cleanup_warning_path=control_root / "FORMAL_PUBLISH_CLEANUP_WARNING.json",
    )


@dataclass(frozen=True)
class ResolvedOutputPaths:
    formal_run_root: Path | None
    prediction_dir: Path
    decision_dir: Path


def resolve_cli_output_paths(args: argparse.Namespace) -> ResolvedOutputPaths:
    pred_arg = Path(args.prediction_dir).expanduser() if args.prediction_dir else None
    dec_arg = Path(args.decision_dir).expanduser() if args.decision_dir else None
    root_arg = Path(args.formal_run_root).expanduser() if args.formal_run_root else None

    if args.execution_stage == "formal":
        if root_arg is not None:
            expected_pred = root_arg / "predictions"
            expected_dec = root_arg / "decisions"
            if pred_arg is not None and pred_arg.resolve() != expected_pred.resolve():
                raise FormalRunFailed(
                    "formal_output_paths_conflict_with_formal_run_root",
                    stage="cli_validation",
                )
            if dec_arg is not None and dec_arg.resolve() != expected_dec.resolve():
                raise FormalRunFailed(
                    "formal_output_paths_conflict_with_formal_run_root",
                    stage="cli_validation",
                )
            return ResolvedOutputPaths(root_arg, expected_pred, expected_dec)
        if pred_arg is None or dec_arg is None:
            raise FormalRunFailed("formal_output_paths_incomplete", stage="cli_validation")
        try:
            layout = resolve_formal_run_layout(
                formal_run_root=None,
                prediction_dir=pred_arg,
                decision_dir=dec_arg,
            )
        except FormalSuiteExecutionError as exc:
            raise FormalRunFailed(str(exc.args[0]), stage="cli_validation") from exc
        return ResolvedOutputPaths(None, layout.prediction_dir, layout.decision_dir)

    if pred_arg is None or dec_arg is None:
        raise CLIValidationError("dry_run_requires_prediction_and_decision_dirs")
    return ResolvedOutputPaths(None, pred_arg, dec_arg)


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
class FormalPublishResult:
    success: bool
    committed: bool
    cleanup_complete: bool
    cleanup_warnings: list[str]
    rollback_attempted: bool
    rollback_succeeded: bool
    run_root_state: RunRootPublishState | None = None
    errors: list[str] = field(default_factory=list)


class FormalPublishError(RuntimeError):
    """Raised when transactional formal publish fails after rollback is attempted."""

    def __init__(self, message: str, *, publish_result: FormalPublishResult) -> None:
        super().__init__(message)
        self.publish_result = publish_result


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


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


def _remove_directory_backup(backup: Path | None) -> None:
    if backup is None or not backup.exists():
        return
    _remove_path(backup)


def publish_formal_run_root_transactionally(
    *,
    temp_run_root: Path,
    final_run_root: Path,
) -> FormalPublishResult:
    """Publish a formal run root with PREPARE / COMMIT / CLEANUP phases."""
    if temp_run_root.resolve() == final_run_root.resolve():
        raise FormalSuiteExecutionError("formal_temp_and_final_run_root_must_differ")
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


_SHA256_HEX_RE = __import__("re").compile(r"^[a-f0-9]{64}$")


def _is_valid_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_HEX_RE.match(value))


def emit_post_commit_warning(
    *,
    warning_code: str,
    message: str,
    path: Path | None = None,
) -> dict[str, Any]:
    warning = {
        "code": warning_code,
        "message": sanitize_error_message(message),
        "path": str(path) if path else None,
    }
    print(json.dumps({"formal_post_commit_warning": warning}), file=sys.stderr)
    return warning


def load_and_validate_frozen_prediction_schema(
    schema_path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not schema_path or not expected_sha256:
        errors.append("staged_prediction_schema_missing")
        return None, errors
    if not schema_path.is_file():
        errors.append("staged_prediction_schema_missing")
        return None, errors
    try:
        raw_text = schema_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append("staged_prediction_schema_invalid_json")
        return None, errors
    except OSError:
        errors.append("staged_prediction_schema_missing")
        return None, errors
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        errors.append("staged_prediction_schema_invalid_json")
        return None, errors
    if not isinstance(payload, dict):
        errors.append("staged_prediction_schema_invalid_json")
        return None, errors
    observed_sha = sha256_file(schema_path)
    if observed_sha != expected_sha256:
        errors.append("staged_prediction_schema_sha256_mismatch")
    if errors:
        return None, errors

    from external_baselines.interop.schema import validate_schema_draft202012

    draft_errors = validate_schema_draft202012(
        payload,
        primary_schema_name=schema_path.name,
    )
    for code in draft_errors:
        if code == "external_schema_invalid_draft202012":
            errors.append("staged_prediction_schema_invalid_draft202012")
        elif code == "external_schema_reference_unresolvable":
            errors.append("staged_prediction_schema_invalid_draft202012")
        elif code == "staged_prediction_schema_unsupported_draft":
            errors.append(code)
        elif code.startswith("jsonschema_unavailable"):
            errors.append("jsonschema_unavailable")
        else:
            errors.append(code)
    if errors:
        return None, errors
    return payload, errors


def _manifest_artifact_path_errors(run_root: Path, rel_path: str) -> list[str]:
    try:
        resolve_manifest_artifact_path(run_root, rel_path)
    except ManifestArtifactPathError as exc:
        return [str(exc.args[0])]
    return []


def _validate_staged_prediction_file(
    pred_file: Path,
    *,
    method_id: str,
    expected_case_id_set: set[str],
    prediction_schema: dict[str, Any],
    primary_schema_name: str,
) -> list[str]:
    from external_baselines.common.taxonomy_normalizer import assert_canonical_interop_record

    errors: list[str] = []
    if not pred_file.is_file():
        return [f"missing_prediction_{method_id}"]
    try:
        lines = pred_file.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [f"prediction_not_utf8_{method_id}"]
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            errors.append(f"invalid_prediction_json_{method_id}:line_{line_no}")
            continue
        if not isinstance(record, dict):
            errors.append(f"invalid_prediction_object_{method_id}:line_{line_no}")
            continue
        records.append(record)
        schema_errors = validate_interop_record(
            record,
            schema=prediction_schema,
            require_external_schema=True,
            primary_schema_name=primary_schema_name,
        )
        if schema_errors:
            errors.append(f"invalid_interop_record_{method_id}:line_{line_no}")
        if str(record.get("schema_version") or "") != "firebench-interop-v1":
            errors.append(f"wrong_schema_version_{method_id}:line_{line_no}")
        if str(record.get("method_id") or "") != method_id:
            errors.append(f"wrong_method_id_{method_id}:line_{line_no}")
        case_id = str(record.get("case_id") or "").strip()
        if not case_id:
            errors.append(f"missing_case_id_{method_id}:line_{line_no}")
        pred = record.get("prediction")
        if not isinstance(pred, dict):
            errors.append(f"missing_prediction_object_{method_id}:line_{line_no}")
        elif not isinstance(pred.get("final_response"), dict):
            errors.append(f"missing_final_response_{method_id}:line_{line_no}")
        try:
            assert_canonical_interop_record(record, dev_aliases_enabled=False)
        except Exception:  # noqa: BLE001
            errors.append(f"invalid_canonical_interop_{method_id}:line_{line_no}")
    if len(records) != len(expected_case_id_set):
        errors.append(f"prediction_count_mismatch_{method_id}")
    observed = [str(r.get("case_id") or "") for r in records]
    if len(observed) != len(set(observed)):
        errors.append(f"duplicate_case_id_{method_id}")
    if set(observed) != expected_case_id_set:
        errors.append(f"case_id_set_mismatch_{method_id}")
    return errors


def _validate_staged_supplemental_jsonl(
    artifact_path: Path,
    *,
    method_id: str,
    expected_case_id_set: set[str],
    artifact_name: str,
) -> list[str]:
    errors: list[str] = []
    if not artifact_path.is_file():
        return [f"missing_supplemental_{artifact_name}_{method_id}"]
    try:
        lines = artifact_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [f"supplemental_not_utf8_{artifact_name}_{method_id}"]
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            errors.append(f"invalid_{artifact_name}_json_{method_id}:line_{line_no}")
            continue
        if not isinstance(record, dict):
            errors.append(f"invalid_{artifact_name}_object_{method_id}:line_{line_no}")
            continue
        records.append(record)
        if str(record.get("method_id") or "") != method_id:
            errors.append(f"wrong_method_in_{artifact_name}_{method_id}:line_{line_no}")
    if len(records) != len(expected_case_id_set):
        errors.append(f"{artifact_name}_count_mismatch_{method_id}")
    observed = [str(r.get("case_id") or "") for r in records]
    if len(observed) != len(set(observed)):
        errors.append(f"duplicate_case_id_in_{artifact_name}_{method_id}")
    if set(observed) != expected_case_id_set:
        errors.append(f"case_id_set_mismatch_in_{artifact_name}_{method_id}")
    return errors


def _validate_staged_unmapped_taxonomy(
    artifact_path: Path,
    *,
    method_id: str,
) -> list[str]:
    errors: list[str] = []
    if not artifact_path.is_file():
        return [f"missing_unmapped_taxonomy_{method_id}"]
    try:
        lines = artifact_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [f"unmapped_taxonomy_not_utf8_{method_id}"]
    record_count = 0
    for line_no, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            json.loads(text)
        except json.JSONDecodeError:
            errors.append(f"invalid_unmapped_taxonomy_json_{method_id}:line_{line_no}")
            continue
        record_count += 1
    if record_count != 0:
        errors.append(f"nonempty_unmapped_taxonomy_{method_id}")
    return errors


def validate_staged_formal_run_root(
    temp_run_root: Path,
    *,
    method_ids: list[str],
    expected_case_ids: list[str] | set[str],
    prediction_schema_path: Path,
    expected_prediction_schema_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_case_id_set = set(expected_case_ids)
    if not expected_case_id_set:
        errors.append("expected_case_ids_empty")
    if len(expected_case_id_set) != len(list(expected_case_ids)):
        errors.append("expected_case_ids_duplicate")
    schema_payload, schema_errors = load_and_validate_frozen_prediction_schema(
        prediction_schema_path,
        expected_prediction_schema_sha256,
    )
    errors.extend(schema_errors)
    schema_ready = schema_payload is not None and not schema_errors

    predictions = temp_run_root / "predictions"
    decisions = temp_run_root / "decisions"
    suite_summary_path = temp_run_root / "suite_summary.json"
    run_manifest_path = temp_run_root / "run_manifest.json"
    preflight_path = temp_run_root / "diagnostics" / "decision_suite_preflight.json"

    for label, path in (
        ("predictions", predictions),
        ("decisions", decisions),
        ("suite_summary.json", suite_summary_path),
        ("run_manifest.json", run_manifest_path),
        ("diagnostics/decision_suite_preflight.json", preflight_path),
    ):
        if not path.exists():
            errors.append(f"missing_{label}")

    for method_id in method_ids:
        pred_file = predictions / f"{method_id}.jsonl"
        summary_file = decisions / method_id / "run_summary.json"
        if schema_ready and schema_payload is not None:
            errors.extend(
                _validate_staged_prediction_file(
                    pred_file,
                    method_id=method_id,
                    expected_case_id_set=expected_case_id_set,
                    prediction_schema=schema_payload,
                    primary_schema_name=prediction_schema_path.name,
                )
            )
        method_dir = decisions / method_id
        errors.extend(
            _validate_staged_supplemental_jsonl(
                method_dir / "decisions.jsonl",
                method_id=method_id,
                expected_case_id_set=expected_case_id_set,
                artifact_name="decisions",
            )
        )
        errors.extend(
            _validate_staged_supplemental_jsonl(
                method_dir / "responses.jsonl",
                method_id=method_id,
                expected_case_id_set=expected_case_id_set,
                artifact_name="responses",
            )
        )
        errors.extend(
            _validate_staged_unmapped_taxonomy(
                method_dir / "unmapped_taxonomy.jsonl",
                method_id=method_id,
            )
        )
        if not summary_file.is_file():
            errors.append(f"missing_run_summary_{method_id}")
            continue
        try:
            method_summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append(f"invalid_method_summary_json_{method_id}")
            continue
        if str(method_summary.get("method_id") or "") != method_id:
            errors.append(f"method_summary_method_id_mismatch_{method_id}")
        if method_summary.get("execution_stage") != "formal":
            errors.append(f"method_summary_execution_stage_{method_id}")
        if method_summary.get("case_count") != len(expected_case_id_set):
            errors.append(f"method_summary_case_count_{method_id}")
        if method_summary.get("successful_count") != len(expected_case_id_set):
            errors.append(f"method_summary_successful_count_{method_id}")
        if int(method_summary.get("parsing_failure_count") or 0) != 0:
            errors.append(f"method_summary_parsing_failures_{method_id}")
        if int(method_summary.get("schema_failure_count") or 0) != 0:
            errors.append(f"method_summary_schema_failures_{method_id}")
        if method_summary.get("formal_result") is not True:
            errors.append(f"method_summary_formal_result_{method_id}")
        method_compliance = method_summary.get("formal_compliance") or {}
        if method_compliance.get("formal_result") is not True:
            errors.append(f"method_summary_compliance_formal_false_{method_id}")
        taxonomy = method_summary.get("taxonomy_validation") or {}
        if taxonomy.get("valid") is not True:
            errors.append(f"method_summary_taxonomy_invalid_{method_id}")
        pred_sha = method_summary.get("prediction_file_sha256")
        if not _is_valid_sha256_hex(pred_sha):
            errors.append(f"method_summary_prediction_sha_invalid_{method_id}")
        elif pred_file.is_file() and pred_sha != sha256_file(pred_file):
            errors.append(f"method_summary_prediction_sha_mismatch_{method_id}")

    if suite_summary_path.is_file():
        try:
            summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("invalid_suite_summary_json")
            summary = {}
        if summary.get("execution_stage") != "formal":
            errors.append("suite_summary_execution_stage")
        if summary.get("formal") is not True:
            errors.append("suite_summary_formal_flag")
        if summary.get("case_count") != len(expected_case_id_set):
            errors.append("suite_summary_case_count")
        if sorted(summary.get("methods") or []) != sorted(method_ids):
            errors.append("suite_summary_methods_mismatch")
        compliance = summary.get("formal_compliance") or {}
        if compliance.get("formal_result") is not True:
            errors.append("staged_formal_result_not_true")
        if compliance.get("transactional_publish_committed") is not True:
            errors.append("staged_publish_not_committed")
        if compliance.get("pre_publish_compliance_passed") is not True:
            errors.append("staged_pre_publish_not_passed")
        if compliance.get("transactional_cleanup_complete") is not None:
            errors.append("suite_summary_cleanup_must_be_null")
        transactional = summary.get("transactional_publish") or {}
        if transactional.get("cleanup_complete") is not None:
            errors.append("suite_summary_transactional_cleanup_must_be_null")
        coverage = summary.get("coverage") or {}
        for method_id in method_ids:
            report = coverage.get(method_id) or {}
            if report.get("errors"):
                errors.append(f"coverage_errors_{method_id}")
            if report.get("expected_case_count") != len(expected_case_id_set):
                errors.append(f"coverage_expected_count_{method_id}")
            if report.get("prediction_count") != len(expected_case_id_set):
                errors.append(f"coverage_actual_count_{method_id}")
            if report.get("missing_case_ids"):
                errors.append(f"coverage_missing_cases_{method_id}")
            if report.get("extra_case_ids"):
                errors.append(f"coverage_extra_cases_{method_id}")
            if report.get("duplicate_case_ids"):
                errors.append(f"coverage_duplicate_cases_{method_id}")

    preflight = {}
    if preflight_path.is_file():
        try:
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("invalid_preflight_json")
            preflight = {}
    preflight_integrity = (
        preflight.get("runner_bundle_integrity") if isinstance(preflight, dict) else {}
    )
    if not isinstance(preflight_integrity, dict):
        preflight_integrity = {}

    if run_manifest_path.is_file():
        try:
            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("invalid_run_manifest_json")
            manifest = {}
        if manifest.get("execution_stage") != "formal":
            errors.append("manifest_execution_stage")
        if manifest.get("formal_result") is not True:
            errors.append("manifest_formal_result")
        if sorted(manifest.get("method_ids") or []) != sorted(method_ids):
            errors.append("manifest_method_set_mismatch")
        if manifest.get("case_count") != len(expected_case_id_set):
            errors.append("manifest_case_count")
        for field, path in (
            ("suite_summary_sha256", suite_summary_path),
            ("preflight_sha256", preflight_path),
        ):
            recorded = manifest.get(field)
            if not _is_valid_sha256_hex(recorded):
                errors.append(f"missing_or_invalid_{field}")
            elif path.is_file() and recorded != sha256_file(path):
                errors.append(f"{field}_mismatch")
        prediction_files = manifest.get("prediction_files") or {}
        decision_summary_files = manifest.get("decision_summary_files") or {}
        for method_id in method_ids:
            pred_file = predictions / f"{method_id}.jsonl"
            summary_file = decisions / method_id / "run_summary.json"
            pred_hash = prediction_files.get(method_id)
            if not _is_valid_sha256_hex(pred_hash):
                errors.append(f"prediction_sha256_missing_{method_id}")
            elif pred_file.is_file() and pred_hash != sha256_file(pred_file):
                errors.append(f"prediction_sha256_mismatch_{method_id}")
            dec_hash = decision_summary_files.get(method_id)
            if not _is_valid_sha256_hex(dec_hash):
                errors.append(f"decision_summary_sha256_missing_{method_id}")
            elif summary_file.is_file() and dec_hash != sha256_file(summary_file):
                errors.append(f"decision_summary_sha256_mismatch_{method_id}")
        artifact_hashes = manifest.get("artifact_hashes") or {}
        for rel_path, recorded_hash in artifact_hashes.items():
            rel_text = str(rel_path or "")
            path_errors = _manifest_artifact_path_errors(temp_run_root, rel_text)
            errors.extend(path_errors)
            if path_errors:
                continue
            try:
                artifact_path = resolve_manifest_artifact_path(temp_run_root, rel_text)
            except ManifestArtifactPathError:
                continue
            if not _is_valid_sha256_hex(recorded_hash):
                errors.append(f"artifact_sha256_invalid:{rel_text}")
            elif not artifact_path.is_file():
                errors.append(f"missing_supplemental_artifact:{rel_text}")
            elif sha256_file(artifact_path) != recorded_hash:
                errors.append(f"artifact_sha256_mismatch:{rel_text}")
        inventory = manifest.get("artifact_inventory") or {}
        for rel_path in (
            *(inventory.get("core_files") or []),
            *(inventory.get("supplemental_files") or []),
        ):
            errors.extend(_manifest_artifact_path_errors(temp_run_root, str(rel_path or "")))
        decision_artifact_files = manifest.get("decision_artifact_files") or {}
        for method_id in method_ids:
            method_artifacts = decision_artifact_files.get(method_id) or {}
            for artifact_name in (*DECISION_SUPPLEMENTAL_ARTIFACTS, "run_summary.json"):
                rel = f"decisions/{method_id}/{artifact_name}"
                recorded = method_artifacts.get(artifact_name)
                artifact_path = decisions / method_id / artifact_name
                if not _is_valid_sha256_hex(recorded):
                    errors.append(f"decision_artifact_sha256_missing_{method_id}:{artifact_name}")
                elif artifact_path.is_file() and recorded != sha256_file(artifact_path):
                    errors.append(f"decision_artifact_sha256_mismatch_{method_id}:{artifact_name}")
                elif artifact_name in DECISION_SUPPLEMENTAL_ARTIFACTS and not artifact_path.is_file():
                    errors.append(f"missing_supplemental_artifact:{rel}")

        provenance_checks = {
            "input_cases_provenance": (
                "input_cases",
                (
                    "input_cases_path",
                    "input_cases_relpath",
                    "input_cases_source",
                    "input_cases_inside_bundle",
                    "input_cases_declared_sha256",
                    "input_cases_sha256",
                    "input_cases_checksum_valid",
                    "input_cases_checksum_match",
                    "input_cases_authoritative",
                    "input_cases_formal_eligible",
                ),
            ),
            "prediction_schema_provenance": (
                "prediction_schema",
                (
                    "prediction_schema_path",
                    "prediction_schema_source",
                    "prediction_schema_inside_bundle",
                    "prediction_schema_declared_sha256",
                    "prediction_schema_sha256",
                    "prediction_schema_checksum_match",
                    "prediction_schema_authoritative",
                    "prediction_schema_formal_eligible",
                ),
            ),
        }
        for manifest_field, (error_prefix, keys) in provenance_checks.items():
            recorded = manifest.get(manifest_field)
            if not isinstance(recorded, dict):
                errors.append(f"manifest_{error_prefix}_provenance_missing")
                continue
            for key in keys:
                if recorded.get(key) != preflight_integrity.get(key):
                    errors.append(f"manifest_{key}_mismatch")

    if errors:
        raise FormalSuiteExecutionError(
            "staged_formal_run_root_invalid: " + ", ".join(errors)
        )
    return {
        "ok": True,
        "method_count": len(method_ids),
        "case_count": len(expected_case_id_set),
    }


def build_formal_run_manifest(
    *,
    run_id: str,
    method_ids: list[str],
    case_count: int,
    runner_bundle: Path,
    experiment_manifest: Path | None,
    temp_run_root: Path,
    suite_summary: dict[str, Any],
    preflight_path: Path,
    formal_result: bool,
) -> dict[str, Any]:
    predictions_dir = temp_run_root / "predictions"
    decisions_dir = temp_run_root / "decisions"
    suite_summary_path = temp_run_root / "suite_summary.json"
    prediction_files = {
        mid: sha256_file(predictions_dir / f"{mid}.jsonl")
        for mid in method_ids
        if (predictions_dir / f"{mid}.jsonl").is_file()
    }
    decision_summary_files = {
        mid: sha256_file(decisions_dir / mid / "run_summary.json")
        for mid in method_ids
        if (decisions_dir / mid / "run_summary.json").is_file()
    }
    schema_provenance = {}
    input_cases_provenance = {}
    preflight = read_json(preflight_path, default={}) if preflight_path.is_file() else {}
    if isinstance(preflight, dict):
        integrity = preflight.get("runner_bundle_integrity") or {}
        if isinstance(integrity, dict):
            schema_provenance = {
                key: integrity.get(key)
                for key in (
                    "prediction_schema_path",
                    "prediction_schema_source",
                    "prediction_schema_inside_bundle",
                    "prediction_schema_declared_sha256",
                    "prediction_schema_sha256",
                    "prediction_schema_checksum_match",
                    "prediction_schema_authoritative",
                    "prediction_schema_formal_eligible",
                )
            }
            input_cases_provenance = {
                key: integrity.get(key)
                for key in (
                    "input_cases_path",
                    "input_cases_relpath",
                    "input_cases_source",
                    "input_cases_inside_bundle",
                    "input_cases_declared_sha256",
                    "input_cases_sha256",
                    "input_cases_checksum_valid",
                    "input_cases_checksum_match",
                    "input_cases_authoritative",
                    "input_cases_formal_eligible",
                )
            }
    artifact_hashes: dict[str, str] = {}
    decision_artifact_files: dict[str, dict[str, str]] = {}
    supplemental_files: list[str] = []
    for mid in method_ids:
        pred_rel = f"predictions/{mid}.jsonl"
        pred_path = predictions_dir / f"{mid}.jsonl"
        if pred_path.is_file():
            artifact_hashes[pred_rel] = sha256_file(pred_path)
        method_artifacts: dict[str, str] = {}
        for artifact_name in (*DECISION_SUPPLEMENTAL_ARTIFACTS, "run_summary.json"):
            rel = f"decisions/{mid}/{artifact_name}"
            artifact_path = decisions_dir / mid / artifact_name
            if artifact_path.is_file():
                digest = sha256_file(artifact_path)
                artifact_hashes[rel] = digest
                method_artifacts[artifact_name] = digest
                if artifact_name in DECISION_SUPPLEMENTAL_ARTIFACTS:
                    supplemental_files.append(rel)
        if method_artifacts:
            decision_artifact_files[mid] = method_artifacts
    core_files = [
        "suite_summary.json",
        "run_manifest.json",
        "diagnostics/decision_suite_preflight.json",
        *[f"predictions/{mid}.jsonl" for mid in method_ids],
        *[f"decisions/{mid}/run_summary.json" for mid in method_ids],
    ]
    return {
        "run_id": run_id,
        "execution_stage": "formal",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method_ids": method_ids,
        "case_count": case_count,
        "runner_bundle": str(runner_bundle),
        "experiment_manifest": str(experiment_manifest) if experiment_manifest else None,
        "suite_summary_sha256": sha256_file(suite_summary_path) if suite_summary_path.is_file() else "",
        "preflight_sha256": sha256_file(preflight_path) if preflight_path.is_file() else "",
        "prediction_files": prediction_files,
        "decision_summary_files": decision_summary_files,
        "decision_artifact_files": decision_artifact_files,
        "artifact_hashes": artifact_hashes,
        "input_cases_provenance": input_cases_provenance,
        "prediction_schema_provenance": schema_provenance,
        "formal_result": formal_result,
        "artifact_inventory": {
            "core_files": core_files,
            "supplemental_files": supplemental_files,
        },
        "suite_summary": {
            "formal_compliance": suite_summary.get("formal_compliance"),
            "transactional_publish": suite_summary.get("transactional_publish"),
        },
    }


def _write_publish_receipt(
    *,
    control_root: Path,
    run_id: str,
    layout: FormalRunLayout,
    publish_result: FormalPublishResult,
    formal_result: bool,
) -> None:
    run_record = ensure_dir(control_root / "runs" / run_id)
    backup_path = (
        publish_result.run_root_state.backup_path
        if publish_result.run_root_state is not None
        else None
    )
    suite_summary_path = layout.run_root / "suite_summary.json"
    run_manifest_path = layout.run_root / "run_manifest.json"
    write_json(
        run_record / "publish_receipt.json",
        {
            "run_id": run_id,
            "committed": publish_result.committed,
            "formal_result": formal_result,
            "formal_run_root": str(layout.run_root),
            "backup_path": str(backup_path) if backup_path else None,
            "cleanup_complete": publish_result.cleanup_complete,
            "cleanup_warnings": list(publish_result.cleanup_warnings),
            "formal_run_manifest_sha256": sha256_file(run_manifest_path)
            if run_manifest_path.is_file()
            else None,
            "formal_suite_summary_sha256": sha256_file(suite_summary_path)
            if suite_summary_path.is_file()
            else None,
        },
    )


def _write_cleanup_warning(
    *,
    layout: FormalRunLayout,
    publish_result: FormalPublishResult,
) -> None:
    if not publish_result.cleanup_warnings:
        return
    write_json(
        layout.cleanup_warning_path,
        {
            "committed": publish_result.committed,
            "cleanup_complete": publish_result.cleanup_complete,
            "cleanup_warnings": publish_result.cleanup_warnings,
        },
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


def _write_formal_failed_marker(
    *,
    control_root: Path,
    run_id: str,
    stage: str,
    method_id: str | None,
    error_type: str,
    error_message: str,
    temporary_artifacts_path: str | None,
    rollback_attempted: bool = False,
    rollback_succeeded: bool = False,
    rollback_errors: list[str] | None = None,
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
        "failure_summary_path": failure_summary_path,
    }
    write_json(control_root / "FORMAL_RUN_FAILED.json", marker)


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
    control_root: Path,
    suite_summary: dict[str, Any],
    temp_root: Path | None,
    keep_failed_temp_artifacts: bool,
    method_id: str | None = None,
    error_type: str = "FormalRunFailed",
    rollback_attempted: bool = False,
    rollback_succeeded: bool = False,
    rollback_errors: list[str] | None = None,
) -> None:
    run_record_dir = ensure_dir(control_root / "runs" / run_id)
    failure_summary_path = write_formal_failure_diagnostics(
        diagnostics_dir=run_record_dir,
        run_id=run_id,
        suite_summary=suite_summary,
    )
    _write_formal_failed_marker(
        control_root=control_root,
        run_id=run_id,
        stage=stage,
        method_id=method_id,
        error_type=error_type,
        error_message=message,
        temporary_artifacts_path=str(temp_root) if temp_root and keep_failed_temp_artifacts else None,
        rollback_attempted=rollback_attempted,
        rollback_succeeded=rollback_succeeded,
        rollback_errors=rollback_errors,
        failure_summary_path=str(failure_summary_path.relative_to(control_root)),
    )
    if temp_root and not keep_failed_temp_artifacts:
        _cleanup_formal_temp(temp_root)
    raise FormalRunFailed(message, stage=stage, summary=suite_summary)


def _raise_formal_early_failure(
    *,
    exc: Exception,
    run_id: str,
    control_root: Path,
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
        control_root=control_root,
        suite_summary=suite_summary,
        temp_root=temp_root,
        keep_failed_temp_artifacts=keep_failed_temp_artifacts,
        error_type=type(exc).__name__,
    )


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.execution_stage != "formal":
        if args.method_set != "comparison_suite":
            raise SystemExit("Only comparison_suite is supported for the decision suite.")
        return
    if args.enable_dev_aliases:
        raise FormalRunFailed(
            "Formal execution forbids --enable-dev-aliases.",
            stage="cli_validation",
        )
    if args.limit is not None:
        raise FormalRunFailed(FORMAL_LIMIT_FORBIDDEN_MESSAGE, stage="cli_validation")
    if args.method_set != "comparison_suite":
        raise FormalRunFailed(
            "Only comparison_suite is supported for the decision suite.",
            stage="cli_validation",
        )


def _emit_cli_error(
    *,
    execution_stage: str,
    stage: str,
    error: str,
) -> None:
    print(
        json.dumps(
            {
                "ok": False,
                "execution_stage": execution_stage,
                "stage": stage,
                "formal_result": False,
                "error": sanitize_error_message(error),
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)


def _emit_formal_cli_error(
    exc: FormalRunFailed,
    *,
    control_root: Path | None,
    run_id: str,
) -> None:
    if control_root is not None:
        try:
            _write_formal_failed_marker(
                control_root=control_root,
                run_id=run_id,
                stage=exc.stage,
                method_id=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                temporary_artifacts_path=None,
            )
        except Exception:  # noqa: BLE001
            pass
    print(
        json.dumps(
            {
                "ok": False,
                "execution_stage": "formal",
                "stage": exc.stage,
                "formal_result": False,
                "error_type": type(exc).__name__,
                "error": sanitize_error_message(str(exc)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


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


LLMTransportFactory = Callable[[str, dict[str, Any]], Callable[..., str] | None]
EmbeddingBackendFactory = Callable[[str, dict[str, Any]], Any | None]


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
    llm_transport_factory: LLMTransportFactory | None = None,
    embedding_backend_factory: EmbeddingBackendFactory | None = None,
) -> dict[str, Any]:
    formal = execution_stage == "formal"
    run_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    temp_root: Path | None = None
    write_prediction_dir = prediction_dir
    write_decision_dir = decision_dir
    layout: FormalRunLayout | None = None
    control_root = resolve_formal_control_root_from_paths(
        formal_run_root=formal_run_root,
        prediction_dir=prediction_dir,
        decision_dir=decision_dir,
    ) or (decision_dir.parent.parent / ".formal.control")
    return _run_decision_suite_impl(
        formal=formal,
        run_id=run_id,
        temp_root=temp_root,
        write_prediction_dir=write_prediction_dir,
        write_decision_dir=write_decision_dir,
        layout=layout,
        control_root=control_root,
        runner_bundle=runner_bundle,
        prediction_dir=prediction_dir,
        decision_dir=decision_dir,
        execution_stage=execution_stage,
        limit=limit,
        experiment_manifest=experiment_manifest,
        methods=methods,
        dev_aliases_enabled=dev_aliases_enabled,
        keep_failed_temp_artifacts=keep_failed_temp_artifacts,
        formal_run_root=formal_run_root,
        llm_transport_factory=llm_transport_factory,
        embedding_backend_factory=embedding_backend_factory,
    )


def _run_decision_suite_impl(
    *,
    formal: bool,
    run_id: str,
    temp_root: Path | None,
    write_prediction_dir: Path,
    write_decision_dir: Path,
    layout: FormalRunLayout | None,
    control_root: Path,
    runner_bundle: Path,
    prediction_dir: Path,
    decision_dir: Path,
    execution_stage: str,
    limit: int | None,
    experiment_manifest: Path | None,
    methods: list[str] | None,
    dev_aliases_enabled: bool,
    keep_failed_temp_artifacts: bool,
    formal_run_root: Path | None,
    llm_transport_factory: LLMTransportFactory | None,
    embedding_backend_factory: EmbeddingBackendFactory | None,
) -> dict[str, Any]:
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
            control_root = layout.control_root

        validate_decision_suite_execution(
            execution_stage=execution_stage,
            experiment_manifest=experiment_manifest,
            method_ids=method_ids,
            runner_bundle=runner_bundle,
            limit=limit,
        )

        load_limit = None if formal else limit
        coverage = inspect_runner_bundle_case_coverage(runner_bundle, limit=load_limit, formal=formal)
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

        bundle = load_runner_bundle(runner_bundle, formal=formal)
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
        if formal:
            run_record_dir = ensure_dir(control_root / "runs" / run_id)
            write_json(run_record_dir / "preflight.json", preflight)
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
                control_root=control_root,
                suite_summary=suite_summary,
                temp_root=temp_root,
                keep_failed_temp_artifacts=keep_failed_temp_artifacts,
                error_type="PreflightError",
            )

        if formal and layout is not None:
            temp_root = create_formal_temp_run_root(layout.run_root, run_id)
            write_prediction_dir = ensure_dir(temp_root / "predictions")
            write_decision_dir = ensure_dir(temp_root / "decisions")
            staged_preflight = ensure_dir(temp_root / "diagnostics")
            write_json(staged_preflight / "decision_suite_preflight.json", preflight)

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
            with runtime_cache_scope():
                for method_id in method_ids:
                    method_config = method_configs[method_id]
                    transport = (
                        llm_transport_factory(method_id, method_config)
                        if llm_transport_factory is not None
                        else None
                    )
                    injected_embedding = (
                        embedding_backend_factory(method_id, method_config)
                        if embedding_backend_factory is not None and method_id in EMBEDDING_METHODS
                        else None
                    )
                    llm = build_llm_client(method_config, transport=transport)
                    pipeline = resolve_pipeline(method_id)
                    runtime = None
                    body_exception: BaseException | None = None
                    interop_rows: list[dict[str, Any]] = []
                    parsing_failures = 0
                    schema_failures = 0
                    t0 = time.perf_counter()
                    try:
                        runtime = prepare_method_runtime(
                            method_id,
                            method_config,
                            embedding_backend=injected_embedding,
                        )
                        evidence = collect_method_runtime_evidence(
                            method_id=method_id,
                            config=method_config,
                            llm=llm,
                            runtime=runtime,
                        )
                        method_evidences[method_id] = evidence
                        accepts_runtime = pipeline_accepts_runtime(pipeline)
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
                    except BaseException as exc:
                        body_exception = exc
                        raise
                    finally:
                        if runtime is not None and not runtime_is_cached(runtime):
                            close_method_runtime_safely(
                                runtime,
                                body_exception=body_exception,
                            )

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
        except RuntimeCleanupError as exc:
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
                    stage="runtime_cleanup",
                    run_id=run_id,
                    control_root=control_root,
                    suite_summary=suite_summary,
                    temp_root=temp_root,
                    keep_failed_temp_artifacts=keep_failed_temp_artifacts,
                    method_id=None,
                    error_type=type(exc).__name__,
                )
            raise
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
                    control_root=control_root,
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
            staged_preflight_path = temp_root / "diagnostics" / "decision_suite_preflight.json"
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
                phase="planned_final",
                transactional_publish_committed=True,
                transactional_cleanup_complete=None,
                **integrity_flags,
            )
            suite_summary["transactional_publish"] = {
                "committed": True,
                "cleanup_status": "reported_externally",
                "cleanup_complete": None,
                "cleanup_warnings": None,
            }
            write_json(temp_root / "suite_summary.json", suite_summary)
            run_manifest = build_formal_run_manifest(
                run_id=run_id,
                method_ids=method_ids,
                case_count=len(case_ids),
                runner_bundle=runner_bundle,
                experiment_manifest=experiment_manifest,
                temp_run_root=temp_root,
                suite_summary=suite_summary,
                preflight_path=staged_preflight_path,
                formal_result=True,
            )
            write_json(temp_root / "run_manifest.json", run_manifest)
            validate_staged_formal_run_root(
                temp_root,
                method_ids=method_ids,
                expected_case_ids=case_ids,
                prediction_schema_path=schema_path,
                expected_prediction_schema_sha256=schema_sha or "",
            )
            post_commit_warnings: list[dict[str, Any]] = []
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
                    control_root=control_root,
                    suite_summary=suite_summary,
                    temp_root=temp_root,
                    keep_failed_temp_artifacts=keep_failed_temp_artifacts,
                    error_type=type(exc).__name__,
                    rollback_attempted=publish_result.rollback_attempted,
                    rollback_succeeded=publish_result.rollback_succeeded,
                    rollback_errors=publish_result.errors,
                )

            for warning_message in publish_result.cleanup_warnings:
                post_commit_warnings.append(
                    emit_post_commit_warning(
                        warning_code="backup_cleanup_failed",
                        message=warning_message,
                    )
                )
            suite_summary["transactional_publish_runtime"] = {
                "committed": publish_result.committed,
                "cleanup_complete": publish_result.cleanup_complete,
                "cleanup_warnings": list(publish_result.cleanup_warnings),
            }
            if publish_result.cleanup_warnings:
                try:
                    _write_cleanup_warning(layout=layout, publish_result=publish_result)
                except Exception as exc:  # noqa: BLE001
                    post_commit_warnings.append(
                        emit_post_commit_warning(
                            warning_code="cleanup_warning_record_write_failed",
                            message=str(exc),
                            path=layout.cleanup_warning_path,
                        )
                    )
            try:
                _write_publish_receipt(
                    control_root=control_root,
                    run_id=run_id,
                    layout=layout,
                    publish_result=publish_result,
                    formal_result=True,
                )
            except Exception as exc:  # noqa: BLE001
                post_commit_warnings.append(
                    emit_post_commit_warning(
                        warning_code="publish_receipt_write_failed",
                        message=str(exc),
                        path=control_root / "runs" / run_id / "publish_receipt.json",
                    )
                )
            try:
                if layout.failure_marker_path.is_file():
                    layout.failure_marker_path.unlink()
            except Exception as exc:  # noqa: BLE001
                post_commit_warnings.append(
                    emit_post_commit_warning(
                        warning_code="stale_failure_marker_remove_failed",
                        message=str(exc),
                        path=layout.failure_marker_path,
                    )
                )
            if post_commit_warnings:
                suite_summary["post_commit_warnings"] = post_commit_warnings
        elif formal:
            suite_summary["formal_compliance"]["formal_result"] = False
            suite_summary["formal_compliance"]["transactional_publish_complete"] = False
            suite_summary["formal_compliance"]["transactional_publish_committed"] = False
            _raise_formal_failure(
                message="Pre-publish compliance checks did not pass.",
                stage="pre_publish_compliance",
                run_id=run_id,
                control_root=control_root,
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
                control_root=control_root,
                temp_root=temp_root,
                keep_failed_temp_artifacts=keep_failed_temp_artifacts,
            )
        raise

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Five-method decision comparison suite")
    parser.add_argument("--runner-bundle", required=True)
    parser.add_argument("--method-set", choices=["comparison_suite"], default="comparison_suite")
    parser.add_argument("--execution-stage", choices=["dry_run", "formal"], default="dry_run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--decision-dir", default=None)
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
    return parser


RECOMMENDED_FORMAL_ARGS = (
    "--runner-bundle",
    "bundle",
    "--method-set",
    "comparison_suite",
    "--execution-stage",
    "formal",
    "--formal-run-root",
    "outputs/formal/test_public",
    "--experiment-manifest",
    "configs/experiments/controlled_main_table_v1.yaml",
)


def main(argv: list[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]

    try:
        validate_cli_args(args)
        output_paths = resolve_cli_output_paths(args)
    except FormalRunFailed as exc:
        _emit_formal_cli_error(exc, control_root=None, run_id=run_id)
    except CLIValidationError as exc:
        _emit_cli_error(
            execution_stage=args.execution_stage,
            stage=exc.stage,
            error=str(exc),
        )

    prediction_dir = output_paths.prediction_dir
    decision_dir = output_paths.decision_dir
    formal_run_root = output_paths.formal_run_root
    control_root = resolve_formal_control_root_from_paths(
        formal_run_root=formal_run_root,
        prediction_dir=prediction_dir,
        decision_dir=decision_dir,
    )

    try:
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
    except FormalRunFailed as exc:
        _emit_formal_cli_error(exc, control_root=control_root, run_id=run_id)
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
    main()
