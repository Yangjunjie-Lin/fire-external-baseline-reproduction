"""Transactional exporter for completed formal prediction artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from external_baselines.interop.deepeval_handoff.adapter import (
    HandoffAdaptationError,
    adapt_firebench_interop_to_external_prediction,
)
from external_baselines.interop.deepeval_handoff.constants import (
    CONTRACT_ID,
    CONTRACT_SOURCE_PATH,
    FORMAL_RAG_METHODS,
    PROVENANCE_PATH,
)
from external_baselines.interop.deepeval_handoff.contracts import contract_report
from external_baselines.interop.deepeval_handoff.manifest import (
    build_manifest,
    repository_identity,
    sha256_file,
    write_json,
    write_jsonl,
)
from external_baselines.interop.deepeval_handoff.schema_validation import load_json_object, validate_or_raise
from external_baselines.interop.deepeval_handoff.validator import validate_handoff
from external_baselines.method_registry import comparison_suite_methods


class HandoffExportError(RuntimeError):
    """Raised when no complete handoff can be published."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise HandoffExportError(f"prediction_file_unreadable:{path}:{exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise HandoffExportError(f"blank_jsonl_line:{path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HandoffExportError(f"invalid_jsonl:{path}:{line_number}:{exc}") from exc
        if not isinstance(value, dict):
            raise HandoffExportError(f"jsonl_record_must_be_object:{path}:{line_number}")
        records.append(value)
    return records


def _suite_formal_result(summary: dict[str, Any]) -> bool:
    compliance = summary.get("formal_compliance") if isinstance(summary.get("formal_compliance"), dict) else {}
    return summary.get("formal") is True and compliance.get("formal_result") is True


def _transaction_committed(summary: dict[str, Any]) -> bool:
    compliance = summary.get("formal_compliance") if isinstance(summary.get("formal_compliance"), dict) else {}
    transactional = summary.get("transactional_publish") if isinstance(summary.get("transactional_publish"), dict) else {}
    return compliance.get("transactional_publish_committed") is True and transactional.get("committed", True) is True


def _formal_source_errors(
    *,
    run_root: Path,
    run_manifest: dict[str, Any],
    suite_summary: dict[str, Any],
    method_ids: list[str],
) -> list[str]:
    errors: list[str] = []
    if run_manifest.get("execution_stage") != "formal":
        errors.append("run_manifest_execution_stage_not_formal")
    if run_manifest.get("formal_result") is not True:
        errors.append("run_manifest_formal_result_false")
    if suite_summary.get("execution_stage") != "formal":
        errors.append("suite_summary_execution_stage_not_formal")
    if not _suite_formal_result(suite_summary):
        errors.append("suite_summary_formal_result_false")
    if not _transaction_committed(suite_summary):
        errors.append("transactional_publish_not_committed")
    if sorted(method_ids) != sorted(comparison_suite_methods()):
        errors.append("formal_method_set_differs_from_registry_authority")
    if sorted(run_manifest.get("method_ids") or []) != sorted(method_ids):
        errors.append("run_manifest_method_set_mismatch")
    source_identity = run_manifest.get("source_repository_identity")
    if not isinstance(source_identity, dict) or not source_identity.get("git_commit"):
        errors.append("formal_source_worktree_identity_missing")
    prediction_hashes = run_manifest.get("prediction_files") or {}
    decision_hashes = run_manifest.get("decision_summary_files") or {}
    for method_id in method_ids:
        prediction_path = run_root / "predictions" / f"{method_id}.jsonl"
        summary_path = run_root / "decisions" / method_id / "run_summary.json"
        if not prediction_path.is_file():
            errors.append(f"prediction_file_missing:{method_id}")
        elif prediction_hashes.get(method_id) != sha256_file(prediction_path):
            errors.append(f"prediction_sha256_mismatch:{method_id}")
        if not summary_path.is_file():
            errors.append(f"decision_summary_missing:{method_id}")
        elif decision_hashes.get(method_id) != sha256_file(summary_path):
            errors.append(f"decision_summary_sha256_mismatch:{method_id}")
    input_provenance = run_manifest.get("input_cases_provenance") or {}
    if not input_provenance.get("input_cases_sha256"):
        errors.append("input_cases_identity_missing")
    if not (run_manifest.get("runner_bundle_sha256") or run_manifest.get("runner_bundle_identity")):
        errors.append("runner_bundle_identity_missing")
    return errors


def _source_artifacts(
    *,
    run_root: Path,
    run_manifest: dict[str, Any],
    method_id: str,
) -> dict[str, Any]:
    prediction_path = run_root / "predictions" / f"{method_id}.jsonl"
    input_provenance = run_manifest.get("input_cases_provenance") or {}
    runner_identity = run_manifest.get("runner_bundle_identity")
    runner_sha = run_manifest.get("runner_bundle_sha256")
    if not runner_sha and isinstance(runner_identity, dict):
        runner_sha = runner_identity.get("sha256") or runner_identity.get("aggregate_sha256")
    configs = run_manifest.get("runtime_config_sha256")
    config_sha = configs.get(method_id) if isinstance(configs, dict) else configs
    return {
        "prediction_path": f"predictions/{method_id}.jsonl",
        "prediction_sha256": sha256_file(prediction_path),
        "resource_bundle_sha256": runner_sha,
        "runtime_config_sha256": config_sha,
        "dataset_sha256": input_provenance.get("input_cases_sha256"),
    }


def _atomic_publish(staged: Path, output: Path, *, replace_existing: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace_existing:
        raise HandoffExportError(f"output_exists:{output}; pass --replace-existing")
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    moved_existing = False
    try:
        if output.exists():
            os.replace(output, backup)
            moved_existing = True
        os.replace(staged, output)
        if moved_existing:
            shutil.rmtree(backup)
    except Exception:
        if not output.exists() and moved_existing and backup.exists():
            os.replace(backup, output)
        raise


def _write_failure_diagnostic(run_root: Path, payload: dict[str, Any]) -> Path:
    path = run_root.parent / "deepeval_handoff_diagnostics" / f"{run_root.name}_export_failure.json"
    write_json(path, payload)
    return path


def export_handoff(
    *,
    formal_run_root: Path,
    main_repo: Path,
    output: Path,
    top_k: int = 5,
    methods: Iterable[str] | None = None,
    allow_development_source: bool = False,
    replace_existing: bool = False,
) -> dict[str, Any]:
    repository_root = _repository_root()
    run_root = formal_run_root.resolve()
    output = output.resolve()
    temp_root: Path | None = None
    try:
        if type(top_k) is not int or top_k <= 0:
            raise HandoffExportError("top_k_must_be_positive_integer")
        run_manifest_path = run_root / "run_manifest.json"
        suite_summary_path = run_root / "suite_summary.json"
        run_manifest = load_json_object(run_manifest_path)
        suite_summary = load_json_object(suite_summary_path)
        declared = run_manifest.get("method_ids") or suite_summary.get("methods")
        if not isinstance(declared, list) or not declared or not all(isinstance(item, str) and item for item in declared):
            raise HandoffExportError("source_method_set_missing_or_invalid")
        method_ids = list(methods) if methods is not None else list(declared)
        if not method_ids or len(method_ids) != len(set(method_ids)):
            raise HandoffExportError("requested_method_set_missing_or_duplicate")
        unknown = sorted(set(method_ids) - set(declared))
        if unknown:
            raise HandoffExportError("requested_methods_not_in_source_manifest:" + ",".join(unknown))

        formal_errors = _formal_source_errors(
            run_root=run_root,
            run_manifest=run_manifest,
            suite_summary=suite_summary,
            method_ids=list(declared),
        )
        if method_ids != list(declared):
            formal_errors.append("requested_method_set_not_complete")
        if repository_identity(repository_root).get("worktree_clean") is not True:
            formal_errors.append("handoff_source_worktree_not_clean")
        formal_source = not formal_errors
        if not formal_source and not allow_development_source:
            raise HandoffExportError("formal_source_validation_failed:" + ";".join(formal_errors))

        contract = contract_report(repository_root=repository_root, main_repo=main_repo.resolve())
        if not contract["ok"]:
            raise HandoffExportError("contract_validation_failed:" + ";".join(contract["errors"]))
        external_schema = load_json_object(main_repo.resolve() / CONTRACT_SOURCE_PATH)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp_", dir=str(output.parent)))
        case_sets: dict[str, set[str]] = {}
        method_manifests: dict[str, Any] = {}
        for method_id in method_ids:
            source_path = run_root / "predictions" / f"{method_id}.jsonl"
            records = _read_jsonl(source_path)
            artifacts = _source_artifacts(run_root=run_root, run_manifest=run_manifest, method_id=method_id)
            adapted: list[dict[str, Any]] = []
            for record in records:
                try:
                    prediction = adapt_firebench_interop_to_external_prediction(
                        record,
                        top_k=top_k,
                        source_artifacts=artifacts,
                        formal=formal_source,
                    )
                except HandoffAdaptationError as exc:
                    raise HandoffExportError(f"adaptation_failed:{method_id}:{exc}") from exc
                validate_or_raise(prediction, external_schema, subject=f"external_prediction:{method_id}")
                adapted.append(prediction)
            case_ids = [item["case_id"] for item in adapted]
            if len(case_ids) != len(set(case_ids)):
                raise HandoffExportError(f"duplicate_case_ids:{method_id}")
            case_sets[method_id] = set(case_ids)
            target = temp_root / "predictions" / f"{method_id}.jsonl"
            write_jsonl(target, adapted)
            retrieval_records = sum("retrieval_context" in item for item in adapted)
            retrieval_required = method_id in FORMAL_RAG_METHODS or any(
                (item.get("metadata") or {}).get("retrieval_required") is True for item in adapted
            )
            method_manifests[method_id] = {
                "path": f"predictions/{method_id}.jsonl",
                "sha256": sha256_file(target),
                "record_count": len(adapted),
                "comparison_level": "output_and_rag" if retrieval_required else "output_only",
                "retrieval_required": retrieval_required,
                "retrieval_coverage": (retrieval_records / len(adapted)) if retrieval_required and adapted else None,
            }

        reference = next(iter(case_sets.values()), set())
        if any(case_set != reference for case_set in case_sets.values()):
            raise HandoffExportError("method_case_sets_differ")
        expected_count = run_manifest.get("case_count")
        if expected_count is not None and expected_count != len(reference):
            raise HandoffExportError("source_case_count_mismatch")
        shutil.copyfile(repository_root / PROVENANCE_PATH, temp_root / "contract_provenance.json")
        (temp_root / "README.md").write_text(
            "# External Baseline DeepEval Handoff v1\n\n"
            "This bundle contains completed external baseline predictions only.\n"
            "It does not contain benchmark Gold.\n"
            "It has not been scored by DeepEval.\n"
            "It must be evaluated by fire-agent-demo under the frozen evaluator protocol.\n",
            encoding="utf-8",
        )
        input_provenance = run_manifest.get("input_cases_provenance") or {}
        runner_identity = run_manifest.get("runner_bundle_identity")
        runner_sha = run_manifest.get("runner_bundle_sha256")
        if not runner_sha and isinstance(runner_identity, dict):
            runner_sha = runner_identity.get("sha256") or runner_identity.get("aggregate_sha256")
        manifest = build_manifest(
            repository_root=repository_root,
            formal_run_root=run_root,
            source_manifest_sha256=sha256_file(run_manifest_path),
            suite_summary_sha256=sha256_file(suite_summary_path),
            formal_source=formal_source,
            transactional_publish_complete=_transaction_committed(suite_summary),
            contract={
                "contract_id": CONTRACT_ID,
                "main_repository": "Yangjunjie-Lin/fire-agent-demo",
                "main_commit": contract["source_commit"],
                "source_schema_path": CONTRACT_SOURCE_PATH.as_posix(),
                "schema_sha256": contract["source_sha256"],
                "local_snapshot_sha256": contract["local_snapshot_sha256"],
                "snapshot_match": True,
            },
            split=run_root.name,
            case_ids=reference,
            input_cases_sha256=input_provenance.get("input_cases_sha256"),
            runner_bundle_sha256=runner_sha,
            top_k=top_k,
            methods=method_manifests,
        )
        write_json(temp_root / "handoff_manifest.json", manifest)
        report = validate_handoff(temp_root, main_repo=main_repo.resolve(), write_report=True)
        if not report["ok"]:
            raise HandoffExportError("staged_handoff_validation_failed:" + ";".join(report["errors"]))
        _atomic_publish(temp_root, output, replace_existing=replace_existing)
        temp_root = None
        return {
            "ok": True,
            "source_run": str(run_root),
            "formal_source": formal_source,
            "publication_eligible": formal_source,
            "development_artifact": not formal_source,
            "formal_source_errors": formal_errors,
            "method_count": len(method_ids),
            "case_count": len(reference),
            "handoff_top_k": top_k,
            "contract_sha256": contract["source_sha256"],
            "target_directory": str(output),
        }
    except Exception as exc:
        diagnostic = _write_failure_diagnostic(
            run_root,
            {"ok": False, "error": str(exc), "source_run": str(run_root), "target_directory": str(output)},
        )
        if isinstance(exc, HandoffExportError):
            raise HandoffExportError(f"{exc}; diagnostic={diagnostic}") from exc
        raise HandoffExportError(f"handoff_export_failed:{exc}; diagnostic={diagnostic}") from exc
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
