"""Fail-closed validation for DeepEval handoff bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from external_baselines.interop.deepeval_handoff.constants import (
    CONTRACT_ID,
    CONTRACT_SOURCE_PATH,
    DIRECT_METHOD,
    FORBIDDEN_HANDOFF_KEYS,
    FORMAL_RAG_METHODS,
    HANDOFF_MANIFEST_VERSION,
    PROVENANCE_PATH,
    SNAPSHOT_PATH,
    VALIDATION_REPORT_VERSION,
)
from external_baselines.interop.deepeval_handoff.contracts import contract_report, sha256_file
from external_baselines.interop.deepeval_handoff.manifest import case_ids_sha256, write_json
from external_baselines.interop.deepeval_handoff.schema_validation import (
    load_json_object,
    validation_errors,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"prediction_file_unreadable:{path}:{exc}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"blank_jsonl_line:{path}:{line_number}")
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid_jsonl:{path}:{line_number}:{exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"jsonl_line_must_be_object:{path}:{line_number}")
            continue
        records.append(value)
    return records, errors


def _forbidden_paths(value: Any, *, path: str = "$") -> list[str]:
    offenders: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key.lower() in FORBIDDEN_HANDOFF_KEYS:
                offenders.append(item_path)
            offenders.extend(_forbidden_paths(item, path=item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            offenders.extend(_forbidden_paths(item, path=f"{path}[{index}]"))
    return offenders


def _source_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (_repository_root() / path)


def validate_handoff(
    handoff: Path,
    *,
    main_repo: Path,
    write_report: bool = True,
) -> dict[str, Any]:
    root = _repository_root()
    errors: list[str] = []
    warnings: list[str] = []
    method_reports: dict[str, Any] = {}
    coverage_report: dict[str, Any] = {}
    manifest_path = handoff / "handoff_manifest.json"
    if not manifest_path.is_file():
        errors.append("handoff_manifest_missing")
        manifest: dict[str, Any] = {}
    else:
        try:
            manifest = load_json_object(manifest_path)
        except ValueError as exc:
            errors.append(str(exc))
            manifest = {}

    manifest_schema_path = root / "schemas/deepeval_handoff/deepeval_handoff_manifest_v1.schema.json"
    if manifest and manifest_schema_path.is_file():
        errors.extend(f"manifest_schema:{item}" for item in validation_errors(manifest, load_json_object(manifest_schema_path)))
    if manifest.get("schema_version") != HANDOFF_MANIFEST_VERSION:
        errors.append("manifest_schema_version_mismatch")
    evaluation = manifest.get("evaluation_handoff") if isinstance(manifest.get("evaluation_handoff"), dict) else {}
    for flag in (
        "gold_accessed",
        "deepeval_executed",
        "judge_called",
        "paid_api_used",
        "real_world_execution_allowed",
    ):
        if evaluation.get(flag) is not False:
            errors.append(f"safety_flag_must_be_false:{flag}")
    top_k = evaluation.get("handoff_top_k")
    if type(top_k) is not int or top_k <= 0:
        errors.append("invalid_handoff_top_k")
        top_k = 0

    contract = contract_report(repository_root=root, main_repo=main_repo)
    errors.extend(f"contract:{item}" for item in contract["errors"])
    manifest_contract = manifest.get("contract") if isinstance(manifest.get("contract"), dict) else {}
    if manifest_contract.get("contract_id") != CONTRACT_ID:
        errors.append("manifest_contract_id_mismatch")
    for manifest_key, report_key in (
        ("main_commit", "source_commit"),
        ("schema_sha256", "source_sha256"),
        ("local_snapshot_sha256", "local_snapshot_sha256"),
    ):
        if manifest_contract.get(manifest_key) != contract.get(report_key):
            errors.append(f"manifest_contract_mismatch:{manifest_key}")

    try:
        external_schema = load_json_object(main_repo / CONTRACT_SOURCE_PATH)
    except ValueError as exc:
        errors.append(str(exc))
        external_schema = {}

    methods = manifest.get("methods") if isinstance(manifest.get("methods"), dict) else {}
    case_sets: dict[str, set[str]] = {}
    for method_id, method_manifest in methods.items():
        report_errors: list[str] = []
        if not isinstance(method_manifest, dict):
            errors.append(f"method_manifest_must_be_object:{method_id}")
            continue
        rel = method_manifest.get("path")
        expected_rel = f"predictions/{method_id}.jsonl"
        if rel != expected_rel:
            report_errors.append("prediction_path_mismatch")
        prediction_path = handoff / expected_rel
        if not prediction_path.is_file():
            report_errors.append("prediction_file_missing")
            records: list[dict[str, Any]] = []
        else:
            records, load_errors = _load_jsonl(prediction_path)
            report_errors.extend(load_errors)
            if method_manifest.get("sha256") != sha256_file(prediction_path):
                report_errors.append("prediction_sha256_mismatch")
        observed: list[str] = []
        retrieval_count = 0
        for index, record in enumerate(records, start=1):
            prefix = f"record_{index}"
            if external_schema:
                report_errors.extend(f"{prefix}:schema:{item}" for item in validation_errors(record, external_schema))
            if record.get("schema_version") != CONTRACT_ID:
                report_errors.append(f"{prefix}:schema_version_mismatch")
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                report_errors.append(f"{prefix}:case_id_invalid")
            else:
                observed.append(case_id)
            if record.get("system_name") != method_id:
                report_errors.append(f"{prefix}:system_name_mismatch")
            actual_output = record.get("actual_output")
            if not isinstance(actual_output, str) or not actual_output.strip():
                report_errors.append(f"{prefix}:actual_output_invalid")
            structured = record.get("structured_prediction")
            structured = structured if isinstance(structured, dict) else {}
            if structured.get("real_world_execution_allowed") is True:
                report_errors.append(f"{prefix}:real_world_execution_allowed")
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            if metadata.get("system_execution_capability") is not False:
                report_errors.append(f"{prefix}:system_execution_capability_must_be_false")
            contexts_present = "retrieval_context" in record
            contexts = record.get("retrieval_context")
            if method_id == DIRECT_METHOD:
                if contexts_present:
                    report_errors.append(f"{prefix}:direct_retrieval_context_must_be_omitted")
                if metadata.get("comparison_level") != "output_only":
                    report_errors.append(f"{prefix}:direct_comparison_level")
            elif method_id in FORMAL_RAG_METHODS or method_manifest.get("retrieval_required") is True:
                if not isinstance(contexts, list) or not contexts:
                    report_errors.append(f"{prefix}:rag_retrieval_context_missing")
                if metadata.get("comparison_level") != "output_and_rag":
                    report_errors.append(f"{prefix}:rag_comparison_level")
            if isinstance(contexts, list):
                retrieval_count += 1
                if len(contexts) > top_k:
                    report_errors.append(f"{prefix}:retrieval_context_exceeds_top_k")
                ranks: list[int] = []
                for context_index, context in enumerate(contexts, start=1):
                    if not isinstance(context, dict):
                        report_errors.append(f"{prefix}:context_{context_index}_not_object")
                        continue
                    text = context.get("text")
                    rank = context.get("rank")
                    if not isinstance(text, str) or not text.strip():
                        report_errors.append(f"{prefix}:context_{context_index}_text_invalid")
                    if type(rank) is not int or rank <= 0:
                        report_errors.append(f"{prefix}:context_{context_index}_rank_invalid")
                    else:
                        ranks.append(rank)
                if len(ranks) != len(set(ranks)) or any(b <= a for a, b in zip(ranks, ranks[1:])):
                    report_errors.append(f"{prefix}:context_ranks_not_strictly_increasing")
                if metadata.get("native_retrieval_context_count") is not None:
                    submitted = metadata.get("submitted_retrieval_context_count")
                    native = metadata.get("native_retrieval_context_count")
                    truncated = metadata.get("retrieval_context_truncated_for_handoff")
                    if submitted != len(contexts):
                        report_errors.append(f"{prefix}:submitted_context_count_mismatch")
                    if type(native) is not int or native < len(contexts):
                        report_errors.append(f"{prefix}:native_context_count_invalid")
                    elif truncated is not (native > len(contexts)):
                        report_errors.append(f"{prefix}:context_truncation_flag_mismatch")
            forbidden = _forbidden_paths(record)
            report_errors.extend(f"{prefix}:forbidden_key:{path}" for path in forbidden)

        duplicates = sorted({item for item in observed if observed.count(item) > 1})
        if duplicates:
            report_errors.append("duplicate_case_ids:" + ",".join(duplicates))
        case_sets[method_id] = set(observed)
        if method_manifest.get("record_count") != len(records):
            report_errors.append("record_count_mismatch")
        method_reports[method_id] = {
            "ok": not report_errors,
            "errors": report_errors,
            "record_count": len(records),
            "case_count": len(set(observed)),
            "retrieval_record_count": retrieval_count,
        }
        errors.extend(f"{method_id}:{item}" for item in report_errors)

    if not methods:
        errors.append("manifest_methods_missing")
    reference_set = next(iter(case_sets.values()), set())
    mismatched = sorted(method_id for method_id, case_set in case_sets.items() if case_set != reference_set)
    if mismatched:
        errors.append("method_case_sets_differ:" + ",".join(mismatched))
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    if dataset.get("case_count") != len(reference_set):
        errors.append("dataset_case_count_mismatch")
    if reference_set and dataset.get("case_ids_sha256") != case_ids_sha256(reference_set):
        errors.append("case_ids_sha256_mismatch")
    coverage_report = {
        "case_count": len(reference_set),
        "case_ids_sha256": case_ids_sha256(reference_set),
        "identical_case_sets": not mismatched,
    }

    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    source_root = _source_path(source.get("formal_run_root"))
    if source_root is None:
        errors.append("source_formal_run_root_missing")
    else:
        for name, key in (
            ("run_manifest.json", "formal_run_manifest_sha256"),
            ("suite_summary.json", "suite_summary_sha256"),
        ):
            path = source_root / name
            if not path.is_file():
                errors.append(f"source_artifact_missing:{name}")
            elif source.get(key) != sha256_file(path):
                errors.append(f"source_artifact_sha256_mismatch:{name}")
    if (handoff / "contract_provenance.json").is_file():
        if sha256_file(handoff / "contract_provenance.json") != sha256_file(root / PROVENANCE_PATH):
            errors.append("bundle_contract_provenance_mismatch")
    else:
        errors.append("bundle_contract_provenance_missing")
    if manifest_contract.get("local_snapshot_sha256") != sha256_file(root / SNAPSHOT_PATH):
        errors.append("contract_snapshot_sha256_mismatch")

    report = {
        "schema_version": VALIDATION_REPORT_VERSION,
        "ok": not errors,
        "errors": sorted(set(errors)),
        "warnings": warnings,
        "method_reports": method_reports,
        "contract_report": contract,
        "coverage_report": coverage_report,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
    }
    if write_report:
        write_json(handoff / "validation_report.json", report)
    return report
