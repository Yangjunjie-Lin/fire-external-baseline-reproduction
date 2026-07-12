"""Formal Runner Bundle integrity validation against frozen identity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file
from external_baselines.interop.bundle import validate_bundle_checksum


def extract_frozen_runner_bundle_identity(
    freeze: dict[str, Any],
    *,
    formal: bool = False,
) -> dict[str, Any]:
    """Normalize frozen Runner Bundle identity fields from a freeze manifest."""
    runner_block = freeze.get("runner_bundle") if isinstance(freeze.get("runner_bundle"), dict) else {}
    legacy_bundle = runner_block.get("bundle_checksum") or freeze.get("runner_bundle_checksum")
    producer = runner_block.get("producer_declared_checksum")
    consumer = runner_block.get("consumer_computed_hash")
    if formal and legacy_bundle and producer in (None, "") and consumer in (None, ""):
        return {
            "legacy_ambiguous_bundle_checksum": str(legacy_bundle),
            "input_cases_sha256": runner_block.get("input_cases_sha256") or freeze.get("input_cases_sha256"),
            "prediction_schema_sha256": runner_block.get("prediction_schema_sha256")
            or freeze.get("prediction_schema_checksum"),
            "corpus_aggregate_sha256": runner_block.get("corpus_aggregate_sha256")
            or freeze.get("corpus_checksum"),
        }
    if not producer and not consumer and legacy_bundle:
        producer = None
        consumer = legacy_bundle
    return {
        "producer_declared_checksum": producer,
        "consumer_computed_hash": consumer,
        "producer_checksum_available": runner_block.get("producer_checksum_available"),
        "input_cases_sha256": runner_block.get("input_cases_sha256") or freeze.get("input_cases_sha256"),
        "prediction_schema_sha256": runner_block.get("prediction_schema_sha256")
        or freeze.get("prediction_schema_checksum"),
        "corpus_aggregate_sha256": runner_block.get("corpus_aggregate_sha256")
        or freeze.get("corpus_checksum"),
    }


def compute_runner_bundle_identity(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute live Runner Bundle identity checksums from loaded bundle metadata."""
    scenarios_path = bundle.get("scenarios_path")
    schema_path = bundle.get("prediction_schema_path")
    corpus_manifest = bundle.get("corpus_manifest") if isinstance(bundle.get("corpus_manifest"), dict) else {}
    return {
        "producer_declared_checksum": bundle.get("producer_declared_checksum") or bundle.get("bundle_checksum"),
        "consumer_computed_hash": bundle.get("consumer_computed_bundle_hash")
        or bundle.get("recomputed_bundle_checksum"),
        "input_cases_sha256": sha256_file(scenarios_path) if scenarios_path and Path(scenarios_path).is_file() else None,
        "prediction_schema_sha256": bundle.get("prediction_schema_sha256")
        or (sha256_file(schema_path) if schema_path and Path(schema_path).is_file() else None),
        "corpus_aggregate_sha256": corpus_manifest.get("aggregate_sha256"),
    }


def validate_formal_runner_bundle_integrity(
    bundle: dict[str, Any],
    *,
    frozen_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate current Runner Bundle against frozen per-file identity expectations."""
    live = compute_runner_bundle_identity(bundle)
    file_report = bundle.get("file_checksum_report") or {}
    errors: list[str] = []
    mismatches: list[dict[str, Any]] = []

    frozen = dict(frozen_identity or {})
    if frozen.get("legacy_ambiguous_bundle_checksum"):
        errors.append("legacy_ambiguous_bundle_checksum_not_allowed")

    expected_producer = frozen.get("producer_declared_checksum")
    expected_consumer = str(frozen.get("consumer_computed_hash") or "").strip()
    expected_input = str(frozen.get("input_cases_sha256") or "").strip()
    expected_schema = str(frozen.get("prediction_schema_sha256") or "").strip()
    expected_corpus = str(frozen.get("corpus_aggregate_sha256") or "").strip()

    live_producer = live.get("producer_declared_checksum")
    live_consumer = str(live.get("consumer_computed_hash") or "").strip()

    frozen_identity_complete = all((expected_consumer, expected_input, expected_schema, expected_corpus))
    if not frozen_identity_complete:
        errors.append("frozen_bundle_identity_missing")

    producer_required = expected_producer not in (None, "")
    producer_checksum_match = True
    if producer_required:
        if not live_producer or str(live_producer) != str(expected_producer):
            producer_checksum_match = False
            errors.append("producer_declared_checksum_mismatch")
            mismatches.append(
                {
                    "field": "producer_declared_checksum",
                    "expected": expected_producer,
                    "actual": live_producer,
                }
            )
    elif live_producer not in (None, ""):
        producer_checksum_match = True

    consumer_hash_match = bool(expected_consumer and live_consumer and expected_consumer == live_consumer)
    if expected_consumer and not consumer_hash_match:
        errors.append("consumer_computed_hash_mismatch")
        mismatches.append(
            {
                "field": "consumer_computed_hash",
                "expected": expected_consumer,
                "actual": live_consumer or None,
            }
        )

    bundle_validation = validate_bundle_checksum(bundle)
    file_report_checked = bool(file_report.get("checked"))
    file_report_ok = file_report.get("ok") is True if file_report_checked else None

    if file_report_checked and file_report_ok is not True:
        errors.append("runner_bundle_file_checksum_mismatch")
        for item in file_report.get("mismatches") or []:
            mismatches.append({"type": "file_checksum", "detail": item})

    input_cases_integrity = True
    if expected_input:
        actual_input = live.get("input_cases_sha256")
        if not actual_input or str(actual_input) != expected_input:
            input_cases_integrity = False
            errors.append("input_cases_checksum_mismatch")
            mismatches.append(
                {"field": "input_cases_sha256", "expected": expected_input, "actual": actual_input}
            )

    prediction_schema_integrity = True
    if expected_schema:
        actual_schema = live.get("prediction_schema_sha256")
        if not actual_schema or str(actual_schema) != expected_schema:
            prediction_schema_integrity = False
            errors.append("prediction_schema_checksum_mismatch")
            mismatches.append(
                {"field": "prediction_schema_sha256", "expected": expected_schema, "actual": actual_schema}
            )

    corpus_integrity = True
    if expected_corpus:
        actual_corpus = live.get("corpus_aggregate_sha256")
        if not actual_corpus or str(actual_corpus) != expected_corpus:
            corpus_integrity = False
            errors.append("corpus_checksum_mismatch")
            mismatches.append(
                {"field": "corpus_aggregate_sha256", "expected": expected_corpus, "actual": actual_corpus}
            )

    if file_report_checked:
        integrity_evidence_ok = (
            file_report_ok is True
            and consumer_hash_match
            and (producer_checksum_match if producer_required else True)
        )
    elif frozen_identity_complete:
        integrity_evidence_ok = (
            consumer_hash_match
            and (producer_checksum_match if producer_required else True)
            and input_cases_integrity
            and prediction_schema_integrity
            and corpus_integrity
        )
    else:
        errors.append("runner_bundle_integrity_evidence_incomplete")
        integrity_evidence_ok = False

    ok = (
        not errors
        and integrity_evidence_ok
        and consumer_hash_match
        and (producer_checksum_match if producer_required else True)
        and input_cases_integrity
        and prediction_schema_integrity
        and corpus_integrity
    )

    return {
        "ok": ok,
        "producer_declared_checksum": live_producer,
        "expected_producer_declared_checksum": expected_producer,
        "producer_checksum_match": producer_checksum_match,
        "consumer_computed_hash": live_consumer or None,
        "expected_consumer_computed_hash": expected_consumer or None,
        "consumer_hash_match": consumer_hash_match,
        "input_cases_integrity": input_cases_integrity,
        "prediction_schema_integrity": prediction_schema_integrity,
        "corpus_integrity": corpus_integrity,
        "per_file_checksums_checked": file_report_checked,
        "per_file_checksums_ok": file_report_ok,
        "input_cases_sha256": live.get("input_cases_sha256"),
        "expected_input_cases_sha256": expected_input or None,
        "prediction_schema_sha256": live.get("prediction_schema_sha256"),
        "expected_prediction_schema_sha256": expected_schema or None,
        "corpus_aggregate_sha256": live.get("corpus_aggregate_sha256"),
        "expected_corpus_aggregate_sha256": expected_corpus or None,
        "file_checksum_report_ok": file_report_ok,
        "mismatches": mismatches,
        "errors": errors,
        "bundle_checksum_validation": bundle_validation,
    }
