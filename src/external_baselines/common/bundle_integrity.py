"""Formal Runner Bundle integrity validation against frozen identity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file
from external_baselines.interop.bundle import validate_bundle_checksum


def extract_frozen_runner_bundle_identity(freeze: dict[str, Any]) -> dict[str, Any]:
    """Normalize frozen Runner Bundle identity fields from a freeze manifest."""
    runner_block = freeze.get("runner_bundle")
    if isinstance(runner_block, dict):
        return {
            "bundle_checksum": runner_block.get("bundle_checksum") or freeze.get("runner_bundle_checksum"),
            "input_cases_sha256": runner_block.get("input_cases_sha256"),
            "prediction_schema_sha256": runner_block.get("prediction_schema_sha256")
            or freeze.get("prediction_schema_checksum"),
            "corpus_aggregate_sha256": runner_block.get("corpus_aggregate_sha256")
            or freeze.get("corpus_checksum"),
        }
    return {
        "bundle_checksum": freeze.get("runner_bundle_checksum"),
        "input_cases_sha256": freeze.get("input_cases_sha256"),
        "prediction_schema_sha256": freeze.get("prediction_schema_checksum"),
        "corpus_aggregate_sha256": freeze.get("corpus_checksum"),
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
    expected_bundle = str(frozen.get("bundle_checksum") or "").strip()
    expected_input = str(frozen.get("input_cases_sha256") or "").strip()
    expected_schema = str(frozen.get("prediction_schema_sha256") or "").strip()
    expected_corpus = str(frozen.get("corpus_aggregate_sha256") or "").strip()

    if not any((expected_bundle, expected_input, expected_schema, expected_corpus)):
        errors.append("frozen_bundle_identity_missing")

    bundle_validation = validate_bundle_checksum(
        bundle,
        expected=expected_bundle or None,
    )

    if not file_report.get("ok", True):
        errors.append("runner_bundle_file_checksum_mismatch")
        for item in file_report.get("mismatches") or []:
            mismatches.append({"type": "file_checksum", "detail": item})

    if expected_bundle:
        declared = str(bundle_validation.get("producer_declared_checksum") or "")
        if declared and declared != expected_bundle:
            errors.append("runner_bundle_checksum_mismatch")
            mismatches.append(
                {
                    "field": "bundle_checksum",
                    "expected": expected_bundle,
                    "actual": declared,
                }
            )
        elif not declared and str(bundle_validation.get("recomputed") or "") != expected_bundle:
            errors.append("runner_bundle_checksum_mismatch")

    if expected_input:
        actual_input = live.get("input_cases_sha256")
        if not actual_input or str(actual_input) != expected_input:
            errors.append("input_cases_checksum_mismatch")
            mismatches.append(
                {"field": "input_cases_sha256", "expected": expected_input, "actual": actual_input}
            )

    if expected_schema:
        actual_schema = live.get("prediction_schema_sha256")
        if not actual_schema or str(actual_schema) != expected_schema:
            errors.append("prediction_schema_checksum_mismatch")
            mismatches.append(
                {"field": "prediction_schema_sha256", "expected": expected_schema, "actual": actual_schema}
            )

    if expected_corpus:
        actual_corpus = live.get("corpus_aggregate_sha256")
        if not actual_corpus or str(actual_corpus) != expected_corpus:
            errors.append("corpus_checksum_mismatch")
            mismatches.append(
                {"field": "corpus_aggregate_sha256", "expected": expected_corpus, "actual": actual_corpus}
            )

    ok = not errors and bool(bundle_validation.get("ok", False) or file_report.get("ok", True))
    if frozen and not errors and not file_report.get("ok", True):
        ok = False

    return {
        "ok": ok,
        "producer_declared_checksum": live.get("producer_declared_checksum"),
        "consumer_computed_hash": live.get("consumer_computed_hash"),
        "expected_frozen_checksum": expected_bundle or None,
        "input_cases_sha256": live.get("input_cases_sha256"),
        "expected_input_cases_sha256": expected_input or None,
        "prediction_schema_sha256": live.get("prediction_schema_sha256"),
        "expected_prediction_schema_sha256": expected_schema or None,
        "corpus_aggregate_sha256": live.get("corpus_aggregate_sha256"),
        "expected_corpus_aggregate_sha256": expected_corpus or None,
        "file_checksum_report_ok": bool(file_report.get("ok", True)),
        "mismatches": mismatches,
        "errors": errors,
        "bundle_checksum_validation": bundle_validation,
    }
