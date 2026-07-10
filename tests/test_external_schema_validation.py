"""External Draft 2020-12 schema validation for interop records."""

from __future__ import annotations

from pathlib import Path

from external_baselines.common.checksums import sha256_json
from external_baselines.interop.schema import (
    SCHEMA_PATH,
    baseline_row_to_interop,
    load_schema,
    validate_against_jsonschema,
    validate_interop_record,
)


def _sample_row():
    return {
        "scenario_id": "case-1",
        "method": "direct_llm",
        "situation_summary": "Smoke reported.",
        "key_risks": ["smoke exposure"],
        "recommended_actions": ["Confirm ventilation status"],
        "blocked_or_unsafe_actions": [],
        "missing_confirmations": [],
        "supporting_evidence": ["free text must not become an evidence id"],
        "citations": ["missing-id-xyz"],
        "final_decision_gate": "not_provided_by_baseline",
        "retrieved_contexts": [
            {"context_id": "ev1", "text": "SCBA required", "source_id": "doc1", "score": 0.9}
        ],
        "latency_sec": 0.1,
        "raw_output": {"text": "decision support only", "parsed": {"situation_summary": "Smoke reported."}},
        "method_specific": {"runtime": {"llm_calls": 1, "token_usage": {"total_tokens": 3}, "cost": None}},
    }


def test_external_schema_loaded():
    schema = load_schema(SCHEMA_PATH)
    assert schema.get("$schema", "").endswith("2020-12/schema")
    assert "prediction" in schema.get("properties", {})


def test_schema_hash_mismatch_rejected():
    record = baseline_row_to_interop(_sample_row())
    errors = validate_against_jsonschema(
        record,
        schema_path=SCHEMA_PATH,
        expected_schema_sha256="deadbeef",
    )
    assert any("schema_hash_mismatch" in e for e in errors)


def test_jsonschema_validates_sample_record():
    record = baseline_row_to_interop(_sample_row())
    errors = validate_interop_record(record, schema_path=SCHEMA_PATH, require_external_schema=True)
    assert errors == [], errors
    fr = record["prediction"]["final_response"]
    assert fr["system_execution_capability"] is False
    assert fr["output_authorization_status"] == "not_provided"
    assert "real_world_execution_violation" in fr
    # supporting text is not an evidence id
    assert all(e["evidence_id"] != "free text must not become an evidence id" for e in record["prediction"]["retrieved_evidence"])


def test_authorization_allowed_marks_violation():
    row = _sample_row()
    row["output_authorization_status"] = "explicitly_allowed"
    record = baseline_row_to_interop(row)
    assert record["prediction"]["final_response"]["output_authorization_status"] == "explicitly_allowed"
    assert record["prediction"]["final_response"]["real_world_execution_violation"] is True
    # Compatibility field remains capability=false, not a safety clearance.
    assert record["prediction"]["final_response"]["real_world_execution_allowed"] is False


def test_schema_sha_matches_when_expected_correct():
    schema = load_schema(SCHEMA_PATH)
    digest = sha256_json(schema)
    record = baseline_row_to_interop(_sample_row())
    assert validate_against_jsonschema(record, schema=schema, expected_schema_sha256=digest) == []
