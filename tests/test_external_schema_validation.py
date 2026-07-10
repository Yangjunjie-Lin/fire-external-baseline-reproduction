"""External Draft 2020-12 schema validation for interop records."""

from __future__ import annotations

from external_baselines.common.checksums import sha256_file
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
        "key_risks": ["smoke_detected"],
        "recommended_actions": [{"action_id": "prepare_respiratory_protection", "text": "Prepare SCBA", "priority": "high"}],
        "blocked_or_unsafe_actions": ["BLOCK_ENTRY_WITHOUT_RESPIRATORY_PROTECTION"],
        "missing_confirmations": ["smoke_level"],
        "supporting_evidence": ["free text must not become an evidence id"],
        "citations": ["missing-id-xyz"],
        "final_decision_gate": "await_human_confirmation",
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
    assert schema.get("properties", {}).get("schema_version", {}).get("const") == "firebench-interop-v1"


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
    assert record["schema_version"] == "firebench-interop-v1"
    fr = record["prediction"]["final_response"]
    assert fr["real_world_execution_allowed"] is False
    assert fr["status"] in {
        "provided",
        "awaiting_human_confirmation",
        "blocked",
        "not_applicable",
        "unknown",
    }
    assert isinstance(record["prediction"]["blocked_actions"], list)
    assert all(isinstance(x, str) for x in record["prediction"]["blocked_actions"])
    # Extended fields live in method_metadata, not Track A prediction.
    assert "retrieved_evidence" not in record["prediction"]
    assert "raw_output" not in record["prediction"]
    assert "system_execution_capability" in record["method_metadata"]
    meta_evidence = record["method_metadata"]["retrieved_evidence"]
    assert all(e["evidence_id"] != "free text must not become an evidence id" for e in meta_evidence)


def test_authorization_allowed_marks_violation():
    row = _sample_row()
    row["output_authorization_status"] = "explicitly_allowed"
    record = baseline_row_to_interop(row)
    assert record["method_metadata"]["output_authorization_status"] == "explicitly_allowed"
    assert record["method_metadata"]["real_world_execution_violation"] is True
    assert record["prediction"]["final_response"]["real_world_execution_allowed"] is False


def test_schema_sha_matches_when_expected_correct():
    digest = sha256_file(SCHEMA_PATH)
    record = baseline_row_to_interop(_sample_row())
    assert validate_against_jsonschema(
        record, schema_path=SCHEMA_PATH, expected_schema_sha256=digest
    ) == []
