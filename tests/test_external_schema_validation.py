"""External Draft 2020-12 schema validation for interop records."""

from __future__ import annotations

import socket
import urllib.request

from external_baselines.common.checksums import sha256_file
from external_baselines.interop.schema import (
    SCHEMA_DRAFT_PATH,
    SCHEMA_PATH,
    baseline_row_to_interop,
    build_no_network_schema_registry,
    load_schema,
    validate_against_jsonschema,
    validate_interop_record,
    validate_schema_draft202012,
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


def _schema_with_ref(ref: str, *, keyword: str = "$ref") -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        keyword: ref,
    }


def _primary_schema_with_ref(ref: str, *, schema_id: str = "urn:test:primary") -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        "$defs": {"record": {"type": "object"}},
        "$ref": ref,
    }


def test_schema_allows_internal_fragment_ref():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"record": {"type": "object"}},
        "$ref": "#/$defs/record",
    }
    assert validate_schema_draft202012(schema) == []
    assert validate_against_jsonschema({}, schema=schema) == []


def test_no_network_registry_registers_primary_schema_id():
    schema = _primary_schema_with_ref("urn:test:primary#/$defs/record")
    assert validate_schema_draft202012(schema, primary_schema_name="bundle_prediction.json") == []
    assert validate_against_jsonschema({}, schema=schema, primary_schema_name="bundle_prediction.json") == []


def test_no_network_registry_registers_primary_schema_name():
    schema = _primary_schema_with_ref("bundle_prediction.json#/$defs/record")
    assert validate_schema_draft202012(schema, primary_schema_name="bundle_prediction.json") == []
    assert validate_against_jsonschema({}, schema=schema, primary_schema_name="bundle_prediction.json") == []


def test_no_network_registry_does_not_register_repo_local_schema_name():
    schema = _primary_schema_with_ref(f"{SCHEMA_PATH.name}#/$defs/record")
    errors = validate_schema_draft202012(schema, primary_schema_name="bundle_prediction.json")
    assert any("external_schema_reference_not_registered" in err for err in errors)


def test_no_network_registry_does_not_register_repo_local_schema_id():
    local_id = load_schema(SCHEMA_DRAFT_PATH).get("$id")
    schema = _primary_schema_with_ref(f"{local_id}#/$defs/record", schema_id="urn:test:primary")
    errors = validate_schema_draft202012(schema, primary_schema_name="bundle_prediction.json")
    assert any("external_schema_reference_not_registered" in err for err in errors)


def test_frozen_schema_cannot_reference_repo_local_snapshot_filename():
    schema = _primary_schema_with_ref(SCHEMA_PATH.name)
    errors = validate_against_jsonschema({}, schema=schema, primary_schema_name="bundle_prediction.json")
    assert any("external_schema_reference_not_registered" in err for err in errors)


def test_frozen_schema_cannot_reference_repo_local_snapshot_id():
    local_id = load_schema(SCHEMA_DRAFT_PATH).get("$id")
    schema = _primary_schema_with_ref(str(local_id), schema_id="urn:test:primary")
    errors = validate_against_jsonschema({}, schema=schema, primary_schema_name="bundle_prediction.json")
    assert any("external_schema_reference_not_registered" in err for err in errors)


def test_primary_schema_can_reference_its_own_filename():
    schema = _primary_schema_with_ref(f"{SCHEMA_PATH.name}#/$defs/record")
    assert validate_schema_draft202012(schema, primary_schema_name=SCHEMA_PATH.name) == []
    assert validate_against_jsonschema({}, schema=schema, primary_schema_name=SCHEMA_PATH.name) == []


def test_primary_schema_can_reference_its_own_id():
    schema = _primary_schema_with_ref("urn:test:primary#/$defs/record")
    assert validate_schema_draft202012(schema, primary_schema_name="bundle_prediction.json") == []
    assert validate_against_jsonschema({}, schema=schema, primary_schema_name="bundle_prediction.json") == []


def test_no_network_registry_accepts_explicit_additional_resources():
    schema = _primary_schema_with_ref("common.json#/$defs/shared")
    common = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "common.json",
        "$defs": {"shared": {"type": "object"}},
    }
    assert validate_schema_draft202012(
        schema,
        primary_schema_name="bundle_prediction.json",
        additional_resources={"common.json": common},
    ) == []
    assert validate_against_jsonschema(
        {},
        schema=schema,
        primary_schema_name="bundle_prediction.json",
        additional_resources={"common.json": common},
    ) == []


def test_registry_behavior_does_not_depend_on_repo_schema_files(monkeypatch):
    import external_baselines.interop.schema as schema_module

    class PoisonPath:
        name = "poison.schema.json"

        def exists(self):
            raise AssertionError("registry must not inspect repo-local schema files")

    monkeypatch.setattr(schema_module, "SCHEMA_PATH", PoisonPath())
    monkeypatch.setattr(schema_module, "SCHEMA_DRAFT_PATH", PoisonPath())
    schema = _primary_schema_with_ref("bundle_prediction.json#/$defs/record")
    assert schema_module.validate_schema_draft202012(schema, primary_schema_name="bundle_prediction.json") == []
    assert schema_module.validate_against_jsonschema(
        {},
        schema=schema,
        primary_schema_name="bundle_prediction.json",
    ) == []
    build_no_network_schema_registry(primary_schema=schema, primary_schema_name="bundle_prediction.json")


def test_schema_rejects_https_ref():
    errors = validate_schema_draft202012(_schema_with_ref("https://example.com/schema.json"))
    assert any("external_schema_remote_reference_forbidden" in err for err in errors)


def test_schema_rejects_http_ref():
    errors = validate_schema_draft202012(_schema_with_ref("http://example.com/schema.json"))
    assert any("external_schema_remote_reference_forbidden" in err for err in errors)


def test_schema_rejects_file_uri_ref():
    errors = validate_schema_draft202012(_schema_with_ref("file:///tmp/schema.json"))
    assert any("external_schema_file_reference_forbidden" in err for err in errors)


def test_schema_rejects_absolute_posix_file_ref():
    errors = validate_schema_draft202012(_schema_with_ref("/tmp/schema.json"))
    assert any("external_schema_file_reference_forbidden" in err for err in errors)


def test_schema_rejects_windows_drive_ref():
    errors = validate_schema_draft202012(_schema_with_ref(r"C:\tmp\schema.json"))
    assert any("external_schema_file_reference_forbidden" in err for err in errors)


def test_schema_rejects_unc_ref():
    errors = validate_schema_draft202012(_schema_with_ref(r"\\server\share\schema.json"))
    assert any("external_schema_file_reference_forbidden" in err for err in errors)


def test_schema_rejects_unregistered_relative_ref():
    errors = validate_schema_draft202012(_schema_with_ref("other_schema.json"))
    assert any("external_schema_reference_not_registered" in err for err in errors)


def test_schema_rejects_remote_dynamic_ref():
    errors = validate_schema_draft202012(
        _schema_with_ref("https://example.com/dynamic.json", keyword="$dynamicRef")
    )
    assert any("external_schema_remote_reference_forbidden" in err for err in errors)


def test_schema_validation_does_not_call_network(monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        monkeypatch.setattr(requests, "get", _blocked)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"record": {"type": "object"}},
        "$ref": "#/$defs/record",
    }
    assert validate_against_jsonschema({}, schema=schema) == []


def test_meta_and_record_validation_use_same_ref_policy():
    schema = _schema_with_ref("https://example.com/schema.json")
    meta_errors = validate_schema_draft202012(schema)
    record_errors = validate_against_jsonschema({}, schema=schema)
    assert any("external_schema_remote_reference_forbidden" in err for err in meta_errors)
    assert any("external_schema_remote_reference_forbidden" in err for err in record_errors)


def test_lightweight_validator_rejects_non_object_record():
    errors = validate_interop_record([], require_external_schema=False)  # type: ignore[arg-type]
    assert "invalid:record_must_be_object" in errors


def test_lightweight_validator_rejects_prediction_list():
    record = baseline_row_to_interop(_sample_row())
    record["prediction"] = ["invalid"]
    errors = validate_interop_record(record, require_external_schema=False)
    assert "invalid:prediction_must_be_object" in errors


def test_lightweight_validator_rejects_final_response_string():
    record = baseline_row_to_interop(_sample_row())
    record["prediction"]["final_response"] = "invalid"
    errors = validate_interop_record(record, require_external_schema=False)
    assert "invalid:prediction.final_response_must_be_object" in errors


def test_lightweight_validator_rejects_runtime_list():
    record = baseline_row_to_interop(_sample_row())
    record["runtime"] = ["invalid"]
    errors = validate_interop_record(record, require_external_schema=False)
    assert "invalid:runtime_must_be_object" in errors


def test_lightweight_validator_rejects_recommended_actions_object():
    record = baseline_row_to_interop(_sample_row())
    record["prediction"]["recommended_actions"] = {"bad": True}
    errors = validate_interop_record(record, require_external_schema=False)
    assert "invalid:prediction.recommended_actions_must_be_array" in errors


def test_lightweight_validator_rejects_blocked_actions_string():
    record = baseline_row_to_interop(_sample_row())
    record["prediction"]["blocked_actions"] = "bad"
    errors = validate_interop_record(record, require_external_schema=False)
    assert "invalid:prediction.blocked_actions_must_be_array" in errors


def test_lightweight_validator_rejects_non_boolean_human_review():
    record = baseline_row_to_interop(_sample_row())
    record["prediction"]["human_review_required"] = "false"
    errors = validate_interop_record(record, require_external_schema=False)
    assert "invalid:prediction.human_review_required_must_be_boolean" in errors


def test_lightweight_validator_never_raises_attribute_error():
    malformed = [
        [],
        {"prediction": []},
        {"prediction": {"final_response": "bad"}},
        {"runtime": []},
    ]
    for record in malformed:
        errors = validate_interop_record(record, require_external_schema=False)  # type: ignore[arg-type]
        assert isinstance(errors, list)
        assert errors
