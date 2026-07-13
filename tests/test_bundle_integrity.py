from __future__ import annotations

import json

import pytest

from external_baselines.common.checksums import sha256_file
from external_baselines.common.io import load_scenarios
from external_baselines.interop.bundle import (
    BundleIntegrityError,
    assert_path_inside_bundle,
    load_runner_bundle,
    recompute_bundle_checksum,
    validate_bundle_checksum,
)


def _write_formal_bundle(tmp_path, input_text: str, *, input_name: str = "input_cases.jsonl"):
    root = tmp_path / "bundle"
    root.mkdir()
    input_path = root / input_name
    input_path.write_text(input_text, encoding="utf-8")
    schema_path = root / "prediction_schema.json"
    schema_path.write_text(
        json.dumps({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}),
        encoding="utf-8",
    )
    manifest = {
        "bundle_type": "runner",
        "files": {
            "input_cases": input_name,
            "prediction_schema": "prediction_schema.json",
        },
        "checksums": {
            input_name: sha256_file(input_path),
            "prediction_schema.json": sha256_file(schema_path),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_checksum_detects_tampering(tmp_path):
    (tmp_path / "scenarios.json").write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "x"}]}),
        encoding="utf-8",
    )
    declared = recompute_bundle_checksum(tmp_path)
    bundle = {
        "bundle_root": str(tmp_path),
        "bundle_checksum": declared,
        "producer_declared_checksum": declared,
        "consumer_computed_bundle_hash": declared,
        "recomputed_bundle_checksum": declared,
        "file_checksum_report": {},
    }
    assert validate_bundle_checksum(bundle)["ok"] is True

    (tmp_path / "scenarios.json").write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "tampered"}]}),
        encoding="utf-8",
    )
    # Recompute consumer hash after tamper; declared producer checksum no longer matches.
    bundle["consumer_computed_bundle_hash"] = recompute_bundle_checksum(tmp_path)
    bundle["recomputed_bundle_checksum"] = bundle["consumer_computed_bundle_hash"]
    assert validate_bundle_checksum(bundle)["ok"] is False


def test_empty_path_is_rejected(tmp_path):
    with pytest.raises(BundleIntegrityError):
        load_runner_bundle("")
    with pytest.raises(BundleIntegrityError):
        assert_path_inside_bundle("", tmp_path)


def test_nested_gold_key_hard_fails(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"metadata": {"gold": {"answer": "forbidden"}}}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        load_runner_bundle(tmp_path)


def test_formal_input_jsonl_accepts_valid_object_with_case_id(tmp_path):
    root = _write_formal_bundle(
        tmp_path,
        '{"case_id":"FBPUB_000001","input":{"scenario":"smoke near exit"}}\n',
    )

    bundle = load_runner_bundle(root, formal=True)

    assert bundle["input_cases_formal_eligible"] is True


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ('"text"\n', "formal_input_cases_record_must_be_object:line_1"),
        ("123\n", "formal_input_cases_record_must_be_object:line_1"),
        ("[]\n", "formal_input_cases_record_must_be_object:line_1"),
        ("null\n", "formal_input_cases_record_must_be_object:line_1"),
    ],
)
def test_formal_input_jsonl_rejects_non_object_rows(tmp_path, row, code):
    root = _write_formal_bundle(tmp_path, row)

    with pytest.raises(BundleIntegrityError, match=code):
        load_runner_bundle(root, formal=True)


def test_formal_input_jsonl_rejects_invalid_json(tmp_path):
    root = _write_formal_bundle(tmp_path, "{not-json}\n")

    with pytest.raises(BundleIntegrityError, match="formal_input_cases_invalid_json:line_1"):
        load_runner_bundle(root, formal=True)


def test_formal_input_jsonl_requires_case_id(tmp_path):
    root = _write_formal_bundle(tmp_path, '{"input":{"scenario":"smoke"}}\n')

    with pytest.raises(BundleIntegrityError, match="formal_input_cases_case_id_missing:line_1"):
        load_runner_bundle(root, formal=True)


def test_formal_input_jsonl_rejects_empty_case_id(tmp_path):
    root = _write_formal_bundle(tmp_path, '{"case_id":"   ","input":{"scenario":"smoke"}}\n')

    with pytest.raises(BundleIntegrityError, match="formal_input_cases_case_id_empty:line_1"):
        load_runner_bundle(root, formal=True)


def test_formal_input_jsonl_rejects_non_string_case_id(tmp_path):
    root = _write_formal_bundle(tmp_path, '{"case_id":123,"input":{"scenario":"smoke"}}\n')

    with pytest.raises(BundleIntegrityError, match="formal_input_cases_case_id_must_be_string:line_1"):
        load_runner_bundle(root, formal=True)


def test_formal_input_jsonl_rejects_duplicate_case_id(tmp_path):
    root = _write_formal_bundle(
        tmp_path,
        "\n".join(
            [
                '{"case_id":"FBPUB_000001","input":{"scenario":"smoke"}}',
                '{"case_id":"FBPUB_000001","input":{"scenario":"fire"}}',
            ]
        )
        + "\n",
    )

    with pytest.raises(BundleIntegrityError, match="formal_input_cases_duplicate_case_id:line_2"):
        load_runner_bundle(root, formal=True)


def test_formal_input_jsonl_rejects_empty_file(tmp_path):
    root = _write_formal_bundle(tmp_path, "\n")

    with pytest.raises(BundleIntegrityError, match="formal_input_cases_empty"):
        load_runner_bundle(root, formal=True)


def test_formal_input_cases_must_be_jsonl(tmp_path):
    root = _write_formal_bundle(
        tmp_path,
        '[{"case_id":"FBPUB_000001","input":{"scenario":"smoke"}}]\n',
        input_name="input_cases.json",
    )

    with pytest.raises(BundleIntegrityError, match="formal_runner_bundle_input_cases_must_be_jsonl"):
        load_runner_bundle(root, formal=True)


def test_dry_run_loader_retains_documented_legacy_flexibility(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text('"ignored legacy row"\n{"scenario_id":"s1","scenario_text":"smoke"}\n', encoding="utf-8")

    scenarios = load_scenarios(path)

    assert [scenario["case_id"] for scenario in scenarios] == ["s1"]
