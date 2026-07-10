"""Interop contract tests against main-project firebench-interop-v1 shapes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baselines.common.io import load_scenarios, to_prediction_input
from external_baselines.interop.bundle import (
    BundleIntegrityError,
    assert_no_evaluator_bundle_access,
    load_runner_bundle,
    validate_bundle_checksum,
)
from external_baselines.interop.schema import baseline_row_to_interop, validate_interop_record
from external_baselines.method_registry import main_table_methods

ROOT = Path(__file__).resolve().parents[1]
MAIN_BUNDLE = (
    ROOT.parent / "fire-agent-demo" / "artifacts" / "firebench_interop_v1" / "runner_seed_curated"
)


def test_load_main_project_input_cases_jsonl(tmp_path):
    path = tmp_path / "input_cases.jsonl"
    path.write_text(
        json.dumps({
            "case_id": "FB_TEST_001",
            "input": {"language": "en", "input_mode": "text_only", "scenario": "Electrical fire smoke."},
            "dynamic_snapshots": [{"sensors": {"smoke_level": {"value": "high"}}}],
            "category": "electrical_fire",
        })
        + "\n",
        encoding="utf-8",
    )
    rows = load_scenarios(path)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "FB_TEST_001"
    assert rows[0]["scenario_text"] == "Electrical fire smoke."
    assert rows[0]["language"] == "en"
    assert rows[0]["input_mode"] == "text_only"
    assert rows[0]["dynamic_snapshots"]


def test_nested_input_scenario_and_prediction_input():
    from external_baselines.common.io import flatten_scenario

    record = {
        "case_id": "FB_TEST_002",
        "input": {"language": "zh", "input_mode": "text_with_dynamic_snapshot", "scenario": "浓烟"},
        "dynamic_snapshots": [{"ts": 1}],
        "context": {"k": "v"},
    }
    flat = flatten_scenario(record)
    pred = to_prediction_input(flat)
    assert pred["scenario_text"] == "浓烟"
    assert pred["dynamic_snapshots"] == [{"ts": 1}]
    assert pred["language"] == "zh"
    assert pred["context"] == {"k": "v"}


def test_blocked_actions_are_string_ids():
    row = {
        "scenario_id": "c1",
        "method": "bm25_rag",
        "key_risks": ["electrical_risk"],
        "recommended_actions": [{"action_id": "verify_power_isolation", "text": "Check power", "priority": "high"}],
        "blocked_or_unsafe_actions": ["BLOCK_UNVERIFIED_WATER_SUPPRESSION"],
        "missing_confirmations": ["power_cutoff_status"],
        "final_decision_gate": "await_human_confirmation",
        "retrieved_contexts": [],
        "latency_sec": 0.01,
        "raw_output": {"parsed": {}},
        "method_specific": {"runtime": {"llm_calls": 1, "token_usage": {}, "cost": None}},
    }
    record = baseline_row_to_interop(row)
    assert record["prediction"]["blocked_actions"] == ["BLOCK_UNVERIFIED_WATER_SUPPRESSION"]
    assert record["prediction"]["final_decision_gate"] == "await_human_confirmation"
    assert record["prediction"]["final_response"]["status"] == "awaiting_human_confirmation"


def test_gate_and_status_enums():
    row = {
        "scenario_id": "c1",
        "method": "direct_llm",
        "final_decision_gate": "not_provided_by_baseline",
        "recommended_actions": [],
        "blocked_or_unsafe_actions": [],
        "missing_confirmations": [],
        "key_risks": [],
        "latency_sec": 0.0,
        "method_specific": {"runtime": {"llm_calls": 0, "token_usage": {}, "cost": None}},
    }
    record = baseline_row_to_interop(row)
    assert record["prediction"]["final_decision_gate"] == "unknown"
    assert record["prediction"]["final_response"]["status"] == "unknown"


@pytest.mark.skipif(not MAIN_BUNDLE.exists(), reason="main-project runner_seed_curated not present")
def test_load_main_runner_bundle_manifest():
    bundle = load_runner_bundle(MAIN_BUNDLE)
    assert bundle["formal_manifest_files_used"] is True
    assert bundle["scenarios_path"].endswith("input_cases.jsonl")
    assert Path(bundle["prediction_schema_path"]).name == "prediction_schema.json"
    report = validate_bundle_checksum(bundle)
    assert report["ok"] is True
    assert report["file_checksum_report"]["ok"] is True
    cases = load_scenarios(bundle["scenarios_path"], limit=2)
    assert len(cases) == 2
    assert cases[0]["scenario_text"]
    assert "dynamic_snapshots" in cases[0]


def test_evaluator_bundle_path_rejected(tmp_path):
    bad = tmp_path / "evaluator_seed_curated"
    bad.mkdir()
    with pytest.raises(PermissionError):
        assert_no_evaluator_bundle_access(bad)


def test_valid_non_gold_metadata_not_rejected(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "label_coverage_version.json").write_text('{"version": 1}\n', encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"bundle_type": "runner", "files": {}, "checksums": {}}),
        encoding="utf-8",
    )
    # Should not raise on label_coverage_version filename.
    load_runner_bundle(root)


def test_main_table_methods_registry():
    assert list(main_table_methods()) == [
        "direct_llm",
        "bm25_rag",
        "ekell_style_controlled_shared_llm",
    ]


def test_empty_path_rejected(tmp_path):
    from external_baselines.interop.bundle import assert_path_inside_bundle

    with pytest.raises(BundleIntegrityError):
        assert_path_inside_bundle("", tmp_path)
    with pytest.raises(BundleIntegrityError):
        assert_path_inside_bundle("   ", tmp_path)
