"""Formal execution guard, alias parity, canonical output, and contract script tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baselines.common.decision_output import DecisionParseError, parse_decision_output
from external_baselines.common.decision_suite_guard import (
    SMOKE_CONFIG_FORBIDDEN_MESSAGE,
    FormalConfigurationError,
    FormalSuiteExecutionError,
    assert_formal_smoke_config_forbidden,
    validate_decision_suite_execution,
)
from external_baselines.common.firebench_taxonomy import (
    DEV_ALIASES_PATH,
    FORMAL_ALIAS_KEYS,
    alias_map,
    load_dev_aliases,
    load_formal_aliases,
    membership_set,
)
from external_baselines.common.formal_config_validator import FormalConfigError, validate_llm_for_formal
from external_baselines.common.taxonomy_normalizer import (
    assert_canonical_interop_record,
    validate_canonical_interop_record,
)
from scripts.check_firebench_contract_snapshot import main as schema_check_main
from scripts.check_firebench_taxonomy_snapshot import compare_taxonomy_snapshots
from scripts.run_decision_comparison_suite import _base_smoke_config, run_decision_suite

ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO = ROOT.parent / "fire-agent-demo"


def _valid_payload(**overrides) -> dict:
    base = {
        "decision": {
            "risk_signals": ["electrical_risk"],
            "risk_level": "high",
            "recommended_actions": [
                {
                    "action_id": "verify_power_isolation",
                    "text": "确认电源已经切断。",
                    "priority": "high",
                    "evidence_refs": [],
                }
            ],
            "blocked_actions": ["BLOCK_UNVERIFIED_WATER_SUPPRESSION"],
            "missing_confirmations": ["power_cutoff_status"],
            "human_review_required": True,
            "final_decision_gate": "await_human_confirmation",
        },
        "response": {
            "status": "awaiting_human_confirmation",
            "text": "当前存在电气风险，请先确认断电。",
            "citations": [],
        },
    }
    base.update(overrides)
    return base


def _interop_record(**overrides) -> dict:
    base = {
        "schema_version": "firebench-interop-v1",
        "case_id": "FBPUB_000001",
        "method_id": "direct_llm",
        "prediction": {
            "risk_signals": ["electrical_risk"],
            "risk_level": "high",
            "recommended_actions": [
                {
                    "action_id": "verify_power_isolation",
                    "text": "确认电源。",
                    "priority": "high",
                    "evidence_refs": [],
                }
            ],
            "blocked_actions": ["BLOCK_UNVERIFIED_WATER_SUPPRESSION"],
            "missing_confirmations": ["power_cutoff_status"],
            "human_review_required": True,
            "final_decision_gate": "await_human_confirmation",
            "final_response": {
                "status": "awaiting_human_confirmation",
                "text": "请先确认断电。",
                "citations": [],
                "real_world_execution_allowed": False,
            },
        },
    }
    base.update(overrides)
    return base


def test_formal_requires_experiment_manifest():
    with pytest.raises(FormalSuiteExecutionError, match="experiment manifest"):
        validate_decision_suite_execution(
            execution_stage="formal",
            experiment_manifest=None,
            method_ids=[
                "direct_llm",
                "bm25_rag",
                "dense_rag",
                "hybrid_rag",
                "ekell_style_controlled_shared_llm",
            ],
        )


def test_formal_rejects_missing_manifest_file(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FormalSuiteExecutionError, match="missing file"):
        validate_decision_suite_execution(
            execution_stage="formal",
            experiment_manifest=missing,
            method_ids=[
                "direct_llm",
                "bm25_rag",
                "dense_rag",
                "hybrid_rag",
                "ekell_style_controlled_shared_llm",
            ],
        )


def test_formal_rejects_example_manifest(tmp_path):
    example = tmp_path / "controlled_main_table_v1.yaml.example"
    example.write_text("experiment_id: x\n", encoding="utf-8")
    with pytest.raises(FormalSuiteExecutionError, match="\\.example"):
        validate_decision_suite_execution(
            execution_stage="formal",
            experiment_manifest=example,
            method_ids=[
                "direct_llm",
                "bm25_rag",
                "dense_rag",
                "hybrid_rag",
                "ekell_style_controlled_shared_llm",
            ],
        )


def test_formal_rejects_heuristic_provider():
    with pytest.raises(FormalConfigError, match="smoke LLM provider"):
        validate_llm_for_formal(
            {
                "llm": {
                    "provider": "heuristic",
                    "model": "real-model",
                    "model_version": "v1",
                }
            }
        )


def test_formal_rejects_smoke_model_name():
    with pytest.raises(FormalConfigError, match="smoke/heuristic LLM model name"):
        validate_llm_for_formal(
            {
                "llm": {
                    "provider": "openai_compatible",
                    "model": "smoke-fixture-model",
                    "model_version": "v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "seed": 1,
                }
            }
        )


def test_formal_rejects_smoke_dense_backend():
    from external_baselines.common.formal_config_validator import validate_dense_config_for_real_run

    with pytest.raises(FormalConfigError, match="smoke/hash backend"):
        validate_dense_config_for_real_run(
            {
                "dense_rag": {
                    "backend": "smoke",
                    "reject_smoke": True,
                    "model_name": "real-embed",
                    "model_version": "v1",
                    "dimension": 64,
                    "normalize_embeddings": True,
                    "index_path": "data/indexes/dense",
                }
            },
            validation_stage="formal",
        )


def test_formal_rejects_smoke_hybrid_dependency():
    from external_baselines.common.formal_config_validator import validate_hybrid_config_for_real_run

    with pytest.raises(FormalConfigError, match="reject_smoke"):
        validate_hybrid_config_for_real_run(
            {
                "hybrid_rag": {"reject_smoke": False},
                "dense_rag": {
                    "backend": "real",
                    "reject_smoke": True,
                    "model_name": "real-embed",
                    "model_version": "v1",
                    "dimension": 64,
                    "normalize_embeddings": True,
                    "index_path": "data/indexes/dense",
                },
            },
            validation_stage="formal",
        )


def test_formal_rejects_smoke_ekell_backend():
    from external_baselines.common.formal_config_validator import validate_ekell_vector_for_formal

    with pytest.raises(FormalConfigError, match="smoke/hash backend"):
        validate_ekell_vector_for_formal(
            {
                "ekell_vector": {
                    "backend": "smoke",
                    "reject_smoke": True,
                    "model_version": "v1",
                    "dimension": 32,
                    "index_path": "data/indexes/ekell",
                }
            }
        )


def test_formal_validation_occurs_before_llm_build(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    called = {"llm": False}

    def _boom(*_args, **_kwargs):
        called["llm"] = True
        raise AssertionError("build_llm_client should not run for formal without manifest")

    monkeypatch.setattr(suite, "build_llm_client", _boom)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(FormalSuiteExecutionError):
        run_decision_suite(
            runner_bundle=bundle,
            prediction_dir=tmp_path / "pred",
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=None,
        )
    assert called["llm"] is False


def test_dry_run_still_allows_fixture_config():
    cfg = _base_smoke_config("data/corpus", execution_stage="dry_run")
    assert cfg["llm"]["provider"] == "heuristic"
    assert cfg["dense_rag"]["backend"] == "smoke"


def test_formal_smoke_config_forbidden():
    with pytest.raises(FormalConfigurationError, match=SMOKE_CONFIG_FORBIDDEN_MESSAGE):
        assert_formal_smoke_config_forbidden(execution_stage="formal")
    with pytest.raises(FormalConfigurationError):
        _base_smoke_config("data/corpus", execution_stage="formal")


@pytest.mark.skipif(not MAIN_REPO.is_dir(), reason="main repo not available")
def test_formal_alias_snapshot_matches_main_project_fixture():
    result = compare_taxonomy_snapshots(MAIN_REPO)
    assert result["canonical_sets_match"] is True
    assert result["formal_alias_maps_match"] is True


def test_no_extra_formal_aliases():
    formal = load_formal_aliases()
    dev = load_dev_aliases()
    for key in FORMAL_ALIAS_KEYS:
        formal_block = formal.get(key) or {}
        dev_block = dev.get(key) or {}
        overlap = set(formal_block.keys()) & set(dev_block.keys())
        assert not overlap, f"dev alias overlaps formal source in {key}: {overlap}"


def test_no_missing_official_aliases():
    formal = load_formal_aliases()
    for key in FORMAL_ALIAS_KEYS:
        assert len(formal.get(key) or {}) > 0, key


def test_formal_alias_targets_match_exactly():
    formal = load_formal_aliases()
    for key in FORMAL_ALIAS_KEYS:
        for target in (formal.get(key) or {}).values():
            assert target in membership_set(key), (key, target)


def test_dev_aliases_are_disabled_in_formal():
    assert "check_power_isolation" not in alias_map("recommended_action_ids", dev_aliases_enabled=False)


def test_dev_aliases_require_explicit_enable_flag():
    assert (
        alias_map("recommended_action_ids", dev_aliases_enabled=True).get("check_power_isolation")
        == "verify_power_isolation"
    )


def test_final_output_rejects_risk_alias():
    record = _interop_record()
    record["prediction"]["risk_signals"] = ["electrical_hazard"]
    errors = validate_canonical_interop_record(record)
    assert any(e["error"] == "noncanonical_alias_in_final_output" for e in errors)


def test_final_output_rejects_action_alias():
    record = _interop_record()
    record["prediction"]["recommended_actions"][0]["action_id"] = "check_power_isolation"
    errors = validate_canonical_interop_record(record)
    assert any(e["error"] == "noncanonical_alias_in_final_output" for e in errors)


def test_final_output_rejects_block_alias():
    record = _interop_record()
    record["prediction"]["blocked_actions"] = ["block_real_world_execution"]
    errors = validate_canonical_interop_record(record)
    assert any(e["error"] == "noncanonical_alias_in_final_output" for e in errors)


def test_final_output_rejects_confirmation_alias():
    record = _interop_record()
    record["prediction"]["missing_confirmations"] = ["power_status"]
    errors = validate_canonical_interop_record(record)
    assert any(e["error"] == "noncanonical_alias_in_final_output" for e in errors)


def test_final_output_rejects_gate_alias():
    record = _interop_record()
    record["prediction"]["final_decision_gate"] = "allow"
    errors = validate_canonical_interop_record(record, dev_aliases_enabled=True)
    assert any(e["error"] == "noncanonical_alias_in_final_output" for e in errors)


def test_final_output_rejects_noncanonical_character_form():
    record = _interop_record()
    record["prediction"]["risk_signals"] = ["Electrical-Risk"]
    errors = validate_canonical_interop_record(record)
    assert any(e["error"] == "noncanonical_alias_in_final_output" for e in errors)


def test_final_output_accepts_only_canonical_values():
    record = _interop_record()
    assert validate_canonical_interop_record(record) == []
    assert_canonical_interop_record(record)


def test_formal_requires_risk_signals_key():
    payload = _valid_payload()
    del payload["decision"]["risk_signals"]
    with pytest.raises(DecisionParseError, match="missing_risk_signals"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_risk_level_key():
    payload = _valid_payload()
    del payload["decision"]["risk_level"]
    with pytest.raises(DecisionParseError, match="missing_risk_level"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_recommended_actions_key():
    payload = _valid_payload()
    del payload["decision"]["recommended_actions"]
    with pytest.raises(DecisionParseError, match="missing_recommended_actions"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_blocked_actions_key():
    payload = _valid_payload()
    del payload["decision"]["blocked_actions"]
    with pytest.raises(DecisionParseError, match="missing_blocked_actions"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_missing_confirmations_key():
    payload = _valid_payload()
    del payload["decision"]["missing_confirmations"]
    with pytest.raises(DecisionParseError, match="missing_missing_confirmations"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_human_review_required_key():
    payload = _valid_payload()
    del payload["decision"]["human_review_required"]
    with pytest.raises(DecisionParseError, match="missing_human_review_required"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_final_gate_key():
    payload = _valid_payload()
    del payload["decision"]["final_decision_gate"]
    with pytest.raises(DecisionParseError, match="missing_final_decision_gate"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_response_status_key():
    payload = _valid_payload()
    del payload["response"]["status"]
    with pytest.raises(DecisionParseError, match="missing_response_status"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_response_text_key():
    payload = _valid_payload()
    del payload["response"]["text"]
    with pytest.raises(DecisionParseError, match="missing_response_text"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_response_citations_key():
    payload = _valid_payload()
    del payload["response"]["citations"]
    with pytest.raises(DecisionParseError, match="missing_response_citations"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_action_priority_key():
    payload = _valid_payload()
    del payload["decision"]["recommended_actions"][0]["priority"]
    with pytest.raises(DecisionParseError, match="missing_action_priority"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_action_evidence_refs_key():
    payload = _valid_payload()
    del payload["decision"]["recommended_actions"][0]["evidence_refs"]
    with pytest.raises(DecisionParseError, match="missing_action_evidence_refs"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_explicit_unknown_values_are_allowed():
    payload = _valid_payload(
        decision={
            "risk_signals": [],
            "risk_level": "unknown",
            "recommended_actions": [],
            "blocked_actions": [],
            "missing_confirmations": [],
            "human_review_required": False,
            "final_decision_gate": "unknown",
        },
        response={"status": "unknown", "text": "当前信息不足，无法形成可靠决策。", "citations": []},
    )
    out = parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)
    assert out.risk_level == "unknown"
    assert out.final_decision_gate == "unknown"


def test_explicit_empty_arrays_are_allowed():
    payload = _valid_payload(
        decision={
            **_valid_payload()["decision"],
            "risk_signals": [],
            "blocked_actions": [],
            "missing_confirmations": [],
        }
    )
    out = parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)
    assert out.risk_signals == []
    assert out.blocked_actions == []


def test_schema_check_without_main_repo_uses_local_snapshot():
    assert schema_check_main([]) == 0


@pytest.mark.skipif(not MAIN_REPO.is_dir(), reason="main repo not available")
def test_schema_check_with_matching_main_repo_passes():
    assert schema_check_main(["--main-repo", str(MAIN_REPO)]) == 0


def test_schema_check_with_mismatch_fails(tmp_path):
    fake = tmp_path / "fake-main"
    fake.mkdir()
    schema_dir = fake / "schemas" / "firebench_interop_v1"
    schema_dir.mkdir(parents=True)
    (schema_dir / "prediction_schema.json").write_text('{"mismatch": true}', encoding="utf-8")
    assert schema_check_main(["--main-repo", str(fake)]) == 1


def test_schema_check_with_explicit_missing_main_repo_fails(tmp_path):
    missing = tmp_path / "nope"
    assert schema_check_main(["--main-repo", str(missing)]) == 1


@pytest.mark.skipif(not MAIN_REPO.is_dir(), reason="main repo not available")
def test_taxonomy_snapshot_check_detects_extra_alias(monkeypatch):
    patched = json.loads(json.dumps(load_formal_aliases()))
    patched["risk_signals"]["totally_fake_extra_alias"] = "electrical_risk"
    monkeypatch.setattr(
        "scripts.check_firebench_taxonomy_snapshot.load_formal_aliases",
        lambda: patched,
    )
    result = compare_taxonomy_snapshots(MAIN_REPO)
    assert result["formal_alias_maps_match"] is False
    assert result["extra_aliases"]


@pytest.mark.skipif(not MAIN_REPO.is_dir(), reason="main repo not available")
def test_taxonomy_snapshot_check_detects_missing_alias(monkeypatch):
    patched = json.loads(json.dumps(load_formal_aliases()))
    first_key = next(iter(patched["risk_signals"]))
    patched["risk_signals"].pop(first_key)
    monkeypatch.setattr(
        "scripts.check_firebench_taxonomy_snapshot.load_formal_aliases",
        lambda: patched,
    )
    result = compare_taxonomy_snapshots(MAIN_REPO)
    assert result["formal_alias_maps_match"] is False
    assert result["missing_aliases"]


@pytest.mark.skipif(not MAIN_REPO.is_dir(), reason="main repo not available")
def test_taxonomy_snapshot_check_detects_wrong_alias_target(monkeypatch):
    patched = json.loads(json.dumps(load_formal_aliases()))
    first_key = next(iter(patched["risk_signals"]))
    original_target = patched["risk_signals"][first_key]
    patched["risk_signals"][first_key] = next(
        item for item in membership_set("risk_signals") if item != original_target
    )
    monkeypatch.setattr(
        "scripts.check_firebench_taxonomy_snapshot.load_formal_aliases",
        lambda: patched,
    )
    result = compare_taxonomy_snapshots(MAIN_REPO)
    assert result["formal_alias_maps_match"] is False
    assert result["different_targets"]


def test_dev_alias_file_exists():
    assert DEV_ALIASES_PATH.is_file()
