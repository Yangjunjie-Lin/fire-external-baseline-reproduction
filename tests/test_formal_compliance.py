"""Formal execution guard, alias parity, canonical output, and contract script tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from external_baselines.common.decision_output import DecisionParseError, parse_decision_output
from external_baselines.common.decision_suite_guard import (
    SMOKE_CONFIG_FORBIDDEN_MESSAGE,
    FormalConfigurationError,
    FormalRunFailed,
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
from external_baselines.common.runtime_evidence import (
    RuntimeEvidence,
    collect_llm_evidence,
    compute_suite_formal_compliance,
)
from external_baselines.common.taxonomy_normalizer import (
    assert_canonical_interop_record,
    validate_canonical_interop_record,
)
from external_baselines.interop.bundle import (
    BundleIntegrityError,
    inspect_runner_bundle_case_coverage,
    load_runner_bundle,
    validate_formal_runner_bundle_coverage,
)
from scripts.check_firebench_contract_snapshot import main as schema_check_main
from scripts.check_firebench_taxonomy_snapshot import compare_taxonomy_snapshots
from scripts.run_decision_comparison_suite import _base_smoke_config, run_decision_suite
from tests.test_decision_comparison_suite import _make_runner_bundle

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


def test_formal_rejects_limit():
    with pytest.raises(FormalSuiteExecutionError, match="forbids --limit"):
        validate_decision_suite_execution(
            execution_stage="formal",
            experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
            method_ids=[
                "direct_llm",
                "bm25_rag",
                "dense_rag",
                "hybrid_rag",
                "ekell_style_controlled_shared_llm",
            ],
            limit=3,
        )


def test_formal_python_api_rejects_limit(tmp_path):
    bundle = _make_runner_bundle(tmp_path)
    with pytest.raises(FormalSuiteExecutionError, match="forbids --limit"):
        run_decision_suite(
            runner_bundle=bundle,
            prediction_dir=tmp_path / "pred",
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            limit=1,
            experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        )


def test_dry_run_allows_limit(tmp_path):
    bundle = _make_runner_bundle(tmp_path, n_cases=5)
    coverage = inspect_runner_bundle_case_coverage(bundle, limit=2)
    assert coverage.loaded_case_count == 2
    assert coverage.input_file_case_count == 5


def test_formal_uses_full_bundle_case_count(tmp_path):
    bundle = _make_runner_bundle(tmp_path, n_cases=4)
    coverage = inspect_runner_bundle_case_coverage(bundle, limit=None)
    validate_formal_runner_bundle_coverage(coverage)
    assert coverage.loaded_case_count == coverage.input_file_case_count == 4


def test_formal_rejects_manifest_case_count_mismatch(tmp_path):
    bundle = _make_runner_bundle(tmp_path, n_cases=2)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["case_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    coverage = inspect_runner_bundle_case_coverage(bundle, limit=None)
    with pytest.raises(BundleIntegrityError, match="case_count mismatch"):
        validate_formal_runner_bundle_coverage(coverage)


def test_formal_rejects_partial_case_id_set(tmp_path):
    bundle = _make_runner_bundle(tmp_path, n_cases=3)
    coverage = inspect_runner_bundle_case_coverage(bundle, limit=2)
    with pytest.raises(BundleIntegrityError, match="complete Runner Bundle"):
        validate_formal_runner_bundle_coverage(coverage)


def test_formal_dense_rejects_legacy_json_index(tmp_path):
    from external_baselines.retrieval.dense_index import DenseIndexError, validate_dense_index_directory

    legacy = tmp_path / "dense_index.json"
    legacy.write_text("{}", encoding="utf-8")
    with pytest.raises(DenseIndexError, match="legacy_dense_json_forbidden_in_formal"):
        validate_dense_index_directory(legacy)


def test_formal_dense_requires_directory(tmp_path):
    from external_baselines.retrieval.dense_index import DenseIndexError, validate_dense_index_directory

    with pytest.raises(DenseIndexError, match="dense_index_path_not_directory"):
        validate_dense_index_directory(tmp_path / "missing_dir")


def test_formal_dense_requires_index_manifest(tmp_path):
    from external_baselines.retrieval.dense_index import DenseIndexError, validate_dense_index_directory

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    with pytest.raises(DenseIndexError, match="dense_index_manifest_missing"):
        validate_dense_index_directory(index_dir)


def test_formal_dense_requires_documents_jsonl(tmp_path):
    from external_baselines.retrieval.dense_index import DenseIndexError, validate_dense_index_directory

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "index_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DenseIndexError, match="dense_index_documents_missing"):
        validate_dense_index_directory(index_dir)


def test_formal_dense_requires_embeddings_npy(tmp_path):
    from external_baselines.retrieval.dense_index import DenseIndexError, validate_dense_index_directory

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "index_manifest.json").write_text("{}", encoding="utf-8")
    (index_dir / "documents.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(DenseIndexError, match="dense_index_embeddings_missing"):
        validate_dense_index_directory(index_dir)


def test_formal_dense_never_calls_build_dense_index(tmp_path, monkeypatch):
    from external_baselines.dense_rag import pipeline as dense_pipeline
    from external_baselines.retrieval.dense_index import DenseIndexError

    called = {"build": False}

    def _boom(*_args, **_kwargs):
        called["build"] = True
        raise AssertionError("build_dense_index must not run in formal")

    monkeypatch.setattr(dense_pipeline, "build_dense_index", _boom)
    from external_baselines.common.method_runtime import prepare_dense_runtime

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    with pytest.raises(DenseIndexError, match="forbids pipeline rebuild"):
        prepare_dense_runtime(
            {
                "execution_stage": "formal",
                "paper_final": True,
                "paths": {"corpus_dir": str(tmp_path)},
                "dense_rag": {
                    "backend": "text2vec",
                    "model_name": "fake/bge",
                    "model_version": "v1",
                    "dimension": 8,
                    "index_path": str(index_dir),
                    "reject_smoke": True,
                },
            }
        )
    assert called["build"] is False


def test_formal_dense_loads_persisted_index(tmp_path):
    from external_baselines.dense_rag.pipeline import build_dense_index
    from external_baselines.retrieval.dense_index import validate_dense_index_directory
    from tests.test_dense_real_index import FakeEmbeddingModel, _evidence

    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "idx"
    build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=FakeEmbeddingModel(8),
        reject_smoke=True,
    )
    payload = validate_dense_index_directory(index_dir, load_embeddings=False)
    assert len(payload["documents"]) == 3


def test_formal_ekell_requires_index_path():
    from external_baselines.common.formal_config_validator import validate_ekell_vector_for_formal

    with pytest.raises(FormalConfigError, match="persisted directory index_path"):
        validate_ekell_vector_for_formal(
            {
                "ekell_vector": {
                    "backend": "text2vec",
                    "model_name": "fake/bge",
                    "model_version": "v1",
                    "dimension": 8,
                    "reject_smoke": True,
                }
            }
        )


def test_formal_ekell_requires_directory_index(tmp_path):
    from external_baselines.common.formal_config_validator import validate_ekell_vector_for_formal

    legacy = tmp_path / "ekell.json"
    legacy.write_text("{}", encoding="utf-8")
    with pytest.raises(FormalConfigError, match="legacy_ekell_json_forbidden_in_formal"):
        validate_ekell_vector_for_formal(
            {
                "ekell_vector": {
                    "backend": "text2vec",
                    "model_name": "fake/bge",
                    "model_version": "v1",
                    "dimension": 8,
                    "index_path": str(legacy),
                    "reject_smoke": True,
                }
            }
        )


def test_formal_ekell_requires_index_manifest(tmp_path):
    from external_baselines.ekell_style.vector_index import VectorIndexError

    index_dir = tmp_path / "ekell_idx"
    index_dir.mkdir()
    with pytest.raises(VectorIndexError, match="missing required file|ekell_index_manifest_missing"):
        from external_baselines.ekell_style.vector_index import VectorIndex

        VectorIndex.validate_directory(index_dir)


def test_preflight_checks_all_five_methods_before_execution(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    called = {"preflight": 0, "llm": 0}

    def _preflight(**kwargs):
        called["preflight"] += 1
        return {"ok": True, "execution_stage": kwargs["execution_stage"], "methods": {}}

    def _llm(*_args, **_kwargs):
        called["llm"] += 1
        from external_baselines.common.llm_client import HeuristicLLMClient

        return HeuristicLLMClient()

    monkeypatch.setattr(suite, "preflight_decision_suite", _preflight)
    monkeypatch.setattr(suite, "build_llm_client", _llm)
    bundle = _make_runner_bundle(tmp_path)
    run_decision_suite(
        runner_bundle=bundle,
        prediction_dir=tmp_path / "pred",
        decision_dir=tmp_path / "dec",
        execution_stage="dry_run",
        limit=1,
    )
    assert called["preflight"] == 1
    assert called["llm"] >= 1


def test_preflight_failure_prevents_any_llm_build(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    called = {"llm": False}

    def _boom(*_args, **_kwargs):
        called["llm"] = True
        raise AssertionError("LLM must not initialize when formal preflight fails")

    monkeypatch.setattr(suite, "build_llm_client", _boom)
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {"ok": False, "execution_stage": "formal", "methods": {}},
    )
    bundle = _make_runner_bundle(tmp_path)
    with pytest.raises(FormalRunFailed, match="preflight failed"):
        run_decision_suite(
            runner_bundle=bundle,
            prediction_dir=tmp_path / "pred",
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "formal_manifest.yaml",
        )
    assert called["llm"] is False


def test_preflight_failure_prevents_prediction_writes(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {"ok": False, "execution_stage": "formal", "methods": {}},
    )
    with pytest.raises(FormalRunFailed):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path),
            prediction_dir=pred_dir,
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "formal_manifest.yaml",
        )
    assert not any(pred_dir.glob("*.jsonl"))


def test_preflight_report_contains_each_method(tmp_path, monkeypatch):
    from external_baselines.common.decision_suite_preflight import preflight_decision_suite
    from scripts.run_decision_comparison_suite import _base_smoke_config

    bundle = _make_runner_bundle(tmp_path)
    loaded = load_runner_bundle(bundle)
    base = _base_smoke_config(loaded["corpus_dir"], execution_stage="dry_run")
    configs = {mid: dict(base) for mid in [
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ]}
    report = preflight_decision_suite(
        method_ids=list(configs),
        method_configs=configs,
        runner_bundle=bundle,
        execution_stage="dry_run",
    )
    assert set(report["methods"]) == set(configs)


def test_formal_requires_risk_signals_array():
    payload = _valid_payload(decision={**_valid_payload()["decision"], "risk_signals": "electrical_risk"})
    with pytest.raises(DecisionParseError, match="risk_signals_not_array"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_recommended_actions_array():
    payload = _valid_payload(
        decision={**_valid_payload()["decision"], "recommended_actions": {"action_id": "verify_power_isolation"}}
    )
    with pytest.raises(DecisionParseError, match="recommended_actions_not_array"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_blocked_actions_array():
    payload = _valid_payload(
        decision={**_valid_payload()["decision"], "blocked_actions": "BLOCK_UNVERIFIED_WATER_SUPPRESSION"}
    )
    with pytest.raises(DecisionParseError, match="blocked_actions_not_array"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_missing_confirmations_array():
    payload = _valid_payload(
        decision={**_valid_payload()["decision"], "missing_confirmations": "power_cutoff_status"}
    )
    with pytest.raises(DecisionParseError, match="missing_confirmations_not_array"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_response_citations_array():
    payload = _valid_payload(response={**_valid_payload()["response"], "citations": "evidence_001"})
    with pytest.raises(DecisionParseError, match="response_citations_not_array"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_requires_action_evidence_refs_array():
    payload = _valid_payload()
    payload["decision"]["recommended_actions"][0]["evidence_refs"] = "evidence_001"
    with pytest.raises(DecisionParseError, match="action_evidence_refs_not_array"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_rejects_non_string_risk_signal():
    payload = _valid_payload(decision={**_valid_payload()["decision"], "risk_signals": [123]})
    with pytest.raises(DecisionParseError, match="risk_signal_not_string"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_rejects_non_object_action():
    payload = _valid_payload(decision={**_valid_payload()["decision"], "recommended_actions": ["verify_power_isolation"]})
    with pytest.raises(DecisionParseError, match="recommended_action_not_object"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_rejects_non_string_blocked_action():
    payload = _valid_payload(decision={**_valid_payload()["decision"], "blocked_actions": [123]})
    with pytest.raises(DecisionParseError, match="blocked_action_not_string"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_rejects_non_string_confirmation():
    payload = _valid_payload(decision={**_valid_payload()["decision"], "missing_confirmations": [123]})
    with pytest.raises(DecisionParseError, match="confirmation_not_string"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_rejects_non_string_citation():
    payload = _valid_payload(response={**_valid_payload()["response"], "citations": [123]})
    with pytest.raises(DecisionParseError, match="response_citation_not_string"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_formal_rejects_non_string_action_evidence_ref():
    payload = _valid_payload()
    payload["decision"]["recommended_actions"][0]["evidence_refs"] = [123]
    with pytest.raises(DecisionParseError, match="action_evidence_ref_not_string"):
        parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=True)


def test_dry_run_records_array_type_error():
    payload = _valid_payload(decision={**_valid_payload()["decision"], "risk_signals": "electrical_risk"})
    out = parse_decision_output(payload, case_id="c1", method_id="direct_llm", strict=False)
    assert out.parsing_failure is True
    assert "risk_signals_not_array" in out.parsing_errors


def test_formal_summary_initializes_formal_result_false(tmp_path):
    bundle = _make_runner_bundle(tmp_path)
    summary = run_decision_suite(
        runner_bundle=bundle,
        prediction_dir=tmp_path / "pred",
        decision_dir=tmp_path / "dec",
        execution_stage="dry_run",
        limit=1,
    )
    assert summary["formal_compliance"]["formal_result"] is False


def test_formal_summary_uses_actual_llm_provider():
    evidence = collect_llm_evidence(
        method_id="direct_llm",
        config={"llm": {"provider": "openai_compatible", "model": "gpt-real", "model_version": "v1"}},
        llm=None,
    )
    assert evidence.llm_is_smoke is False


def test_formal_summary_rejects_smoke_runtime_evidence():
    evidence = collect_llm_evidence(
        method_id="direct_llm",
        config={"llm": {"provider": "heuristic", "model": "local-deterministic-heuristic-smoke-test"}},
        llm=None,
    )
    compliance = compute_suite_formal_compliance(
        formal=True,
        experiment_manifest_provided=True,
        limit_used=False,
        preflight_ok=True,
        coverage_ok=True,
        method_evidences={"direct_llm": evidence},
        method_compliance={"direct_llm": {"formal_result": False}},
        dev_aliases_enabled=False,
    )
    assert compliance["real_llm"] is False
    assert compliance["formal_result"] is False


def test_dense_runtime_records_index_loaded():
    from external_baselines.common.runtime_evidence import collect_dense_runtime_evidence

    class FakeRuntime:
        index_manifest = {"actual_embedding_used": True, "smoke_fallback_used": False, "document_count": 2}
        embedding_backend = None
        dense_index = object()
        audit = type("A", (), {"index_load_count": 1})()
        index_built_during_run = False

    evidence = collect_dense_runtime_evidence(
        method_id="dense_rag",
        config={"dense_rag": {"index_path": "/idx"}},
        runtime=FakeRuntime(),
    )
    assert evidence.index_loaded is True
    assert evidence.index_built_during_run is False


def test_formal_result_requires_complete_runtime_evidence():
    smoke = collect_llm_evidence(
        method_id="direct_llm",
        config={"llm": {"provider": "heuristic", "model": "smoke"}},
        llm=None,
    )
    real = collect_llm_evidence(
        method_id="direct_llm",
        config={"llm": {"provider": "openai_compatible", "model": "gpt-real"}},
        llm=None,
    )
    bad = compute_suite_formal_compliance(
        formal=True,
        experiment_manifest_provided=True,
        limit_used=False,
        preflight_ok=True,
        coverage_ok=True,
        method_evidences={"direct_llm": smoke},
        method_compliance={"direct_llm": {"formal_result": True}},
        dev_aliases_enabled=False,
    )
    good = compute_suite_formal_compliance(
        formal=True,
        experiment_manifest_provided=True,
        limit_used=False,
        preflight_ok=True,
        coverage_ok=True,
        method_evidences={"direct_llm": real},
        method_compliance={"direct_llm": {"formal_result": True}},
        dev_aliases_enabled=False,
    )
    assert bad["formal_result"] is False
    assert good["real_llm"] is True


def test_method_summary_uses_null_for_non_applicable_index():
    from external_baselines.common.runtime_evidence import method_formal_compliance

    evidence = RuntimeEvidence(method_id="direct_llm", llm_is_smoke=False, llm_initialized=True)
    compliance = method_formal_compliance(
        evidence,
        formal=True,
        method_id="direct_llm",
        coverage_ok=True,
        parsing_failures=0,
        schema_failures=0,
        taxonomy_valid=True,
    )
    assert compliance["real_index"] is None


# --- Helpers for bundle integrity, generation identity, and transactional tests ---


def _shared_generation_llm(**overrides) -> dict:
    base = {
        "provider": "openai_compatible",
        "model": "gpt-real-shared",
        "model_version": "v-shared",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": 20260710,
        "enable_thinking": False,
    }
    base.update(overrides)
    return base


def _finalize_bundle_checksums(bundle_dir: Path) -> dict:
    from external_baselines.common.checksums import sha256_file
    from external_baselines.interop.bundle import load_runner_bundle, recompute_bundle_checksum

    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_map = manifest.get("files") or {}
    checksums = {}
    for rel in files_map.values():
        file_path = bundle_dir / rel
        if file_path.is_file():
            checksums[str(rel)] = sha256_file(file_path)
    manifest["checksums"] = checksums
    manifest["bundle_checksum"] = recompute_bundle_checksum(bundle_dir)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return load_runner_bundle(bundle_dir)


def _frozen_runner_bundle_identity(bundle: dict) -> dict:
    from external_baselines.common.checksums import sha256_file

    corpus_manifest = bundle.get("corpus_manifest") if isinstance(bundle.get("corpus_manifest"), dict) else {}
    return {
        "producer_declared_checksum": bundle.get("producer_declared_checksum"),
        "consumer_computed_hash": bundle.get("consumer_computed_bundle_hash")
        or bundle.get("recomputed_bundle_checksum"),
        "producer_checksum_available": bool(bundle.get("producer_declared_checksum")),
        "input_cases_sha256": sha256_file(bundle["scenarios_path"]),
        "prediction_schema_sha256": bundle.get("prediction_schema_sha256"),
        "corpus_aggregate_sha256": corpus_manifest.get("aggregate_sha256"),
    }


def _write_shared_model_config(tmp_path: Path, llm: dict | None = None) -> Path:
    import yaml

    path = tmp_path / "shared_model.yaml"
    path.write_text(yaml.safe_dump({"llm": llm or _shared_generation_llm()}), encoding="utf-8")
    return path


def _write_experiment_manifest(tmp_path: Path, *, freeze_path: Path, shared_model_path: Path) -> Path:
    import yaml

    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "formal_test",
                "freeze_manifest": str(freeze_path),
                "shared_model_config": str(shared_model_path),
                "paper_final": True,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_freeze_manifest(tmp_path: Path, bundle: dict, **overrides) -> Path:
    freeze_body = _frozen_runner_bundle_identity(bundle)
    freeze_body.update(overrides)
    if "runner_bundle" in overrides:
        freeze_body = overrides["runner_bundle"]
    freeze_path = tmp_path / "freeze_manifest.json"
    payload = {"runner_bundle": freeze_body} if "runner_bundle" not in overrides else overrides
    if "runner_bundle" not in payload:
        payload = {"runner_bundle": freeze_body}
    for key, value in overrides.items():
        if key != "runner_bundle":
            payload[key] = value
    freeze_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return freeze_path


def _build_fake_dense_index(tmp_path: Path) -> Path:
    from external_baselines.dense_rag.pipeline import build_dense_index
    from tests.test_dense_real_index import FakeEmbeddingModel, _evidence

    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "dense_idx"
    build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=FakeEmbeddingModel(8),
        reject_smoke=True,
    )
    return index_dir


def _build_fake_ekell_index(tmp_path: Path) -> Path:
    from external_baselines.ekell_style.embedding_backends import create_embedding_backend
    from external_baselines.ekell_style.kg_loader import FireKG
    from external_baselines.ekell_style.vector_index import VectorIndex
    from tests.test_ekell_index_persistence import FakeEmbeddingModel

    fake = FakeEmbeddingModel(8)
    backend = create_embedding_backend(
        "text2vec",
        model_name="fake/bge",
        model_version="v-test",
        dimension=8,
        model=fake,
        reject_smoke=True,
    )
    kg = FireKG(
        entities=[{"entity_id": "e1", "name": "hose"}],
        relations=[{"relation_id": "r1", "name": "used_for"}],
        triples=[{"head": "hose", "relation": "used_for", "tail": "fire", "source_id": "t1"}],
        evidence_chunks=[{"chunk_id": "c1", "text": "fire hose near exit", "source_id": "s1"}],
    )
    index = VectorIndex.from_kg(kg, backend, reject_smoke=True)
    index_dir = tmp_path / "ekell_idx"
    index.save_directory(index_dir)
    return index_dir


def _passing_formal_method_evidences() -> dict[str, RuntimeEvidence]:
    method_ids = [
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ]
    evidences: dict[str, RuntimeEvidence] = {}
    for mid in method_ids:
        needs_index = mid in {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"}
        evidences[mid] = RuntimeEvidence(
            method_id=mid,
            llm_provider="local_contract_provider",
            llm_model="contract-generation-v1",
            llm_model_version="v1",
            llm_temperature=0.0,
            llm_top_p=1.0,
            llm_max_tokens=1024,
            llm_seed=20260710,
            llm_enable_thinking=False,
            llm_is_smoke=False,
            llm_initialized=True,
            index_loaded=needs_index,
            index_built_during_run=False,
            actual_embedding_used=True if needs_index else None,
            smoke_fallback_used=False if needs_index else None,
            dense_dependency_actual_embedding_used=True if mid == "hybrid_rag" else None,
            dense_dependency_smoke_fallback_used=False if mid == "hybrid_rag" else None,
            index_checksum="same" if needs_index else None,
            dense_dependency_index_checksum="same" if mid == "hybrid_rag" else None,
        )
    return evidences


def _passing_method_compliance() -> dict[str, dict]:
    return {mid: {"formal_result": True} for mid in _passing_formal_method_evidences()}


def _formal_method_configs(
    corpus_dir: str,
    *,
    dense_index: Path | None = None,
    llm_overrides: dict | None = None,
    method_llm_override: dict | None = None,
) -> dict[str, dict]:
    llm = _shared_generation_llm(**(llm_overrides or {}))
    base = {
        "execution_stage": "formal",
        "unified_decision_output": True,
        "strict_decision_parse": True,
        "dev_aliases_enabled": False,
        "paper_final": True,
        "llm": dict(llm),
        "paths": {"corpus_dir": corpus_dir},
        "retrieval": {"top_k": 3},
        "dense_rag": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(dense_index) if dense_index else str(Path(corpus_dir).parent / "missing"),
            "reject_smoke": True,
        },
        "hybrid_rag": {"top_k": 3, "candidate_pool": 5, "reject_smoke": True},
        "ekell_style": {"prompt_dir": str(ROOT / "configs/prompts/controlled"), "neighborhood_k_hop": 1},
        "ekell_vector": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(dense_index) if dense_index else str(Path(corpus_dir).parent / "missing"),
            "reject_smoke": True,
        },
        "scenario_parser": {"use_llm": False},
        "normalization": {"infer_structured_safety_fields": False},
    }
    configs = {
        "direct_llm": dict(base),
        "bm25_rag": dict(base),
        "dense_rag": dict(base),
        "hybrid_rag": dict(base),
        "ekell_style_controlled_shared_llm": dict(base),
    }
    if method_llm_override:
        configs["bm25_rag"]["llm"] = {**llm, **method_llm_override}
    return configs


# --- Bundle checksum tests ---


def test_formal_preflight_validates_bundle_checksum(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    frozen = _frozen_runner_bundle_identity(bundle)
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen)
    assert result["ok"] is True
    assert result["file_checksum_report_ok"] is True


def test_formal_rejects_modified_input_cases(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    frozen = _frozen_runner_bundle_identity(bundle)
    cases_path = Path(bundle["scenarios_path"])
    cases_path.write_text(
        cases_path.read_text(encoding="utf-8") + '{"case_id":"EXTRA","input":{"scenario":"x"}}\n',
        encoding="utf-8",
    )
    bundle = load_runner_bundle(bundle_dir)
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen)
    assert result["ok"] is False
    assert "input_cases_checksum_mismatch" in result["errors"]


def test_formal_rejects_modified_prediction_schema(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    frozen = _frozen_runner_bundle_identity(bundle)
    schema_path = Path(bundle["prediction_schema_path"])
    schema_path.write_text(schema_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    bundle = load_runner_bundle(bundle_dir)
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen)
    assert result["ok"] is False
    assert "prediction_schema_checksum_mismatch" in result["errors"]


def test_formal_rejects_modified_corpus_file(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    frozen = _frozen_runner_bundle_identity(bundle)
    corpus_dir = Path(bundle["corpus_dir"])
    extra = corpus_dir / "evidence_chunks.jsonl"
    extra.write_text(extra.read_text(encoding="utf-8") + '{"chunk_id":"c99","text":"extra"}\n', encoding="utf-8")
    bundle = load_runner_bundle(bundle_dir)
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen)
    assert result["ok"] is False
    assert "corpus_checksum_mismatch" in result["errors"]


def test_formal_rejects_bundle_checksum_mismatch(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    frozen = _frozen_runner_bundle_identity(bundle)
    frozen["consumer_computed_hash"] = "0" * 64
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen)
    assert result["ok"] is False
    assert "consumer_computed_hash_mismatch" in result["errors"]


def test_formal_rejects_missing_frozen_bundle_identity(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity={})
    assert result["ok"] is False
    assert "frozen_bundle_identity_missing" in result["errors"]


def test_bundle_integrity_failure_prevents_llm_build(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    called = {"llm": False}

    def _boom(*_args, **_kwargs):
        called["llm"] = True
        raise AssertionError("LLM must not initialize when bundle integrity fails")

    monkeypatch.setattr(suite, "build_llm_client", _boom)
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {
            "ok": False,
            "runner_bundle_integrity": {"ok": False, "errors": ["input_cases_checksum_mismatch"]},
            "methods": {},
        },
    )
    with pytest.raises(FormalRunFailed, match="preflight failed"):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path),
            prediction_dir=tmp_path / "pred",
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "formal_manifest.yaml",
        )
    assert called["llm"] is False


# --- Shared generation identity tests ---


def _generation_identity_configs(**method_overrides) -> dict[str, dict]:
    llm = _shared_generation_llm()
    configs = {mid: {"llm": dict(llm)} for mid in [
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ]}
    for method_id, override in method_overrides.items():
        configs[method_id]["llm"] = {**llm, **override}
    return configs


def test_formal_all_methods_share_provider():
    from external_baselines.common.generation_identity import validate_shared_generation_identity

    report = validate_shared_generation_identity(
        method_ids=list(_generation_identity_configs().keys()),
        method_configs=_generation_identity_configs(),
    )
    assert report["ok"] is True


def test_formal_all_methods_share_model():
    from external_baselines.common.generation_identity import validate_shared_generation_identity

    configs = _generation_identity_configs(bm25_rag={"model": "other"})
    report = validate_shared_generation_identity(
        method_ids=list(configs.keys()),
        method_configs=configs,
    )
    assert report["ok"] is False
    assert any(m["field"] == "model" for m in report["mismatches"])


def test_formal_all_methods_share_model_version():
    from external_baselines.common.generation_identity import validate_shared_generation_identity

    report = validate_shared_generation_identity(
        method_ids=list(_generation_identity_configs().keys()),
        method_configs=_generation_identity_configs(bm25_rag={"model_version": "v-other"}),
    )
    assert report["ok"] is False
    assert any(m["field"] == "model_version" for m in report["mismatches"])


def test_formal_all_methods_share_temperature():
    from external_baselines.common.generation_identity import validate_shared_generation_identity

    report = validate_shared_generation_identity(
        method_ids=list(_generation_identity_configs().keys()),
        method_configs=_generation_identity_configs(bm25_rag={"temperature": 0.5}),
    )
    assert report["ok"] is False
    assert any(m["field"] == "temperature" for m in report["mismatches"])


def test_formal_all_methods_share_top_p():
    from external_baselines.common.generation_identity import validate_shared_generation_identity

    report = validate_shared_generation_identity(
        method_ids=list(_generation_identity_configs().keys()),
        method_configs=_generation_identity_configs(bm25_rag={"top_p": 0.9}),
    )
    assert report["ok"] is False
    assert any(m["field"] == "top_p" for m in report["mismatches"])


def test_formal_all_methods_share_max_tokens():
    from external_baselines.common.generation_identity import validate_shared_generation_identity

    report = validate_shared_generation_identity(
        method_ids=list(_generation_identity_configs().keys()),
        method_configs=_generation_identity_configs(bm25_rag={"max_tokens": 1200}),
    )
    assert report["ok"] is False
    assert any(m["field"] == "max_tokens" for m in report["mismatches"])


def test_formal_all_methods_share_seed():
    from external_baselines.common.generation_identity import validate_shared_generation_identity

    report = validate_shared_generation_identity(
        method_ids=list(_generation_identity_configs().keys()),
        method_configs=_generation_identity_configs(bm25_rag={"seed": 1}),
    )
    assert report["ok"] is False
    assert any(m["field"] == "seed" for m in report["mismatches"])


def test_formal_all_methods_share_enable_thinking():
    from external_baselines.common.generation_identity import validate_shared_generation_identity

    report = validate_shared_generation_identity(
        method_ids=list(_generation_identity_configs().keys()),
        method_configs=_generation_identity_configs(bm25_rag={"enable_thinking": True}),
    )
    assert report["ok"] is False
    assert any(m["field"] == "enable_thinking" for m in report["mismatches"])


def test_method_config_cannot_override_shared_llm_in_formal():
    from external_baselines.common.generation_identity import detect_method_llm_overrides

    shared = {"llm": _shared_generation_llm()}
    method = {"llm": _shared_generation_llm(max_tokens=1200)}
    overrides = detect_method_llm_overrides(shared_config=shared, method_config=method)
    assert overrides
    assert overrides[0]["field"] == "max_tokens"


def test_generation_mismatch_prevents_any_llm_build(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    called = {"llm": False}

    def _boom(*_args, **_kwargs):
        called["llm"] = True
        raise AssertionError("LLM must not initialize when generation identity mismatches")

    monkeypatch.setattr(suite, "build_llm_client", _boom)
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})

    def _preflight(**kwargs):
        from external_baselines.common.generation_identity import validate_shared_generation_identity

        configs = kwargs["method_configs"]
        for mid in configs:
            configs[mid].setdefault("llm", _shared_generation_llm())
        configs["bm25_rag"]["llm"]["max_tokens"] = 1200
        gen = validate_shared_generation_identity(method_ids=kwargs["method_ids"], method_configs=configs)
        return {"ok": False, "shared_generation_identity": gen, "runner_bundle_integrity": {"ok": True}, "methods": {}}

    monkeypatch.setattr(suite, "preflight_decision_suite", _preflight)
    bundle = _make_runner_bundle(tmp_path)
    with pytest.raises(FormalRunFailed, match="preflight failed"):
        run_decision_suite(
            runner_bundle=bundle,
            prediction_dir=tmp_path / "pred",
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "formal_manifest.yaml",
        )
    assert called["llm"] is False


def test_runtime_generation_identity_is_rechecked():
    from external_baselines.common.generation_identity import validate_runtime_generation_identity

    base = RuntimeEvidence(
        method_id="direct_llm",
        llm_provider="openai_compatible",
        llm_model="gpt-real",
        llm_model_version="v1",
    )
    other = RuntimeEvidence(
        method_id="bm25_rag",
        llm_provider="openai_compatible",
        llm_model="gpt-other",
        llm_model_version="v1",
    )
    report = validate_runtime_generation_identity(
        method_ids=["direct_llm", "bm25_rag"],
        method_evidences={"direct_llm": base, "bm25_rag": other},
    )
    assert report["ok"] is False


# --- Dense embedding evidence tests ---


def test_formal_dense_requires_actual_embedding_used_field():
    from external_baselines.retrieval.dense_index import DenseIndexError, require_dense_formal_embedding_manifest

    with pytest.raises(DenseIndexError, match="actual_embedding_used_missing"):
        require_dense_formal_embedding_manifest({})


def test_formal_dense_requires_actual_embedding_used_true():
    from external_baselines.retrieval.dense_index import DenseIndexError, require_dense_formal_embedding_manifest

    with pytest.raises(DenseIndexError, match="actual_embedding_used_must_be_true"):
        require_dense_formal_embedding_manifest({"actual_embedding_used": False, "smoke_fallback_used": False})


def test_formal_dense_requires_smoke_fallback_used_field():
    from external_baselines.retrieval.dense_index import DenseIndexError, require_dense_formal_embedding_manifest

    with pytest.raises(DenseIndexError, match="smoke_fallback_used_missing"):
        require_dense_formal_embedding_manifest({"actual_embedding_used": True})


def test_formal_dense_requires_smoke_fallback_used_false():
    from external_baselines.retrieval.dense_index import DenseIndexError, require_dense_formal_embedding_manifest

    with pytest.raises(DenseIndexError, match="smoke_fallback_used_must_be_false"):
        require_dense_formal_embedding_manifest({"actual_embedding_used": True, "smoke_fallback_used": True})


def test_dense_missing_embedding_flags_fails_preflight(tmp_path):
    from external_baselines.retrieval.dense_index import DenseIndexError, validate_dense_index_directory

    index_dir = _build_fake_dense_index(tmp_path)
    manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
    manifest.pop("actual_embedding_used", None)
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DenseIndexError, match="actual_embedding_used_missing"):
        validate_dense_index_directory(index_dir, require_explicit_embedding_evidence=True)


def test_dense_none_embedding_evidence_fails_compliance():
    from external_baselines.common.runtime_evidence import method_formal_compliance

    evidence = RuntimeEvidence(
        method_id="dense_rag",
        llm_is_smoke=False,
        index_loaded=True,
        index_built_during_run=False,
        actual_embedding_used=None,
        smoke_fallback_used=None,
    )
    compliance = method_formal_compliance(
        evidence,
        formal=True,
        method_id="dense_rag",
        coverage_ok=True,
        parsing_failures=0,
        schema_failures=0,
        taxonomy_valid=True,
    )
    assert compliance["formal_result"] is False
    assert compliance["real_index"] is False


def test_hybrid_requires_explicit_real_dense_evidence():
    from external_baselines.common.runtime_evidence import collect_hybrid_runtime_evidence

    class DenseRuntime:
        index_manifest = {"actual_embedding_used": True, "smoke_fallback_used": False, "index_checksum": "abc"}
        dense_index = object()
        audit = type("A", (), {"index_load_count": 1})()
        index_built_during_run = False

    class HybridRuntime:
        dense_runtime = DenseRuntime()
        lexical_retriever = object()

    evidence = collect_hybrid_runtime_evidence(
        method_id="hybrid_rag",
        config={"dense_rag": {"index_path": "/idx"}, "hybrid_rag": {"rrf_k": 60, "candidate_pool": 10}},
        runtime=HybridRuntime(),
    )
    assert evidence.dense_dependency_actual_embedding_used is True
    assert evidence.dense_dependency_smoke_fallback_used is False


# --- Manifest SHA tests ---


def test_runtime_evidence_records_real_manifest_file_sha(tmp_path):
    from external_baselines.common.checksums import sha256_file
    from external_baselines.common.runtime_evidence import collect_dense_runtime_evidence

    index_dir = _build_fake_dense_index(tmp_path)
    expected_sha = sha256_file(index_dir / "index_manifest.json")

    class FakeRuntime:
        index_manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        index_manifest["index_dir"] = str(index_dir)
        dense_index = type("I", (), {"index_dir": str(index_dir)})()
        audit = type("A", (), {"index_load_count": 1})()
        index_built_during_run = False

    evidence = collect_dense_runtime_evidence(
        method_id="dense_rag",
        config={"dense_rag": {"index_path": str(index_dir)}},
        runtime=FakeRuntime(),
    )
    assert evidence.index_manifest_sha256 == expected_sha


def test_index_checksum_differs_from_manifest_file_sha_when_expected(tmp_path):
    from external_baselines.common.checksums import sha256_file
    from external_baselines.common.runtime_evidence import collect_dense_runtime_evidence

    index_dir = _build_fake_dense_index(tmp_path)
    manifest_sha = sha256_file(index_dir / "index_manifest.json")

    class FakeRuntime:
        index_manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        index_manifest["index_dir"] = str(index_dir)
        dense_index = type("I", (), {"index_dir": str(index_dir)})()
        audit = type("A", (), {"index_load_count": 1})()
        index_built_during_run = False

    evidence = collect_dense_runtime_evidence(
        method_id="dense_rag",
        config={"dense_rag": {"index_path": str(index_dir)}},
        runtime=FakeRuntime(),
    )
    assert evidence.index_checksum
    assert evidence.index_manifest_sha256 == manifest_sha
    assert evidence.index_checksum != evidence.index_manifest_sha256


def test_dense_manifest_sha_matches_sha256_file(tmp_path):
    from external_baselines.common.checksums import sha256_file
    from external_baselines.common.runtime_evidence import _manifest_file_sha

    index_dir = _build_fake_dense_index(tmp_path)
    assert _manifest_file_sha(index_dir) == sha256_file(index_dir / "index_manifest.json")


def test_ekell_manifest_sha_matches_sha256_file(tmp_path):
    from external_baselines.common.checksums import sha256_file
    from external_baselines.common.runtime_evidence import _manifest_file_sha
    from external_baselines.ekell_style.vector_index import VectorIndex

    index_dir = tmp_path / "ekell_idx"
    dense_dir = _build_fake_dense_index(tmp_path)
    manifest = json.loads((dense_dir / "index_manifest.json").read_text(encoding="utf-8"))
    manifest["index_type"] = "ekell_kg_vector_index"
    index_dir.mkdir()
    for name in ("index_manifest.json", "documents.jsonl", "embeddings.npy"):
        shutil.copy(dense_dir / name, index_dir / name)
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _manifest_file_sha(index_dir) == sha256_file(index_dir / "index_manifest.json")
    VectorIndex.validate_directory(index_dir, load_embeddings=False)


def test_configured_and_resolved_index_paths_match(tmp_path):
    from external_baselines.common.runtime_evidence import collect_dense_runtime_evidence

    index_dir = _build_fake_dense_index(tmp_path)

    class FakeRuntime:
        index_manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        index_manifest["index_dir"] = str(index_dir)
        dense_index = type("I", (), {"index_dir": str(index_dir)})()
        audit = type("A", (), {"index_load_count": 1})()
        index_built_during_run = False

    evidence = collect_dense_runtime_evidence(
        method_id="dense_rag",
        config={"dense_rag": {"index_path": str(index_dir)}},
        runtime=FakeRuntime(),
    )
    assert Path(evidence.configured_index_path).resolve() == Path(evidence.resolved_index_path).resolve()
    assert not evidence.errors


# --- E-KELL preflight tests ---


def _ekell_prompt_dir(tmp_path: Path) -> Path:
    src = ROOT / "configs/prompts/controlled"
    dst = tmp_path / "prompts"
    shutil.copytree(src, dst)
    return dst


def test_ekell_preflight_requires_projection_prompt(tmp_path):
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    prompt_dir = _ekell_prompt_dir(tmp_path)
    (prompt_dir / "stepwise_projection.txt").unlink()
    report = _validate_ekell_prompts(prompt_dir, freeze=None, formal=True)
    assert "ekell_prompt_missing:stepwise_projection.txt" in report["errors"]


def test_ekell_preflight_requires_intersection_prompt(tmp_path):
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    prompt_dir = _ekell_prompt_dir(tmp_path)
    (prompt_dir / "stepwise_intersection.txt").unlink()
    report = _validate_ekell_prompts(prompt_dir, freeze=None, formal=True)
    assert "ekell_prompt_missing:stepwise_intersection.txt" in report["errors"]


def test_ekell_preflight_requires_union_prompt(tmp_path):
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    prompt_dir = _ekell_prompt_dir(tmp_path)
    (prompt_dir / "stepwise_union.txt").unlink()
    report = _validate_ekell_prompts(prompt_dir, freeze=None, formal=True)
    assert "ekell_prompt_missing:stepwise_union.txt" in report["errors"]


def test_ekell_preflight_requires_negation_prompt(tmp_path):
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    prompt_dir = _ekell_prompt_dir(tmp_path)
    (prompt_dir / "stepwise_negation.txt").unlink()
    report = _validate_ekell_prompts(prompt_dir, freeze=None, formal=True)
    assert "ekell_prompt_missing:stepwise_negation.txt" in report["errors"]


def test_ekell_preflight_requires_final_prompt(tmp_path):
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    prompt_dir = _ekell_prompt_dir(tmp_path)
    (prompt_dir / "final_kg_grounded_response.txt").unlink()
    report = _validate_ekell_prompts(prompt_dir, freeze=None, formal=True)
    assert "ekell_prompt_missing:final_kg_grounded_response.txt" in report["errors"]


def test_ekell_preflight_rejects_empty_prompt(tmp_path):
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    prompt_dir = _ekell_prompt_dir(tmp_path)
    (prompt_dir / "stepwise_projection.txt").write_text("   \n", encoding="utf-8")
    report = _validate_ekell_prompts(prompt_dir, freeze=None, formal=True)
    assert "ekell_prompt_empty:stepwise_projection.txt" in report["errors"]


def test_ekell_preflight_records_prompt_hashes(tmp_path):
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    prompt_dir = _ekell_prompt_dir(tmp_path)
    report = _validate_ekell_prompts(prompt_dir, freeze=None, formal=True)
    assert report["ok"] is True
    assert len(report["prompt_hashes"]) == 5


def test_ekell_preflight_detects_prompt_hash_mismatch(tmp_path):
    from external_baselines.common.checksums import sha256_file
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    prompt_dir = _ekell_prompt_dir(tmp_path)
    actual = sha256_file(prompt_dir / "stepwise_projection.txt")
    report = _validate_ekell_prompts(
        prompt_dir,
        freeze={"ekell_prompt_hashes": {"stepwise_projection.txt": "0" * 64}},
        formal=True,
    )
    assert "ekell_prompt_hash_mismatch:stepwise_projection.txt" in report["errors"]
    assert actual != "0" * 64


def test_ekell_preflight_checks_logical_components():
    from external_baselines.common.decision_suite_preflight import _validate_ekell_logical_components

    assert _validate_ekell_logical_components() == []


def test_ekell_preflight_checks_kg_jsonl_parseability(tmp_path):
    from external_baselines.common.decision_suite_preflight import _validate_kg_jsonl

    corpus = _tiny_corpus_from_test_decision_suite(tmp_path)
    assert _validate_kg_jsonl(corpus) == []


def _tiny_corpus_from_test_decision_suite(tmp_path: Path) -> Path:
    from tests.test_decision_comparison_suite import _tiny_corpus

    return _tiny_corpus(tmp_path)


# --- Dry-run formal result tests ---


def test_dry_run_method_formal_result_is_always_false():
    from external_baselines.common.runtime_evidence import method_formal_compliance

    evidence = RuntimeEvidence(method_id="direct_llm", llm_is_smoke=False, llm_initialized=True)
    compliance = method_formal_compliance(
        evidence,
        formal=False,
        method_id="direct_llm",
        coverage_ok=True,
        parsing_failures=0,
        schema_failures=0,
        taxonomy_valid=True,
    )
    assert compliance["formal_result"] is False
    assert compliance["reason"] == "execution_stage_not_formal"


def test_dry_run_suite_and_method_summaries_are_consistent(tmp_path):
    bundle = _make_runner_bundle(tmp_path)
    summary = run_decision_suite(
        runner_bundle=bundle,
        prediction_dir=tmp_path / "pred",
        decision_dir=tmp_path / "dec",
        execution_stage="dry_run",
        limit=1,
    )
    assert summary["formal_compliance"]["formal_result"] is False
    for method_summary in summary["method_summaries"].values():
        assert method_summary["formal_compliance"]["formal_result"] is False


def test_formal_method_result_requires_formal_stage():
    from external_baselines.common.runtime_evidence import method_formal_compliance

    evidence = RuntimeEvidence(method_id="direct_llm", llm_is_smoke=False, llm_initialized=True)
    dry = method_formal_compliance(
        evidence,
        formal=False,
        method_id="direct_llm",
        coverage_ok=True,
        parsing_failures=0,
        schema_failures=0,
        taxonomy_valid=True,
    )
    formal = method_formal_compliance(
        evidence,
        formal=True,
        method_id="direct_llm",
        coverage_ok=True,
        parsing_failures=0,
        schema_failures=0,
        taxonomy_valid=True,
    )
    assert dry["formal_result"] is False
    assert formal["formal_result"] is True


def test_technical_checks_passed_can_be_true_in_dry_run():
    from external_baselines.common.runtime_evidence import method_formal_compliance

    evidence = RuntimeEvidence(method_id="direct_llm", llm_is_smoke=False, llm_initialized=True)
    compliance = method_formal_compliance(
        evidence,
        formal=False,
        method_id="direct_llm",
        coverage_ok=True,
        parsing_failures=0,
        schema_failures=0,
        taxonomy_valid=True,
    )
    assert compliance["technical_checks_passed"] is True
    assert compliance["formal_result"] is False


# --- Transactional publishing tests ---


def test_formal_writes_to_temporary_directories(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    captured_dirs: list[Path] = []
    orig_ensure = suite.ensure_dir

    def _track(dir_path):
        captured_dirs.append(Path(dir_path))
        return orig_ensure(dir_path)

    monkeypatch.setattr(suite, "ensure_dir", _track)
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {
            "ok": True,
            "runner_bundle_integrity": {
                "ok": True,
                "input_cases_integrity": True,
                "prediction_schema_integrity": True,
                "corpus_integrity": True,
            },
            "shared_generation_identity": {"ok": True},
            "ekell_prompt_bundle_valid": True,
            "methods": {},
        },
    )

    def _runtime_evidence(**kwargs):
        mid = kwargs["method_id"]
        needs_index = mid in {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"}
        return RuntimeEvidence(
            method_id=mid,
            llm_provider="local_contract_provider",
            llm_model="contract-generation-v1",
            llm_model_version="v1",
            llm_temperature=0.0,
            llm_top_p=1.0,
            llm_max_tokens=1024,
            llm_seed=20260710,
            llm_enable_thinking=False,
            llm_is_smoke=False,
            llm_initialized=True,
            index_loaded=needs_index,
            index_built_during_run=False,
            actual_embedding_used=True if needs_index else None,
            smoke_fallback_used=False if needs_index else None,
            dense_dependency_actual_embedding_used=True if mid == "hybrid_rag" else None,
            dense_dependency_smoke_fallback_used=False if mid == "hybrid_rag" else None,
            index_checksum="same" if needs_index else None,
            dense_dependency_index_checksum="same" if mid == "hybrid_rag" else None,
        )

    def _fake_llm(_config):
        from external_baselines.common.llm_client import HeuristicLLMClient

        return HeuristicLLMClient()

    def _fake_pipeline(prediction_input, *, config, llm, runtime=None):
        from external_baselines.common.decision_output import decision_output_to_legacy_row, parse_decision_output

        parsed = parse_decision_output(
            _valid_payload(),
            case_id=prediction_input["case_id"],
            method_id="direct_llm",
            strict=True,
        )
        return decision_output_to_legacy_row(parsed)

    monkeypatch.setattr(suite, "build_llm_client", _fake_llm)
    monkeypatch.setattr(suite, "resolve_pipeline", lambda _mid: _fake_pipeline)
    monkeypatch.setattr(suite, "prepare_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "close_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "collect_method_runtime_evidence", _runtime_evidence)

    bundle = _make_runner_bundle(tmp_path, n_cases=1)
    pred_dir = tmp_path / "pred"
    dec_dir = tmp_path / "dec"
    run_decision_suite(
        runner_bundle=bundle,
        prediction_dir=pred_dir,
        decision_dir=dec_dir,
        execution_stage="formal",
        experiment_manifest=tmp_path / "manifest.yaml",
    )
    assert any(".formal_tmp_" in str(p) for p in captured_dirs)


def test_formal_preflight_failure_publishes_no_predictions(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {"ok": False, "runner_bundle_integrity": {"ok": False}, "methods": {}},
    )
    with pytest.raises(FormalRunFailed):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path),
            prediction_dir=pred_dir,
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "manifest.yaml",
        )
    assert not any(pred_dir.glob("*.jsonl"))
    assert (tmp_path / "FORMAL_RUN_FAILED.json").is_file()


def test_formal_api_failure_publishes_no_predictions(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    pred_dir = tmp_path / "pred"
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {
            "ok": True,
            "runner_bundle_integrity": {"ok": True},
            "shared_generation_identity": {"ok": True},
            "ekell_prompt_bundle_valid": True,
            "methods": {},
        },
    )
    monkeypatch.setattr(suite, "build_llm_client", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("timeout")))

    with pytest.raises(FormalRunFailed):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path),
            prediction_dir=pred_dir,
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "manifest.yaml",
        )
    assert not any(pred_dir.glob("*.jsonl"))


def test_formal_middle_method_failure_publishes_no_partial_results(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    pred_dir = tmp_path / "pred"
    call_count = {"n": 0}

    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {
            "ok": True,
            "runner_bundle_integrity": {"ok": True},
            "shared_generation_identity": {"ok": True},
            "ekell_prompt_bundle_valid": True,
            "methods": {},
        },
    )

    def _fake_llm(_config):
        from external_baselines.common.llm_client import HeuristicLLMClient

        return HeuristicLLMClient()

    def _fake_pipeline(prediction_input, *, config, llm, runtime=None):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("third method failed")
        from external_baselines.common.decision_output import parse_decision_output

        parsed = parse_decision_output(
            _valid_payload(),
            case_id=prediction_input["case_id"],
            method_id="direct_llm",
            strict=True,
        )
        return parsed.to_unified_row()

    monkeypatch.setattr(suite, "build_llm_client", _fake_llm)
    monkeypatch.setattr(suite, "resolve_pipeline", lambda _mid: _fake_pipeline)
    monkeypatch.setattr(suite, "prepare_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "close_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(
        suite,
        "collect_method_runtime_evidence",
        lambda **kwargs: RuntimeEvidence(method_id=kwargs["method_id"], llm_is_smoke=False, llm_initialized=True),
    )

    with pytest.raises(FormalRunFailed):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path, n_cases=1),
            prediction_dir=pred_dir,
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "manifest.yaml",
        )
    assert not any(pred_dir.glob("*.jsonl"))


def test_formal_success_atomically_publishes_all_methods(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    pred_dir = tmp_path / "pred"
    dec_dir = tmp_path / "dec"
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {
            "ok": True,
            "runner_bundle_integrity": {
                "ok": True,
                "input_cases_integrity": True,
                "prediction_schema_integrity": True,
                "corpus_integrity": True,
            },
            "shared_generation_identity": {"ok": True},
            "ekell_prompt_bundle_valid": True,
            "methods": {},
        },
    )

    def _fake_llm(_config):
        from external_baselines.common.llm_client import HeuristicLLMClient

        return HeuristicLLMClient()

    def _fake_pipeline(prediction_input, *, config, llm, runtime=None):
        from external_baselines.common.decision_output import decision_output_to_legacy_row, parse_decision_output

        parsed = parse_decision_output(
            _valid_payload(),
            case_id=prediction_input["case_id"],
            method_id="direct_llm",
            strict=True,
        )
        return decision_output_to_legacy_row(parsed)

    def _runtime_evidence(**kwargs):
        mid = kwargs["method_id"]
        needs_index = mid in {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"}
        return RuntimeEvidence(
            method_id=mid,
            llm_provider="local_contract_provider",
            llm_model="contract-generation-v1",
            llm_model_version="v1",
            llm_temperature=0.0,
            llm_top_p=1.0,
            llm_max_tokens=1024,
            llm_seed=20260710,
            llm_enable_thinking=False,
            llm_is_smoke=False,
            llm_initialized=True,
            index_loaded=needs_index,
            index_built_during_run=False,
            actual_embedding_used=True if needs_index else None,
            smoke_fallback_used=False if needs_index else None,
            dense_dependency_actual_embedding_used=True if mid == "hybrid_rag" else None,
            dense_dependency_smoke_fallback_used=False if mid == "hybrid_rag" else None,
            index_checksum="same" if needs_index else None,
            dense_dependency_index_checksum="same" if mid == "hybrid_rag" else None,
        )

    monkeypatch.setattr(suite, "build_llm_client", _fake_llm)
    monkeypatch.setattr(suite, "resolve_pipeline", lambda _mid: _fake_pipeline)
    monkeypatch.setattr(suite, "prepare_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "close_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "collect_method_runtime_evidence", _runtime_evidence)

    summary = run_decision_suite(
        runner_bundle=_make_runner_bundle(tmp_path, n_cases=1),
        prediction_dir=pred_dir,
        decision_dir=dec_dir,
        execution_stage="formal",
        experiment_manifest=tmp_path / "manifest.yaml",
    )
    assert len(list(pred_dir.glob("*.jsonl"))) == 5
    assert summary["formal_compliance"]["pre_publish_compliance_passed"] is True
    assert summary["formal_compliance"]["transactional_publish_complete"] is True
    assert summary["formal_compliance"]["formal_result"] is True


def test_formal_failure_writes_failed_marker(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {"ok": False, "runner_bundle_integrity": {"ok": False}, "methods": {}},
    )
    with pytest.raises(FormalRunFailed):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path),
            prediction_dir=tmp_path / "pred",
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "manifest.yaml",
        )
    marker = json.loads((tmp_path / "FORMAL_RUN_FAILED.json").read_text(encoding="utf-8"))
    assert marker["formal_outputs_published"] is False
    assert marker["execution_stage"] == "formal"


def test_formal_failure_marker_contains_no_secrets(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {"ok": False, "runner_bundle_integrity": {"ok": False}, "methods": {}},
    )
    with pytest.raises(FormalRunFailed):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path),
            prediction_dir=tmp_path / "pred",
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "manifest.yaml",
        )
    text = (tmp_path / "FORMAL_RUN_FAILED.json").read_text(encoding="utf-8").lower()
    for secret in ("api_key", "apikey", "bearer ", "sk-"):
        assert secret not in text


def test_dry_run_does_not_require_transactional_publish(tmp_path):
    bundle = _make_runner_bundle(tmp_path)
    pred_dir = tmp_path / "pred"
    summary = run_decision_suite(
        runner_bundle=bundle,
        prediction_dir=pred_dir,
        decision_dir=tmp_path / "dec",
        execution_stage="dry_run",
        limit=1,
    )
    assert summary["formal_compliance"]["transactional_publish_complete"] is False
    assert any(pred_dir.glob("*.jsonl"))


def test_decision_suite_builds_all_five_method_configs_from_real_manifest(tmp_path):
    from external_baselines.common.experiment_manifest import (
        build_method_config,
        get_method_entry,
        load_experiment_manifest,
    )

    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: local_contract_provider\n  model: contract-generation-v1\n"
        "  model_version: v1\n  temperature: 0.0\n  top_p: 1.0\n  max_tokens: 1024\n  seed: 20260710\n",
        encoding="utf-8",
    )
    method_paths = {}
    for mid in [
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ]:
        path = tmp_path / f"{mid}.yaml"
        path.write_text("execution_stage: formal\n", encoding="utf-8")
        method_paths[mid] = path
    exp = tmp_path / "formal_manifest.yaml"
    lines = [
        "experiment_id: formal_offline",
        f"shared_model_config: {shared.as_posix()}",
        "base_config: configs/default.yaml",
        "freeze_status: frozen",
        "paper_final: true",
        "methods:",
    ]
    for mid, path in method_paths.items():
        lines.append(f"  - method_id: {mid}")
        lines.append(f"    config: {path.as_posix()}")
        lines.append("    enabled: true")
    exp.write_text("\n".join(lines), encoding="utf-8")
    manifest = load_experiment_manifest(exp)
    configs = {}
    for mid in method_paths:
        entry = get_method_entry(manifest, mid)
        configs[mid] = build_method_config(manifest, entry)
    assert len(configs) == 5
    assert all(cfg["llm"]["model"] == "contract-generation-v1" for cfg in configs.values())


def test_pre_publish_compliance_does_not_require_publish_complete():
    compliance = compute_suite_formal_compliance(
        formal=True,
        experiment_manifest_provided=True,
        limit_used=False,
        preflight_ok=True,
        coverage_ok=True,
        method_evidences=_passing_formal_method_evidences(),
        method_compliance=_passing_method_compliance(),
        dev_aliases_enabled=False,
        runner_bundle_integrity_ok=True,
        input_cases_integrity_ok=True,
        prediction_schema_integrity_ok=True,
        corpus_integrity_ok=True,
        shared_generation_identity_match=True,
        runtime_generation_identity_match=True,
        ekell_prompt_bundle_valid=True,
        transactional_publish_complete=False,
        phase="pre_publish",
    )
    assert compliance["pre_publish_compliance_passed"] is True
    assert compliance["transactional_publish_complete"] is False
    assert compliance["formal_result"] is False


def test_final_formal_result_requires_publish_complete():
    base = dict(
        formal=True,
        experiment_manifest_provided=True,
        limit_used=False,
        preflight_ok=True,
        coverage_ok=True,
        method_evidences=_passing_formal_method_evidences(),
        method_compliance=_passing_method_compliance(),
        dev_aliases_enabled=False,
        runner_bundle_integrity_ok=True,
        input_cases_integrity_ok=True,
        prediction_schema_integrity_ok=True,
        corpus_integrity_ok=True,
        shared_generation_identity_match=True,
        runtime_generation_identity_match=True,
        ekell_prompt_bundle_valid=True,
    )
    pre = compute_suite_formal_compliance(**base, transactional_publish_complete=False, phase="pre_publish")
    final = compute_suite_formal_compliance(**base, transactional_publish_complete=True, phase="final")
    assert pre["pre_publish_compliance_passed"] is True
    assert pre["formal_result"] is False
    assert final["formal_result"] is True


def test_bundle_validation_failure_cannot_be_overridden_by_default_file_report(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    frozen = _frozen_runner_bundle_identity(bundle)
    bundle["producer_declared_checksum"] = "f" * 64
    bundle["file_checksum_report"] = {"ok": True}
    broken = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen)
    assert broken["ok"] is False
    assert broken["producer_checksum_match"] is False


def test_missing_file_checksum_report_does_not_default_to_pass(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = load_runner_bundle(bundle_dir)
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity={})
    assert result["ok"] is False
    assert "frozen_bundle_identity_missing" in result["errors"]


def test_complete_freeze_requires_input_cases_sha(tmp_path):
    from external_baselines.common.formal_config_validator import FormalConfigError
    from external_baselines.common.freeze_manifest import build_freeze_manifest_payload, validate_freeze_manifest
    from external_baselines.method_registry import comparison_suite_methods

    shared = ROOT / "configs/deterministic_heuristic_smoke.yaml"
    method_paths = {
        mid: tmp_path / f"{mid}.yaml"
        for mid in comparison_suite_methods()
    }
    for path in method_paths.values():
        path.write_text("execution_stage: formal\npaper_final: true\n", encoding="utf-8")
    method_entries = [
        {"method_id": mid, "config": str(path), "enabled": True}
        for mid, path in method_paths.items()
    ]
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "experiment_id: x\n"
        f"shared_model_config: {shared.as_posix()}\n"
        "methods:\n"
        + "\n".join(
            f"  - method_id: {entry['method_id']}\n    config: {entry['config']}\n    enabled: true"
            for entry in method_entries
        ),
        encoding="utf-8",
    )
    dev_evidence = tmp_path / "selected_dev_run.json"
    dev_evidence.write_text('{"selected": true}\n', encoding="utf-8")
    payload = build_freeze_manifest_payload(
        experiment_manifest_path=manifest,
        experiment_raw={"shared_model_config": str(shared), "methods": method_entries},
        selected_dev_run=dev_evidence,
        producer_declared_checksum="0" * 64,
        consumer_computed_hash="1" * 64,
        corpus_checksum="0" * 64,
        schema_checksum="0" * 64,
        method_config_paths={mid: str(path) for mid, path in method_paths.items()},
        indexes={
            "dense": {"index_checksum": "a"},
            "hybrid_dense_dependency": {"index_checksum": "a"},
            "ekell": {"index_checksum": "b"},
        },
    )
    payload["runner_bundle"].pop("input_cases_sha256", None)
    payload.pop("input_cases_sha256", None)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalConfigError, match="input_cases_sha256"):
        validate_freeze_manifest(
            freeze,
            experiment_manifest_path=manifest,
            experiment_raw={"shared_model_config": str(shared), "methods": method_entries},
            require_complete=True,
        )


def test_runtime_identity_checks_all_generation_fields():
    from external_baselines.common.generation_identity import validate_runtime_generation_identity

    base = RuntimeEvidence(
        method_id="direct_llm",
        llm_provider="local_contract_provider",
        llm_model="contract-generation-v1",
        llm_model_version="v1",
        llm_temperature=0.0,
        llm_top_p=1.0,
        llm_max_tokens=1024,
        llm_seed=20260710,
        llm_enable_thinking=False,
    )
    other = RuntimeEvidence(
        method_id="bm25_rag",
        llm_provider="local_contract_provider",
        llm_model="contract-generation-v1",
        llm_model_version="v1",
        llm_temperature=0.5,
        llm_top_p=1.0,
        llm_max_tokens=1024,
        llm_seed=20260710,
        llm_enable_thinking=False,
    )
    report = validate_runtime_generation_identity(
        method_ids=["direct_llm", "bm25_rag"],
        method_evidences={"direct_llm": base, "bm25_rag": other},
    )
    assert report["ok"] is False
    assert any(m["field"] == "temperature" for m in report["mismatches"])


def test_integrity_summary_fields_are_independent():
    compliance = compute_suite_formal_compliance(
        formal=True,
        experiment_manifest_provided=True,
        limit_used=False,
        preflight_ok=True,
        coverage_ok=True,
        method_evidences={},
        method_compliance={},
        dev_aliases_enabled=False,
        runner_bundle_integrity_ok=True,
        input_cases_integrity_ok=False,
        prediction_schema_integrity_ok=True,
        corpus_integrity_ok=True,
        phase="pre_publish",
    )
    assert compliance["runner_bundle_integrity"] is True
    assert compliance["input_cases_integrity"] is False
    assert compliance["prediction_schema_integrity"] is True


def test_formal_publish_second_target_failure_rolls_back_both_targets(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    pred_dir = tmp_path / "pred"
    dec_dir = tmp_path / "dec"
    pred_dir.mkdir()
    dec_dir.mkdir()
    (pred_dir / "old.jsonl").write_text('{"old": true}\n', encoding="utf-8")
    (dec_dir / "old.jsonl").write_text('{"old": true}\n', encoding="utf-8")
    calls = {"n": 0}

    def _publish_temp(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("decisions publish failed")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {
            "ok": True,
            "runner_bundle_integrity": {
                "ok": True,
                "input_cases_integrity": True,
                "prediction_schema_integrity": True,
                "corpus_integrity": True,
            },
            "shared_generation_identity": {"ok": True},
            "ekell_prompt_bundle_valid": True,
            "methods": {},
        },
    )
    monkeypatch.setattr(suite, "_publish_temp_directory", _publish_temp)

    def _fake_pipeline(prediction_input, *, config, llm, runtime=None):
        from external_baselines.common.decision_output import decision_output_to_legacy_row, parse_decision_output

        parsed = parse_decision_output(
            _valid_payload(),
            case_id=prediction_input["case_id"],
            method_id="direct_llm",
            strict=True,
        )
        return decision_output_to_legacy_row(parsed)

    monkeypatch.setattr(suite, "build_llm_client", lambda _c: object())
    monkeypatch.setattr(suite, "resolve_pipeline", lambda _mid: _fake_pipeline)
    monkeypatch.setattr(suite, "prepare_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "close_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(
        suite,
        "collect_method_runtime_evidence",
        lambda **kwargs: RuntimeEvidence(
            method_id=kwargs["method_id"],
            llm_provider="local_contract_provider",
            llm_model="contract-generation-v1",
            llm_model_version="v1",
            llm_temperature=0.0,
            llm_top_p=1.0,
            llm_max_tokens=1024,
            llm_seed=20260710,
            llm_enable_thinking=False,
            llm_is_smoke=False,
            llm_initialized=True,
            index_loaded=kwargs["method_id"] in {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"},
            index_built_during_run=False,
            actual_embedding_used=True,
            smoke_fallback_used=False,
            dense_dependency_actual_embedding_used=True,
            dense_dependency_smoke_fallback_used=False,
            index_checksum="same",
            dense_dependency_index_checksum="same",
        ),
    )

    with pytest.raises(FormalRunFailed):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path, n_cases=1),
            prediction_dir=pred_dir,
            decision_dir=dec_dir,
            execution_stage="formal",
            experiment_manifest=tmp_path / "manifest.yaml",
        )
    assert (pred_dir / "old.jsonl").is_file()
    assert (dec_dir / "old.jsonl").is_file()
    marker = json.loads((tmp_path / "FORMAL_RUN_FAILED.json").read_text(encoding="utf-8"))
    assert marker["rollback_succeeded"] is True


def test_formal_decision_suite_end_to_end_with_real_manifest_fixture(tmp_path, monkeypatch):
    """Offline formal E2E using a real non-.example manifest and freeze fixture."""
    from scripts import run_decision_comparison_suite as suite

    bundle_dir = _make_runner_bundle(tmp_path, n_cases=2)
    bundle = _finalize_bundle_checksums(bundle_dir)
    dense_idx = _build_fake_dense_index(tmp_path)
    ekell_idx = _build_fake_ekell_index(tmp_path)

    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: local_contract_provider\n  model: contract-generation-v1\n"
        "  model_version: v1\n  temperature: 0.0\n  top_p: 1.0\n  max_tokens: 1024\n"
        "  seed: 20260710\n  enable_thinking: false\n  api_key_env: LOCAL_CONTRACT_API_KEY\n",
        encoding="utf-8",
    )
    dense_block = (
        f"dense_rag:\n  backend: text2vec\n  model_name: fake/bge\n  model_version: v-test\n"
        f"  dimension: 8\n  normalize_embeddings: true\n  index_path: {dense_idx.as_posix()}\n"
        f"  reject_smoke: true\n"
        f"hybrid_rag:\n  reject_smoke: true\n  top_k: 3\n  candidate_pool: 5\n"
    )
    ekell_block = (
        f"ekell_vector:\n  backend: text2vec\n  model_name: fake/bge\n  model_version: v-test\n"
        f"  dimension: 8\n  normalize_embeddings: true\n  index_path: {ekell_idx.as_posix()}\n"
        f"  reject_smoke: true\n"
        f"ekell_style:\n  prompt_dir: {(ROOT / 'configs/prompts/controlled').as_posix()}\n"
    )
    method_cfgs = {}
    for mid in [
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ]:
        path = tmp_path / f"{mid}_cfg.yaml"
        extra = ""
        if mid in {"dense_rag", "hybrid_rag"}:
            extra = dense_block
        if mid == "ekell_style_controlled_shared_llm":
            extra = ekell_block
        path.write_text(
            f"execution_stage: formal\npaper_final: true\n"
            f"paths:\n  corpus_dir: {bundle['corpus_dir']}\n{extra}",
            encoding="utf-8",
        )
        method_cfgs[mid] = path

    identity = _frozen_runner_bundle_identity(bundle)
    freeze_path = tmp_path / "freeze_manifest.json"
    dev_evidence = tmp_path / "selected_dev_run.json"
    dev_evidence.write_text('{"selected": true}\n', encoding="utf-8")
    freeze_path.write_text(
        json.dumps(
            {
                "freeze_status": "frozen",
                "selected_dev_run_evidence": dev_evidence.as_posix(),
                "runner_bundle": identity,
            }
        ),
        encoding="utf-8",
    )

    exp = tmp_path / "formal_manifest.yaml"
    exp.write_text(
        "\n".join(
            [
                "experiment_id: formal_offline_e2e",
                "schema_version: firebench-interop-v1",
                f"shared_model_config: {shared.as_posix()}",
                "base_config: configs/default.yaml",
                "freeze_status: frozen",
                "paper_final: true",
                f"bundle: {bundle_dir.as_posix()}",
                f"freeze_manifest: {freeze_path.as_posix()}",
                "require_bundle_checksum: true",
                "require_external_schema: true",
                "require_complete_case_match: true",
                "fail_on_schema_error: true",
                "fail_on_duplicate_case_id: true",
                "fail_on_missing_case: true",
                "fail_on_extra_case: true",
                "main_table_methods:",
                "  - direct_llm",
                "  - bm25_rag",
                "  - ekell_style_controlled_shared_llm",
                "comparison_suite_methods:",
                "  - direct_llm",
                "  - bm25_rag",
                "  - dense_rag",
                "  - hybrid_rag",
                "  - ekell_style_controlled_shared_llm",
                "methods:",
                *[
                    line
                    for mid, cfg in method_cfgs.items()
                    for line in (
                        f"  - method_id: {mid}",
                        f"    config: {cfg.as_posix()}",
                        "    enabled: true",
                    )
                ],
            ]
        ),
        encoding="utf-8",
    )

    class ContractLLM:
        provider = "local_contract_provider"
        model = "contract-generation-v1"
        model_version = "v1"

        def usage_snapshot(self):
            from external_baselines.common.llm_client import TokenUsage

            return TokenUsage()

        def usage_delta(self, _before):
            from external_baselines.common.llm_client import TokenUsage

            return TokenUsage()

    def _fake_pipeline(prediction_input, *, config, llm, runtime=None):
        from external_baselines.common.decision_output import decision_output_to_legacy_row, parse_decision_output

        parsed = parse_decision_output(
            _valid_payload(),
            case_id=prediction_input["case_id"],
            method_id="direct_llm",
            strict=True,
        )
        return decision_output_to_legacy_row(parsed)

    def _runtime_evidence(**kwargs):
        mid = kwargs["method_id"]
        needs_index = mid in {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"}
        return RuntimeEvidence(
            method_id=mid,
            llm_provider="local_contract_provider",
            llm_model="contract-generation-v1",
            llm_model_version="v1",
            llm_temperature=0.0,
            llm_top_p=1.0,
            llm_max_tokens=1024,
            llm_seed=20260710,
            llm_enable_thinking=False,
            llm_is_smoke=False,
            llm_initialized=True,
            index_loaded=needs_index,
            index_built_during_run=False,
            actual_embedding_used=True if needs_index else None,
            smoke_fallback_used=False if needs_index else None,
            dense_dependency_actual_embedding_used=True if mid == "hybrid_rag" else None,
            dense_dependency_smoke_fallback_used=False if mid == "hybrid_rag" else None,
            index_checksum="same" if needs_index else None,
            dense_dependency_index_checksum="same" if mid == "hybrid_rag" else None,
        )

    monkeypatch.setenv("LOCAL_CONTRACT_API_KEY", "offline-contract-key")
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "build_llm_client", lambda _c: ContractLLM())
    monkeypatch.setattr(suite, "resolve_pipeline", lambda _mid: _fake_pipeline)
    monkeypatch.setattr(suite, "prepare_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "close_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "collect_method_runtime_evidence", _runtime_evidence)

    pred_dir = tmp_path / "published" / "predictions"
    dec_dir = tmp_path / "published" / "decisions"
    summary = run_decision_suite(
        runner_bundle=bundle_dir,
        prediction_dir=pred_dir,
        decision_dir=dec_dir,
        execution_stage="formal",
        experiment_manifest=exp,
    )
    assert summary["formal_compliance"]["pre_publish_compliance_passed"] is True
    assert summary["formal_compliance"]["transactional_publish_complete"] is True
    assert summary["formal_compliance"]["formal_result"] is True
    assert len(list(pred_dir.glob("*.jsonl"))) == 5
    assert len(summary["method_summaries"]) == 5
    assert not (tmp_path / "published" / "FORMAL_RUN_FAILED.json").exists()
    assert not any(tmp_path.glob(".formal_tmp_*"))
    assert not any(tmp_path.rglob("*.bak"))


def test_generated_output_files_are_not_tracked():
    import subprocess

    tracked = subprocess.check_output(
        ["git", "ls-files", "outputs/"],
        text=True,
        cwd=ROOT,
    ).splitlines()
    allowed = {"outputs/.gitkeep", "outputs/README.md"}
    unexpected = [path for path in tracked if path not in allowed]
    assert unexpected == []


# --- Round-3: exit semantics, transaction rollback, checksum semantics, full guard E2E ---


def test_formal_pre_publish_failure_raises_formal_run_failed(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {
            "ok": True,
            "runner_bundle_integrity": {"ok": True, "input_cases_integrity": True, "prediction_schema_integrity": True, "corpus_integrity": True},
            "shared_generation_identity": {"ok": True},
            "ekell_prompt_bundle_valid": True,
            "methods": {},
        },
    )

    def _fake_pipeline(prediction_input, *, config, llm, runtime=None):
        from external_baselines.common.decision_output import decision_output_to_legacy_row, parse_decision_output

        parsed = parse_decision_output(_valid_payload(), case_id=prediction_input["case_id"], method_id="direct_llm", strict=True)
        return decision_output_to_legacy_row(parsed)

    monkeypatch.setattr(suite, "build_llm_client", lambda _c: object())
    monkeypatch.setattr(suite, "resolve_pipeline", lambda _mid: _fake_pipeline)
    monkeypatch.setattr(suite, "prepare_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "close_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(
        suite,
        "collect_method_runtime_evidence",
        lambda **kwargs: RuntimeEvidence(method_id=kwargs["method_id"], llm_is_smoke=True, llm_initialized=True),
    )
    with pytest.raises(FormalRunFailed, match="Pre-publish compliance"):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path, n_cases=1),
            prediction_dir=tmp_path / "pred",
            decision_dir=tmp_path / "dec",
            execution_stage="formal",
            experiment_manifest=tmp_path / "manifest.yaml",
        )


def test_formal_cli_exits_nonzero_on_pre_publish_failure(tmp_path, monkeypatch):
    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_decision_comparison_suite.py"),
        "--runner-bundle",
        str(_make_runner_bundle(tmp_path)),
        "--prediction-dir",
        str(tmp_path / "pred"),
        "--decision-dir",
        str(tmp_path / "dec"),
        "--execution-stage",
        "formal",
        "--experiment-manifest",
        str(tmp_path / "missing_manifest.yaml"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0
    assert '"ok": false' in result.stderr.lower() or "formal" in result.stderr.lower()


def test_dry_run_cli_can_exit_zero_with_formal_result_false(tmp_path):
    import subprocess
    import sys

    bundle = _make_runner_bundle(tmp_path, n_cases=1)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_decision_comparison_suite.py"),
        "--runner-bundle",
        str(bundle),
        "--prediction-dir",
        str(tmp_path / "pred"),
        "--decision-dir",
        str(tmp_path / "dec"),
        "--execution-stage",
        "dry_run",
        "--limit",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0
    assert '"formal_result": false' in result.stdout.lower()


def test_publish_failure_without_existing_targets_removes_new_predictions(tmp_path):
    from scripts.run_decision_comparison_suite import (
        FormalPublishError,
        publish_formal_artifacts_transactionally,
    )

    pred_final = tmp_path / "pred"
    dec_final = tmp_path / "dec"
    pred_temp = tmp_path / "tpred"
    dec_temp = tmp_path / "tdec"
    pred_temp.mkdir()
    dec_temp.mkdir()
    (pred_temp / "new.jsonl").write_text("{}\n", encoding="utf-8")
    (dec_temp / "direct_llm").mkdir()
    calls = {"n": 0}
    orig_publish = __import__("scripts.run_decision_comparison_suite", fromlist=["_publish_temp_directory"])._publish_temp_directory

    def _fail_second(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("decisions publish failed")
        return orig_publish(src, dst)

    import scripts.run_decision_comparison_suite as suite

    suite._publish_temp_directory = _fail_second
    try:
        with pytest.raises(FormalPublishError):
            publish_formal_artifacts_transactionally(
                temp_prediction_dir=pred_temp,
                final_prediction_dir=pred_final,
                temp_decision_dir=dec_temp,
                final_decision_dir=dec_final,
            )
    finally:
        suite._publish_temp_directory = orig_publish
    assert not pred_final.exists()
    assert not dec_final.exists()


def test_publish_failure_with_existing_targets_restores_both(tmp_path):
    from scripts.run_decision_comparison_suite import FormalPublishError, publish_formal_artifacts_transactionally

    pred_final = tmp_path / "pred"
    dec_final = tmp_path / "dec"
    pred_final.mkdir()
    dec_final.mkdir()
    (pred_final / "old.jsonl").write_text('{"old": true}\n', encoding="utf-8")
    (dec_final / "old.jsonl").write_text('{"old": true}\n', encoding="utf-8")
    pred_temp = tmp_path / "tpred"
    dec_temp = tmp_path / "tdec"
    pred_temp.mkdir()
    dec_temp.mkdir()
    calls = {"n": 0}
    import scripts.run_decision_comparison_suite as suite

    orig_publish = suite._publish_temp_directory

    def _fail_second(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("decisions publish failed")
        return orig_publish(src, dst)

    suite._publish_temp_directory = _fail_second
    try:
        with pytest.raises(FormalPublishError):
            publish_formal_artifacts_transactionally(
                temp_prediction_dir=pred_temp,
                final_prediction_dir=pred_final,
                temp_decision_dir=dec_temp,
                final_decision_dir=dec_final,
            )
    finally:
        suite._publish_temp_directory = orig_publish
    assert (pred_final / "old.jsonl").is_file()
    assert (dec_final / "old.jsonl").is_file()


def test_second_backup_failure_restores_first_backup(tmp_path, monkeypatch):
    from scripts.run_decision_comparison_suite import FormalPublishError, publish_formal_artifacts_transactionally

    pred_final = tmp_path / "pred"
    dec_final = tmp_path / "dec"
    pred_final.mkdir()
    dec_final.mkdir()
    (pred_final / "old.jsonl").write_text('{"old": true}\n', encoding="utf-8")
    (dec_final / "old.jsonl").write_text('{"old": true}\n', encoding="utf-8")
    calls = {"n": 0}
    import scripts.run_decision_comparison_suite as suite

    orig_prepare = suite._prepare_target_backup

    def _fail_second(state):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("decisions backup failed")
        return orig_prepare(state)

    monkeypatch.setattr(suite, "_prepare_target_backup", _fail_second)
    with pytest.raises(FormalPublishError):
        publish_formal_artifacts_transactionally(
            temp_prediction_dir=tmp_path / "tpred",
            final_prediction_dir=pred_final,
            temp_decision_dir=tmp_path / "tdec",
            final_decision_dir=dec_final,
        )
    assert (pred_final / "old.jsonl").is_file()
    assert (dec_final / "old.jsonl").is_file()
    assert not any(tmp_path.glob("*.bak"))


def test_freeze_records_producer_and_consumer_separately(tmp_path):
    from external_baselines.common.freeze_manifest import build_freeze_manifest_payload

    manifest = tmp_path / "m.yaml"
    manifest.write_text("experiment_id: x\nshared_model_config: configs/deterministic_heuristic_smoke.yaml\n", encoding="utf-8")
    payload = build_freeze_manifest_payload(
        experiment_manifest_path=manifest,
        experiment_raw={"shared_model_config": "configs/deterministic_heuristic_smoke.yaml", "methods": []},
        selected_dev_run=tmp_path / "dev.json",
        producer_declared_checksum="a" * 64,
        consumer_computed_hash="b" * 64,
        input_cases_sha256="c" * 64,
        corpus_checksum="d" * 64,
        schema_checksum="e" * 64,
    )
    block = payload["runner_bundle"]
    assert block["producer_declared_checksum"] == "a" * 64
    assert block["consumer_computed_hash"] == "b" * 64
    assert block["producer_declared_checksum"] != block["consumer_computed_hash"]


def test_different_valid_producer_and_consumer_values_pass(tmp_path):
    from external_baselines.common.bundle_integrity import validate_formal_runner_bundle_integrity

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    producer = "d" * 64
    bundle["producer_declared_checksum"] = producer
    bundle["bundle_checksum"] = producer
    frozen = _frozen_runner_bundle_identity(bundle)
    assert frozen["producer_declared_checksum"] != frozen["consumer_computed_hash"]
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen)
    assert result["ok"] is True
    assert result["producer_checksum_match"] is True
    assert result["consumer_hash_match"] is True


def test_legacy_ambiguous_bundle_checksum_rejected_in_formal(tmp_path):
    from external_baselines.common.bundle_integrity import (
        extract_frozen_runner_bundle_identity,
        validate_formal_runner_bundle_integrity,
    )

    bundle_dir = _make_runner_bundle(tmp_path)
    bundle = _finalize_bundle_checksums(bundle_dir)
    frozen = extract_frozen_runner_bundle_identity(
        {"runner_bundle": {"bundle_checksum": "abc", "input_cases_sha256": "x" * 64}},
        formal=True,
    )
    result = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen)
    assert result["ok"] is False
    assert "legacy_ambiguous_bundle_checksum_not_allowed" in result["errors"]


def test_publish_success_includes_final_suite_summary(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    pred_dir = tmp_path / "pred"
    dec_dir = tmp_path / "dec"
    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {
            "ok": True,
            "runner_bundle_integrity": {"ok": True, "input_cases_integrity": True, "prediction_schema_integrity": True, "corpus_integrity": True},
            "shared_generation_identity": {"ok": True},
            "ekell_prompt_bundle_valid": True,
            "methods": {},
        },
    )

    def _runtime_evidence(**kwargs):
        mid = kwargs["method_id"]
        needs_index = mid in {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"}
        return RuntimeEvidence(
            method_id=mid,
            llm_is_smoke=False,
            llm_initialized=True,
            llm_provider="local_contract_provider",
            llm_model="contract-generation-v1",
            llm_model_version="v1",
            llm_temperature=0.0,
            llm_top_p=1.0,
            llm_max_tokens=1024,
            llm_seed=20260710,
            llm_enable_thinking=False,
            index_loaded=needs_index,
            index_built_during_run=False,
            actual_embedding_used=True if needs_index else None,
            smoke_fallback_used=False if needs_index else None,
            dense_dependency_actual_embedding_used=True if mid == "hybrid_rag" else None,
            dense_dependency_smoke_fallback_used=False if mid == "hybrid_rag" else None,
            index_checksum="same" if needs_index else None,
            dense_dependency_index_checksum="same" if mid == "hybrid_rag" else None,
        )

    def _fake_pipeline(prediction_input, *, config, llm, runtime=None):
        from external_baselines.common.decision_output import decision_output_to_legacy_row, parse_decision_output

        parsed = parse_decision_output(_valid_payload(), case_id=prediction_input["case_id"], method_id="direct_llm", strict=True)
        return decision_output_to_legacy_row(parsed)

    monkeypatch.setattr(suite, "build_llm_client", lambda _c: object())
    monkeypatch.setattr(suite, "resolve_pipeline", lambda _mid: _fake_pipeline)
    monkeypatch.setattr(suite, "prepare_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "close_method_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(suite, "collect_method_runtime_evidence", _runtime_evidence)
    run_decision_suite(
        runner_bundle=_make_runner_bundle(tmp_path, n_cases=1),
        prediction_dir=pred_dir,
        decision_dir=dec_dir,
        execution_stage="formal",
        experiment_manifest=tmp_path / "manifest.yaml",
    )
    assert (dec_dir / "suite_summary.json").is_file()
    published = json.loads((dec_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert published["formal_compliance"]["formal_result"] is True


def test_failure_cleanup_does_not_recreate_temp_directory(tmp_path, monkeypatch):
    from scripts import run_decision_comparison_suite as suite

    monkeypatch.setattr(suite, "validate_decision_suite_execution", lambda **kwargs: None)
    monkeypatch.setattr(suite, "validate_formal_method_configs", lambda **kwargs: {})
    monkeypatch.setattr(
        suite,
        "preflight_decision_suite",
        lambda **kwargs: {"ok": True, "runner_bundle_integrity": {"ok": True}, "shared_generation_identity": {"ok": True}, "ekell_prompt_bundle_valid": True, "methods": {}},
    )
    monkeypatch.setattr(suite, "build_llm_client", lambda _c: object())
    monkeypatch.setattr(suite, "resolve_pipeline", lambda _mid: (_ for _ in ()).throw(RuntimeError("fail")))
    dec_dir = tmp_path / "dec"
    with pytest.raises(FormalRunFailed):
        run_decision_suite(
            runner_bundle=_make_runner_bundle(tmp_path, n_cases=1),
            prediction_dir=tmp_path / "pred",
            decision_dir=dec_dir,
            execution_stage="formal",
            experiment_manifest=tmp_path / "manifest.yaml",
        )
    assert not any(tmp_path.glob(".formal_tmp_*"))
    assert not (dec_dir / "suite_summary.json").exists()
