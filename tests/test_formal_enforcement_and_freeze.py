"""Formal runner validation, freeze completeness, and comparison formal rules."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from external_baselines.common.checksums import sha256_file
from external_baselines.common.formal_config_validator import FormalConfigError, validate_experiment_manifest
from external_baselines.common.freeze_manifest import (
    build_freeze_manifest_payload,
    prompt_tree_checksum,
    validate_freeze_manifest,
    validate_frozen_runtime_inputs,
)
from external_baselines.common.path_resolution import PathContext, resolve_path_reference
from external_baselines.method_registry import comparison_suite_methods

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKSUM = object()


def _write_minimal_manifest(tmp_path: Path, *, freeze_status: str = "provisional", freeze_manifest=None) -> Path:
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
        encoding="utf-8",
    )
    method = tmp_path / "direct.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    payload = {
        "schema_version": "firebench-interop-v1",
        "experiment_id": "t",
        "track": "A_shared_outcome",
        "run_mode": "formal",
        "paper_final": True,
        "freeze_status": freeze_status,
        "freeze_manifest": freeze_manifest,
        "bundle": "bundle",
        "base_config": str(ROOT / "configs/default.yaml"),
        "shared_model_config": str(shared),
        "require_bundle_checksum": True,
        "require_external_schema": True,
        "require_complete_case_match": True,
        "fail_on_schema_error": True,
        "fail_on_duplicate_case_id": True,
        "fail_on_missing_case": True,
        "fail_on_extra_case": True,
        "main_table_methods": ["direct_llm"],
        "methods": [{"method_id": "direct_llm", "config": str(method), "enabled": True}],
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_formal_runner_calls_formal_validator(tmp_path: Path) -> None:
    import scripts.run_interop_baselines as rib

    called = {}

    def fake_validate(path, **kwargs):
        called["stage"] = kwargs.get("validation_stage")
        called["method_set"] = kwargs.get("method_set")
        return {"valid": True}

    with patch.object(rib, "validate_experiment_manifest", side_effect=fake_validate), patch(
        "external_baselines.common.execution_lock.assert_execution_allowed",
        side_effect=SystemExit("stopped-after-validate"),
    ), patch.object(
        rib, "load_experiment_manifest", return_value={"bundle": None, "paper_final": True, "freeze_status": "frozen"}
    ):
        with pytest.raises(SystemExit, match="stopped-after-validate"):
            rib.main(
                [
                    "--execution-stage",
                    "formal",
                    "--method-set",
                    "comparison_suite",
                    "--experiment-manifest",
                    str(tmp_path / "x.yaml"),
                    "--bundle",
                    str(tmp_path / "b"),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                ]
            )
    assert called.get("stage") == "formal"
    assert called.get("method_set") == "comparison_suite"


def test_dry_run_runner_calls_dry_run_validator(tmp_path: Path) -> None:
    import scripts.run_interop_baselines as rib

    called = {}

    def fake_validate(path, **kwargs):
        called["stage"] = kwargs.get("validation_stage")
        return {"valid": True}

    with patch.object(rib, "validate_experiment_manifest", side_effect=fake_validate), patch(
        "external_baselines.common.execution_lock.assert_execution_allowed",
        side_effect=SystemExit("stopped"),
    ), patch.object(
        rib,
        "load_experiment_manifest",
        return_value={"bundle": None, "paper_final": True, "freeze_status": "provisional"},
    ):
        with pytest.raises(SystemExit, match="stopped"):
            rib.main(
                [
                    "--execution-stage",
                    "dry_run",
                    "--experiment-manifest",
                    str(tmp_path / "x.yaml"),
                    "--bundle",
                    str(tmp_path / "b"),
                    "--limit",
                    "1",
                    "--output",
                    str(tmp_path / "out.jsonl"),
                ]
            )
    assert called.get("stage") == "dry_run"


def test_formal_runner_stops_before_runtime_on_validation_failure(tmp_path: Path) -> None:
    import scripts.run_interop_baselines as rib

    with patch.object(
        rib,
        "validate_experiment_manifest",
        side_effect=FormalConfigError("boom"),
    ), patch.object(rib, "generate_predictions") as gen, patch.object(
        rib, "load_experiment_manifest", return_value={"bundle": "b", "paper_final": True}
    ):
        with pytest.raises(SystemExit):
            rib.main(
                [
                    "--execution-stage",
                    "formal",
                    "--experiment-manifest",
                    str(tmp_path / "x.yaml"),
                    "--bundle",
                    str(tmp_path / "b"),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                ]
            )
        gen.assert_not_called()


def test_legacy_formal_records_local_schema_snapshot_mismatch(tmp_path: Path) -> None:
    import scripts.run_interop_baselines as rib

    run_manifest = tmp_path / "run_manifest.json"
    report_path = tmp_path / "interop_bundle_report.json"
    output = tmp_path / "predictions.jsonl"
    legacy_output = tmp_path / "legacy.jsonl"
    bundle_schema_sha = "b" * 64

    experiment = {
        "experiment_id": "t",
        "manifest_path": str(tmp_path / "exp.yaml"),
        "bundle": str(tmp_path / "bundle"),
        "paper_final": True,
        "freeze_status": "frozen",
        "output": str(output),
        "legacy_output": str(legacy_output),
        "run_manifest": str(run_manifest),
        "expected_bundle_checksum": None,
    }
    method_entries = [{"method_id": "direct_llm", "config": None, "enabled": True}]
    method_config = {
        "paper_final": True,
        "llm": {"provider": "openai_compatible", "model": "m", "model_version": "v"},
    }
    bundle = {
        "bundle_root": str(tmp_path / "bundle"),
        "scenarios_path": str(tmp_path / "cases.jsonl"),
        "prediction_schema_path": str(rib.SCHEMA_PATH),
        "prediction_schema_sha256": bundle_schema_sha,
        "producer_declared_checksum": "p" * 64,
        "consumer_computed_bundle_hash": "c" * 64,
    }
    row = {
        "scenario_id": "case-1",
        "method": "direct_llm",
        "situation_summary": "Smoke reported.",
        "key_risks": ["smoke_detected"],
        "recommended_actions": [{"action_id": "prepare_respiratory_protection", "text": "Prepare SCBA", "priority": "high"}],
        "blocked_or_unsafe_actions": ["BLOCK_ENTRY_WITHOUT_RESPIRATORY_PROTECTION"],
        "missing_confirmations": ["smoke_level"],
        "citations": [],
        "final_decision_gate": "await_human_confirmation",
        "latency_sec": 0.1,
        "method_specific": {"runtime": {"llm_calls": 0, "token_usage": {}, "cost": None}},
    }

    with patch.object(rib, "validate_experiment_manifest", return_value={"valid": True}), patch(
        "external_baselines.common.execution_lock.assert_execution_allowed",
        return_value={"paper_valid": True, "execution_lock_overridden": False},
    ), patch.object(rib, "load_experiment_manifest", return_value=experiment), patch.object(
        rib, "assert_no_evaluator_bundle_access"
    ), patch.object(rib, "load_runner_bundle", return_value=bundle), patch.object(
        rib, "validate_bundle_checksum", return_value={"ok": True}
    ), patch.object(rib, "enabled_methods", return_value=method_entries), patch.object(
        rib, "build_method_config", return_value=method_config
    ), patch.object(rib, "assert_paper_final_allowed"), patch.object(
        rib, "validate_cross_method_fairness", return_value={"ok": True}
    ), patch.object(rib, "load_scenarios", return_value=[{"case_id": "case-1"}]), patch.object(
        rib, "generate_predictions", return_value=[row]
    ), patch.object(rib, "_shared_embedding_snapshot", return_value={"backend": None}), patch.object(
        rib, "_index_checksum_snapshot", return_value={}
    ), patch.object(rib, "validate_interop_record", return_value=[]):
        rib.main(
            [
                "--execution-stage",
                "formal",
                "--experiment-manifest",
                str(tmp_path / "exp.yaml"),
                "--bundle",
                str(tmp_path / "bundle"),
                "--output",
                str(output),
                "--legacy-output",
                str(legacy_output),
                "--manifest",
                str(run_manifest),
            ]
        )

    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert manifest["schema_authority"] == "runner_bundle"
    assert manifest["local_schema_snapshot_authoritative"] is False
    assert manifest["local_schema_snapshot_match"] is False
    assert report["bundle"]["schema_authority"] == "runner_bundle"
    assert report["bundle"]["local_schema_snapshot_authoritative"] is False
    assert report["bundle"]["local_schema_snapshot_match"] is False


def test_formal_runner_rejects_provisional(tmp_path: Path) -> None:
    path = _write_minimal_manifest(tmp_path, freeze_status="provisional")
    with pytest.raises(FormalConfigError, match="freeze_status=frozen"):
        validate_experiment_manifest(path, validation_stage="formal")


def test_formal_runner_rejects_freeze_checksum_mismatch(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text('{"ok":true}\n', encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "freeze_status": "frozen",
                "selected_dev_run_evidence": str(evidence),
                "experiment_manifest_sha256": "deadbeef",
                "embedding": {"model_version": "rev1"},
            }
        ),
        encoding="utf-8",
    )
    path = _write_minimal_manifest(tmp_path, freeze_status="frozen", freeze_manifest=str(freeze))
    with pytest.raises(FormalConfigError, match="experiment_manifest_sha256|freeze_manifest"):
        validate_experiment_manifest(path, validation_stage="formal")


def test_formal_runner_rejects_bundle_checksum_mismatch_before_llm(tmp_path: Path) -> None:
    freeze = {
        "freeze_status": "frozen",
        "runner_bundle_checksum": "expected-bundle",
        "embedding": {"model_version": "rev1"},
    }
    with pytest.raises(FormalConfigError, match="runner_bundle_checksum"):
        validate_frozen_runtime_inputs(
            freeze,
            bundle={"consumer_computed_bundle_hash": "other-bundle", "producer_declared_checksum": "other-bundle"},
        )


def test_freeze_validates_all_method_config_hashes(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text("llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n", encoding="utf-8")
    methods = {}
    for mid in comparison_suite_methods():
        p = tmp_path / f"{mid}.yaml"
        p.write_text(f"method_id: {mid}\n", encoding="utf-8")
        methods[mid] = str(p)
    evidence = tmp_path / "selected.json"
    evidence.write_text('{"selected":true}\n', encoding="utf-8")
    exp = tmp_path / "exp.yaml"
    exp.write_text("experiment_id: t\nshared_model_config: shared.yaml\n", encoding="utf-8")
    payload = build_freeze_manifest_payload(
        experiment_manifest_path=exp,
        experiment_raw={"shared_model_config": str(shared)},
        selected_dev_run=resolve_path_reference(
            evidence.name,
            context=PathContext(repository_root=tmp_path),
            policy="repository_relative",
            expected_kind="file",
        ),
        method_config_paths=methods,
    )
    assert set(payload["method_config_sha256"]) == set(comparison_suite_methods())
    assert all(payload["method_config_sha256"].values())


def test_freeze_validates_bundle_checksum(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = {
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "runner_bundle_checksum": "abc",
        "embedding": {"model_version": "rev1"},
    }
    with pytest.raises(FormalConfigError, match="runner_bundle_checksum"):
        validate_frozen_runtime_inputs(freeze, bundle={"consumer_computed_bundle_hash": "xyz"})


def test_freeze_validates_corpus_checksum(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = {
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "corpus_checksum": "c" * 64,
        "embedding": {"model_version": "rev1"},
    }
    with pytest.raises(FormalConfigError, match="corpus_checksum"):
        validate_frozen_runtime_inputs(
            freeze,
            bundle={"corpus_manifest": {"aggregate_sha256": "d" * 64}},
        )


def test_freeze_validates_schema_checksum(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = {
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "prediction_schema_checksum": "s1",
        "embedding": {"model_version": "rev1"},
    }
    with pytest.raises(FormalConfigError, match="prediction_schema_checksum|schema"):
        validate_frozen_runtime_inputs(freeze, bundle={"prediction_schema_sha256": "s2"})


def test_freeze_validates_llm_identity(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = {
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "llm": {"provider": "siliconflow", "model": "m", "model_version": "v1"},
        "embedding": {"model_version": "rev1"},
    }
    with pytest.raises(FormalConfigError, match="llm"):
        validate_frozen_runtime_inputs(
            freeze,
            method_configs={
                "direct_llm": {"llm": {"provider": "siliconflow", "model": "m", "model_version": "v2"}}
            },
        )


def test_freeze_validates_embedding_identity(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = {
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "embedding": {
            "backend": "text2vec",
            "model_name": "BAAI/bge-m3",
            "model_version": "rev1",
            "dimension": 1024,
        },
    }
    with pytest.raises(FormalConfigError, match="embedding"):
        validate_frozen_runtime_inputs(
            freeze,
            method_configs={
                "dense_rag": {
                    "dense_rag": {
                        "backend": "text2vec",
                        "model_name": "BAAI/bge-m3",
                        "model_version": "other",
                        "dimension": 1024,
                    }
                }
            },
        )


def test_freeze_validates_dense_index_checksum(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = {
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "embedding": {"model_version": "rev1"},
        "indexes": {"dense": {"index_checksum": "d1"}},
    }
    with pytest.raises(FormalConfigError, match="dense"):
        validate_frozen_runtime_inputs(
            freeze,
            loaded_index_manifests={"dense": {"index_checksum": "d2"}},
        )


def test_freeze_validates_hybrid_dependency_checksum(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = {
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "embedding": {"model_version": "rev1"},
        "indexes": {
            "dense": {"index_checksum": "d1"},
            "hybrid_dense_dependency": {"index_checksum": "d2"},
        },
    }
    with pytest.raises(FormalConfigError, match="hybrid|dense"):
        validate_frozen_runtime_inputs(
            freeze,
            loaded_index_manifests={
                "dense": {"index_checksum": "d1"},
                "hybrid_dense_dependency": {"index_checksum": "d2"},
            },
        )


def test_freeze_validates_ekell_index_checksum(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = {
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "embedding": {"model_version": "rev1"},
        "indexes": {"ekell": {"index_checksum": "e1"}},
    }
    with pytest.raises(FormalConfigError, match="ekell"):
        validate_frozen_runtime_inputs(
            freeze,
            loaded_index_manifests={"ekell": {"index_checksum": "e2"}},
        )


def test_freeze_rejects_missing_dev_evidence(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "freeze_status": "frozen",
                "selected_dev_run_evidence": "missing.json",
                "embedding": {"model_version": "rev1"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="selected_dev_run_evidence"):
        validate_freeze_manifest(freeze, experiment_manifest_path=tmp_path / "m.yaml", experiment_raw={})


def test_complete_freeze_manifest_passes(tmp_path: Path) -> None:
    from external_baselines.common.experiment_manifest import experiment_core_sha256

    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n"
        "  api_key_env: SILICONFLOW_API_KEY\n  temperature: 0.0\n  top_p: 1.0\n"
        "  max_tokens: 1024\n  seed: 1\n",
        encoding="utf-8",
    )
    method = tmp_path / "direct.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    evidence = tmp_path / "selected.json"
    evidence.write_text('{"selected": true}\n', encoding="utf-8")
    manifest = tmp_path / "m.yaml"
    payload = {
        "schema_version": "firebench-interop-v1",
        "experiment_id": "t",
        "track": "A_shared_outcome",
        "run_mode": "formal",
        "paper_final": True,
        "freeze_status": "frozen",
        "bundle": "bundle",
        "base_config": str(ROOT / "configs/default.yaml"),
        "shared_model_config": str(shared),
        "require_bundle_checksum": True,
        "require_external_schema": True,
        "require_complete_case_match": True,
        "fail_on_schema_error": True,
        "fail_on_duplicate_case_id": True,
        "fail_on_missing_case": True,
        "fail_on_extra_case": True,
        "main_table_methods": ["direct_llm"],
        "methods": [{"method_id": "direct_llm", "config": str(method), "enabled": True}],
    }
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze_body = {
        "freeze_id": "controlled_comparison_v1",
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(evidence),
        "experiment_core_sha256": experiment_core_sha256(payload),
        "selected_dev_run_evidence_sha256": sha256_file(evidence),
        "shared_model_config_sha256": sha256_file(shared),
        "method_config_sha256": {"direct_llm": sha256_file(method)},
        "prompt_tree_sha256": prompt_tree_checksum("configs/prompts/controlled"),
        "llm": {"provider": "siliconflow", "model": "m", "model_version": "v"},
        "embedding": {
            "backend": "text2vec",
            "model_name": "BAAI/bge-m3",
            "model_version": "rev1",
            "dimension": 1024,
            "normalize_embeddings": True,
        },
        "indexes": {
            "dense": {"index_checksum": "d1"},
            "hybrid_dense_dependency": {"index_checksum": "d1"},
            "ekell": {"index_checksum": "e1"},
        },
    }
    freeze.write_text(json.dumps(freeze_body), encoding="utf-8")
    payload["freeze_manifest"] = str(freeze)
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    freeze.write_text(json.dumps(freeze_body), encoding="utf-8")
    result = validate_experiment_manifest(manifest, validation_stage="formal", method_set="main_table")
    assert result["valid"] is True


def _run_create_freeze_with_patches(
    tmp_path: Path,
    monkeypatch,
    *,
    dense_normalize=True,
    validate_error=None,
    producer_checksum: str | None | object = _DEFAULT_CHECKSUM,
    consumer_hash: str | None | object = _DEFAULT_CHECKSUM,
    index_error: Exception | None = None,
    calls_out: dict | None = None,
):
    import scripts.create_freeze_manifest as cfm

    experiment_manifest = tmp_path / "experiment.yaml"
    experiment_manifest.write_text("experiment_id: freeze_cli\n", encoding="utf-8")
    evidence = tmp_path / "selected_dev.json"
    evidence.write_text('{"selected": true}\n', encoding="utf-8")
    scenarios = tmp_path / "input_cases.jsonl"
    scenarios.write_text('{"case_id":"FBPUB_000001","input":{"scenario":"smoke"}}\n', encoding="utf-8")
    output = tmp_path / "freeze.json"
    bundle_path = tmp_path / "bundle"
    corpus_dir = bundle_path / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"Entity 1"}\n',
        encoding="utf-8",
    )
    (corpus_dir / "relations.jsonl").write_text(
        '{"relation_id":"r1","name":"related_to"}\n',
        encoding="utf-8",
    )
    (corpus_dir / "triples.jsonl").write_text(
        '{"head":"e1","relation":"related_to","tail":"e1"}\n',
        encoding="utf-8",
    )
    (corpus_dir / "evidence_chunks.jsonl").write_text(
        '{"chunk_id":"c1","text":"smoke evidence","source_id":"s1"}\n',
        encoding="utf-8",
    )
    calls = calls_out if calls_out is not None else {}
    calls.update(
        {
            "validation": [],
            "formal_flags": [],
            "validated_paths": [],
            "index_validation": [],
            "events": [],
        }
    )
    resolved_consumer_hash = "c" * 64 if consumer_hash is _DEFAULT_CHECKSUM else consumer_hash
    resolved_producer_checksum = (
        resolved_consumer_hash
        if producer_checksum is _DEFAULT_CHECKSUM
        else producer_checksum
    )
    experiment = {
        "bundle": str(bundle_path),
        "raw": {
            "shared_model_config": str(tmp_path / "shared.yaml"),
            "methods": [
                {"method_id": mid, "config": str(tmp_path / f"{mid}.yaml")}
                for mid in comparison_suite_methods()
            ],
        },
    }

    def fake_validate(path, **kwargs):
        calls["events"].append("freeze_candidate")
        calls["validation"].append({"path": path, **kwargs})
        return {"valid": True}

    def fake_load_bundle(path, *, formal=False):
        calls["events"].append("bundle_load")
        calls["formal_flags"].append(formal)
        return {
            "producer_declared_checksum": resolved_producer_checksum,
            "consumer_computed_bundle_hash": resolved_consumer_hash,
            "prediction_schema_sha256": "s" * 64,
            "scenarios_path": str(scenarios),
            "input_cases_sha256": "i" * 64,
            "corpus_manifest": {"aggregate_sha256": "a" * 64},
            "corpus_dir": str(corpus_dir),
        }

    def fake_enabled_methods(_experiment, *, method_set):
        assert method_set == "comparison_suite"
        return [{"method_id": mid, "config": str(tmp_path / f"{mid}.yaml")} for mid in comparison_suite_methods()]

    def fake_build_method_config(_experiment, entry):
        mid = entry["method_id"]
        if mid == "dense_rag":
            return {
                "dense_rag": {
                    "backend": "text2vec",
                    "model_name": "fake/bge",
                    "model_version": "v-test",
                    "dimension": 8,
                    "normalize_embeddings": dense_normalize,
                    "index_path": str(tmp_path / "dense_index"),
                }
            }
        if mid == "ekell_style_controlled_shared_llm":
            return {
                "ekell_vector": {
                    "backend": "text2vec",
                    "model_name": "fake/bge",
                    "model_version": "v-test",
                    "dimension": 8,
                    "normalize_embeddings": True,
                    "index_path": str(tmp_path / "ekell_index"),
                }
            }
        return {}

    def fake_validate_freeze(path, **_kwargs):
        calls["validated_paths"].append(Path(path))
        if validate_error is not None:
            raise validate_error
        assert Path(path).is_file()
        return {"ok": True}

    def fake_validate_dense_index(index_dir, **_kwargs):
        calls["events"].append("dense_index")
        calls["index_validation"].append(("dense", Path(index_dir)))
        if index_error is not None:
            raise index_error
        return {
            "index_type": "dense_evidence_index",
            "index_dir": str(index_dir),
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "normalize_embeddings": True,
            "document_count": 1,
            "corpus_checksum": "a" * 64,
            "documents_checksum": "d" * 64,
            "documents_file_checksum": "e" * 64,
            "embeddings_checksum": "f" * 64,
            "evidence_source_checksum": "b" * 64,
            "index_checksum": "1" * 64,
            "index_manifest_sha256": "2" * 64,
            "actual_embedding_used": True,
            "smoke_fallback_used": False,
        }

    def fake_validate_ekell_index(index_dir, **_kwargs):
        calls["events"].append("ekell_index")
        calls["index_validation"].append(("ekell", Path(index_dir)))
        if index_error is not None:
            raise index_error
        return {
            "index_type": "ekell_kg_vector_index",
            "index_dir": str(index_dir),
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "normalize_embeddings": True,
            "document_count": 1,
            "kg_checksum": "3" * 64,
            "corpus_checksum": "a" * 64,
            "documents_checksum": "4" * 64,
            "documents_file_checksum": "5" * 64,
            "embeddings_checksum": "6" * 64,
            "index_checksum": "7" * 64,
            "index_manifest_sha256": "8" * 64,
            "actual_embedding_used": True,
            "smoke_fallback_used": False,
        }

    monkeypatch.setattr(cfm, "validate_experiment_manifest", fake_validate)
    monkeypatch.setattr(cfm, "load_experiment_manifest", lambda _path: experiment)
    monkeypatch.setattr(cfm, "load_runner_bundle", fake_load_bundle)
    monkeypatch.setattr(cfm, "enabled_methods", fake_enabled_methods)
    monkeypatch.setattr(cfm, "build_method_config", fake_build_method_config)
    monkeypatch.setattr(cfm, "validate_freeze_manifest", fake_validate_freeze)
    monkeypatch.setattr(cfm, "validate_dense_index_integrity_for_freeze", fake_validate_dense_index)
    monkeypatch.setattr(cfm.VectorIndex, "validate_directory_for_freeze", fake_validate_ekell_index)
    monkeypatch.setattr(cfm, "ROOT", tmp_path)
    cfm.main(
        [
            "--experiment-manifest",
            str(experiment_manifest),
            "--selected-dev-run",
            evidence.name,
            "--bundle",
            str(bundle_path),
            "--output",
            str(output),
        ]
    )
    return output, calls


def test_complete_freeze_calls_bundle_loader_with_formal_true(tmp_path: Path, monkeypatch) -> None:
    output, calls = _run_create_freeze_with_patches(tmp_path, monkeypatch)

    assert output.is_file()
    assert calls["formal_flags"] == [True]
    assert calls["validation"][0]["validation_stage"] == "freeze_candidate"
    assert calls["validation"][0]["method_set"] == "comparison_suite"
    assert calls["validation"][0]["runtime_bundle_path"]
    assert calls["events"][:4] == [
        "bundle_load",
        "freeze_candidate",
        "dense_index",
        "ekell_index",
    ]
    assert calls["validated_paths"][0].name == "freeze.json.tmp"
    assert [kind for kind, _path in calls["index_validation"]] == ["dense", "ekell"]
    assert not output.with_name(f"{output.name}.tmp").exists()


def test_complete_freeze_rejects_bundle_producer_consumer_checksum_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with pytest.raises(SystemExit, match="formal_bundle_producer_consumer_checksum_mismatch"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            producer_checksum="a" * 64,
            consumer_hash="c" * 64,
        )

    assert not (tmp_path / "freeze.json").exists()
    assert not (tmp_path / "freeze.json.tmp").exists()


def test_complete_freeze_accepts_missing_optional_producer_checksum(tmp_path: Path, monkeypatch) -> None:
    output, _calls = _run_create_freeze_with_patches(
        tmp_path,
        monkeypatch,
        producer_checksum=None,
        consumer_hash="c" * 64,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["runner_bundle"]["producer_declared_checksum"] is None
    assert payload["runner_bundle"]["producer_checksum_available"] is False


def test_complete_freeze_rejects_invalid_producer_checksum_format(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="formal_bundle_producer_checksum_invalid"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            producer_checksum="deadbeef",
            consumer_hash="c" * 64,
        )


def test_complete_freeze_requires_consumer_bundle_hash(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="formal_bundle_consumer_hash_missing"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            producer_checksum=None,
            consumer_hash=None,
        )


def test_bundle_checksum_failure_occurs_before_index_loading(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="formal_bundle_producer_consumer_checksum_mismatch"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            producer_checksum="a" * 64,
            consumer_hash="c" * 64,
            index_error=RuntimeError("index should not load"),
        )

    assert not (tmp_path / "freeze.json").exists()
    assert not (tmp_path / "freeze.json.tmp").exists()


def test_bundle_aggregate_mismatch_fails_before_dense_hashing(tmp_path: Path, monkeypatch) -> None:
    calls: dict = {}
    with pytest.raises(SystemExit, match="formal_bundle_producer_consumer_checksum_mismatch"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            producer_checksum="a" * 64,
            consumer_hash="c" * 64,
            index_error=RuntimeError("dense index should not hash"),
            calls_out=calls,
        )
    assert calls["validation"] == []
    assert calls["index_validation"] == []
    assert calls["events"] == ["bundle_load"]


def test_bundle_aggregate_mismatch_fails_before_ekell_hashing(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="formal_bundle_producer_consumer_checksum_mismatch"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            producer_checksum="a" * 64,
            consumer_hash="c" * 64,
            index_error=RuntimeError("ekell index should not hash"),
        )


def test_bundle_aggregate_mismatch_creates_no_temp_freeze(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="formal_bundle_producer_consumer_checksum_mismatch"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            producer_checksum="a" * 64,
            consumer_hash="c" * 64,
        )

    assert not (tmp_path / "freeze.json.tmp").exists()


def test_bundle_aggregate_mismatch_preserves_existing_freeze(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "freeze.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="formal_bundle_producer_consumer_checksum_mismatch"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            producer_checksum="a" * 64,
            consumer_hash="c" * 64,
        )

    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_complete_freeze_rejects_string_boolean(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="normalize_embeddings"):
        _run_create_freeze_with_patches(tmp_path, monkeypatch, dense_normalize="false")

    assert not (tmp_path / "freeze.json").exists()


def test_cross_method_normalize_embeddings_mismatch_fails_before_index_hashing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict = {}
    with pytest.raises(SystemExit, match="cross_method_normalize_embeddings_mismatch"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            dense_normalize=False,
            calls_out=calls,
        )
    assert calls["index_validation"] == []


def test_complete_freeze_does_not_leave_output_on_failure(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="Incomplete freeze manifest"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            validate_error=FormalConfigError("forced failure"),
        )

    assert not (tmp_path / "freeze.json").exists()
    assert not (tmp_path / "freeze.json.tmp").exists()


def test_complete_freeze_cleans_temp_on_validator_runtime_error(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(SystemExit, match="Complete freeze manifest generation failed"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            validate_error=RuntimeError("validator exploded"),
        )

    assert not (tmp_path / "freeze.json").exists()
    assert not (tmp_path / "freeze.json.tmp").exists()


def test_complete_freeze_preserves_existing_output_when_new_freeze_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "freeze.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="Complete freeze manifest generation failed"):
        _run_create_freeze_with_patches(
            tmp_path,
            monkeypatch,
            validate_error=RuntimeError("validator exploded"),
        )

    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert not (tmp_path / "freeze.json.tmp").exists()


def test_complete_freeze_cleans_temp_when_replace_fails(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "freeze.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")
    original_replace = Path.replace

    def fail_replace(self, target):  # noqa: ANN001
        if self.name == "freeze.json.tmp":
            raise OSError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(SystemExit, match="Complete freeze manifest generation failed"):
        _run_create_freeze_with_patches(tmp_path, monkeypatch)

    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert not (tmp_path / "freeze.json.tmp").exists()


def _complete_freeze_payload(tmp_path: Path) -> tuple[Path, dict, dict[str, str]]:
    shared = tmp_path / "shared.yaml"
    shared.write_text("llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n", encoding="utf-8")
    methods: dict[str, str] = {}
    for mid in comparison_suite_methods():
        method = tmp_path / f"{mid}.yaml"
        method.write_text(f"method_id: {mid}\n", encoding="utf-8")
        methods[mid] = str(method)
    evidence = tmp_path / "selected.json"
    evidence.write_text('{"selected": true}\n', encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    raw = {
        "shared_model_config": shared.name,
        "methods": [
            {"method_id": mid, "config": Path(path).name}
            for mid, path in methods.items()
        ],
    }
    experiment.write_text(yaml.safe_dump(raw), encoding="utf-8")
    payload = build_freeze_manifest_payload(
        experiment_manifest_path=experiment,
        experiment_raw=raw,
        selected_dev_run=resolve_path_reference(
            evidence.name,
            context=PathContext(repository_root=tmp_path),
            policy="repository_relative",
            expected_kind="file",
        ),
        producer_declared_checksum=None,
        consumer_computed_hash="c" * 64,
        input_cases_sha256="9" * 64,
        corpus_checksum="a" * 64,
        schema_checksum="e" * 64,
        method_config_paths=methods,
        embedding={
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "normalize_embeddings": True,
        },
        indexes={
            "dense": {
                "index_type": "dense_evidence_index",
                "backend": "text2vec",
                "model_name": "fake/bge",
                "model_version": "v-test",
                "dimension": 8,
                "normalize_embeddings": True,
                "document_count": 1,
                "index_checksum": "1" * 64,
                "index_manifest_sha256": "2" * 64,
                "corpus_checksum": "a" * 64,
                "documents_checksum": "b" * 64,
                "documents_file_checksum": "d" * 64,
                "embeddings_checksum": "f" * 64,
                "evidence_source_checksum": "7" * 64,
                "actual_embedding_used": True,
                "smoke_fallback_used": False,
            },
            "hybrid_dense_dependency": {
                "index_checksum": "1" * 64,
                "index_manifest_sha256": "2" * 64,
            },
            "ekell": {
                "index_type": "ekell_kg_vector_index",
                "backend": "text2vec",
                "model_name": "fake/bge",
                "model_version": "v-test",
                "dimension": 8,
                "normalize_embeddings": True,
                "document_count": 1,
                "index_checksum": "3" * 64,
                "index_manifest_sha256": "4" * 64,
                "kg_checksum": "5" * 64,
                "corpus_checksum": "a" * 64,
                "documents_checksum": "6" * 64,
                "documents_file_checksum": "7" * 64,
                "embeddings_checksum": "8" * 64,
                "actual_embedding_used": True,
                "smoke_fallback_used": False,
            },
        },
        producer_checksum_available=False,
    )
    return experiment, payload, methods


def _validate_complete_payload(
    experiment: Path,
    payload: dict,
    methods: dict[str, str] | None,
    *,
    repository_root: Path,
) -> dict:
    return validate_freeze_manifest(
        payload,
        experiment_manifest_path=experiment,
        experiment_raw=yaml.safe_load(experiment.read_text(encoding="utf-8")),
        require_complete=True,
        expected_runner_bundle_checksum="c" * 64,
        expected_corpus_checksum="a" * 64,
        expected_prediction_schema_checksum="e" * 64,
        loaded_index_manifests=payload["indexes"],
        method_config_paths=methods,
        repository_root=repository_root,
    )


def test_complete_freeze_records_portable_selected_dev_identity(tmp_path: Path) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    evidence = payload["selected_dev_run_evidence"]

    assert evidence == {
        "declared_path": "selected.json",
        "canonical_path": "selected.json",
        "path_policy": "repository_relative",
        "sha256": sha256_file(tmp_path / "selected.json"),
    }
    assert _validate_complete_payload(
        experiment,
        payload,
        methods,
        repository_root=tmp_path,
    )["ok"] is True


def test_complete_freeze_does_not_authoritatively_store_absolute_selected_dev_path(
    tmp_path: Path,
) -> None:
    _experiment, payload, _methods = _complete_freeze_payload(tmp_path)
    evidence = payload["selected_dev_run_evidence"]

    assert str(tmp_path) not in json.dumps(evidence)
    assert payload["selected_dev_run_evidence_resolved_at_freeze"] == str(
        (tmp_path / "selected.json").resolve()
    )
    assert payload["selected_dev_run_evidence_resolved_authoritative"] is False
    provenance = payload["path_provenance"]["selected_dev_evidence"]
    assert provenance["resolved_path_authoritative"] is False


def test_formal_selected_dev_identity_survives_repository_relocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import external_baselines.common.formal_config_validator as formal_validator

    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    (repo_a / "configs" / "prompts").mkdir(parents=True)
    shutil.copy2(ROOT / "configs" / "default.yaml", repo_a / "configs" / "default.yaml")
    shutil.copytree(
        ROOT / "configs" / "prompts" / "paper_fidelity",
        repo_a / "configs" / "prompts" / "paper_fidelity",
    )
    monkeypatch.setattr(formal_validator, "ROOT_REL", repo_a)
    experiment_a, payload, _methods = _complete_freeze_payload(repo_a)
    (repo_a / "freeze.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copytree(repo_a, repo_b)
    shutil.rmtree(repo_a)
    monkeypatch.setattr(formal_validator, "ROOT_REL", repo_b)

    result = validate_freeze_manifest(
        repo_b / "freeze.json",
        experiment_manifest_path=repo_b / experiment_a.name,
        experiment_raw=yaml.safe_load(
            (repo_b / experiment_a.name).read_text(encoding="utf-8")
        ),
        require_complete=True,
        expected_runner_bundle_checksum="c" * 64,
        expected_corpus_checksum="a" * 64,
        expected_prediction_schema_checksum="e" * 64,
        loaded_index_manifests=payload["indexes"],
        repository_root=repo_b,
    )

    assert result["ok"] is True


def test_formal_rejects_selected_dev_content_changed_same_relative_path(
    tmp_path: Path,
) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    (tmp_path / "selected.json").write_text('{"selected": false}\n', encoding="utf-8")

    with pytest.raises(FormalConfigError, match="selected_dev_run_evidence.sha256 mismatch"):
        _validate_complete_payload(
            experiment,
            payload,
            methods,
            repository_root=tmp_path,
        )


def test_complete_freeze_rejects_external_selected_dev_path(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    experiment, payload, methods = _complete_freeze_payload(repository)
    external = tmp_path / "external-selected.json"
    external.write_text('{"selected": true}\n', encoding="utf-8")
    payload["selected_dev_run_evidence"] = {
        "declared_path": str(external),
        "canonical_path": external.resolve().as_posix(),
        "path_policy": "absolute_external",
        "sha256": sha256_file(external),
    }

    with pytest.raises(
        FormalConfigError,
        match="selected_dev_evidence_external_path_not_portable",
    ):
        _validate_complete_payload(
            experiment,
            payload,
            methods,
            repository_root=repository,
        )


def test_complete_freeze_requires_repository_relative_selected_dev_policy(
    tmp_path: Path,
) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    payload["selected_dev_run_evidence"]["path_policy"] = "experiment_relative"

    with pytest.raises(
        FormalConfigError,
        match="selected_dev_evidence_path_policy_must_be_repository_relative",
    ):
        _validate_complete_payload(
            experiment,
            payload,
            methods,
            repository_root=tmp_path,
        )


def test_legacy_string_selected_dev_evidence_requires_regeneration(
    tmp_path: Path,
) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    payload["selected_dev_run_evidence"] = "selected.json"

    with pytest.raises(
        FormalConfigError,
        match="legacy_selected_dev_evidence_identity_requires_regeneration",
    ):
        _validate_complete_payload(
            experiment,
            payload,
            methods,
            repository_root=tmp_path,
        )


def test_selected_dev_path_cannot_escape_repository_root(tmp_path: Path) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    payload["selected_dev_run_evidence"]["canonical_path"] = "../selected.json"

    with pytest.raises(FormalConfigError, match="escapes_repository_relative_root"):
        _validate_complete_payload(
            experiment,
            payload,
            methods,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("block", "field", "match"),
    [
        ("dense", "index_checksum", "indexes.dense.index_checksum"),
        ("dense", "index_manifest_sha256", "indexes.dense.index_manifest_sha256"),
        ("dense", "documents_checksum", "indexes.dense.documents_checksum"),
        ("dense", "documents_file_checksum", "indexes.dense.documents_file_checksum"),
        ("dense", "embeddings_checksum", "indexes.dense.embeddings_checksum"),
        ("dense", "normalize_embeddings", "indexes.dense.normalize_embeddings"),
        ("dense", "model_name", "indexes.dense.model_name"),
        ("hybrid_dense_dependency", "index_checksum", "indexes.hybrid_dense_dependency.index_checksum"),
        (
            "hybrid_dense_dependency",
            "index_manifest_sha256",
            "indexes.hybrid_dense_dependency.index_manifest_sha256",
        ),
        ("ekell", "index_checksum", "indexes.ekell.index_checksum"),
        ("ekell", "index_manifest_sha256", "indexes.ekell.index_manifest_sha256"),
        ("ekell", "documents_checksum", "indexes.ekell.documents_checksum"),
        ("ekell", "documents_file_checksum", "indexes.ekell.documents_file_checksum"),
        ("ekell", "embeddings_checksum", "indexes.ekell.embeddings_checksum"),
        ("ekell", "kg_checksum", "indexes.ekell.kg_checksum"),
        ("ekell", "normalize_embeddings", "indexes.ekell.normalize_embeddings"),
    ],
)
def test_complete_freeze_requires_index_identity_fields(
    tmp_path: Path,
    block: str,
    field: str,
    match: str,
) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    payload["indexes"][block].pop(field)

    with pytest.raises(FormalConfigError, match=match):
        validate_freeze_manifest(
            payload,
            experiment_manifest_path=experiment,
            experiment_raw=yaml.safe_load(experiment.read_text(encoding="utf-8")),
            require_complete=True,
            expected_runner_bundle_checksum="c" * 64,
            expected_corpus_checksum="a" * 64,
            expected_prediction_schema_checksum="e" * 64,
            loaded_index_manifests=payload["indexes"],
            method_config_paths=methods,
            repository_root=tmp_path,
        )


def test_complete_freeze_rejects_dense_hybrid_checksum_mismatch(tmp_path: Path) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    payload["indexes"]["hybrid_dense_dependency"]["index_checksum"] = "9" * 64

    with pytest.raises(FormalConfigError, match="hybrid dense dependency checksum"):
        validate_freeze_manifest(
            payload,
            experiment_manifest_path=experiment,
            experiment_raw=yaml.safe_load(experiment.read_text(encoding="utf-8")),
            require_complete=True,
            expected_runner_bundle_checksum="c" * 64,
            expected_corpus_checksum="a" * 64,
            expected_prediction_schema_checksum="e" * 64,
            loaded_index_manifests=payload["indexes"],
            method_config_paths=methods,
            repository_root=tmp_path,
        )


def test_complete_freeze_rejects_dense_hybrid_manifest_sha_mismatch(tmp_path: Path) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    payload["indexes"]["hybrid_dense_dependency"]["index_manifest_sha256"] = "9" * 64

    with pytest.raises(FormalConfigError, match="hybrid dense dependency manifest SHA"):
        validate_freeze_manifest(
            payload,
            experiment_manifest_path=experiment,
            experiment_raw=yaml.safe_load(experiment.read_text(encoding="utf-8")),
            require_complete=True,
            expected_runner_bundle_checksum="c" * 64,
            expected_corpus_checksum="a" * 64,
            expected_prediction_schema_checksum="e" * 64,
            loaded_index_manifests=payload["indexes"],
            method_config_paths=methods,
            repository_root=tmp_path,
        )


def test_frozen_runtime_inputs_reject_dense_documents_file_checksum_mismatch(tmp_path: Path) -> None:
    _experiment, payload, _methods = _complete_freeze_payload(tmp_path)
    live = json.loads(json.dumps(payload["indexes"]))
    live["dense"]["documents_file_checksum"] = "9" * 64

    with pytest.raises(FormalConfigError, match="indexes.dense.documents_file_checksum"):
        validate_frozen_runtime_inputs(
            payload,
            loaded_index_manifests=live,
        )


def test_frozen_runtime_inputs_require_complete_live_identity(tmp_path: Path) -> None:
    _experiment, payload, _methods = _complete_freeze_payload(tmp_path)
    live = json.loads(json.dumps(payload["indexes"]))
    del live["ekell"]["embeddings_checksum"]

    with pytest.raises(
        FormalConfigError,
        match="loaded index missing indexes.ekell.embeddings_checksum",
    ):
        validate_frozen_runtime_inputs(
            payload,
            loaded_index_manifests=live,
            require_complete_indexes=True,
        )


def test_frozen_runtime_identity_report_contains_field_level_matches(tmp_path: Path) -> None:
    _experiment, payload, _methods = _complete_freeze_payload(tmp_path)

    report = validate_frozen_runtime_inputs(
        payload,
        loaded_index_manifests=payload["indexes"],
        require_complete_indexes=True,
    )

    assert report["ok"] is True
    assert report["dense"]["index_checksum_match"] is True
    assert report["dense"]["documents_file_checksum_match"] is True
    assert report["dense"]["normalize_embeddings_match"] is True
    assert report["hybrid_dense_dependency"]["index_manifest_sha256_match"] is True
    assert report["ekell"]["kg_checksum_match"] is True
    assert report["ekell"]["embeddings_checksum_match"] is True


@pytest.mark.parametrize(
    ("block", "field", "value", "match"),
    [
        ("dense", "actual_embedding_used", False, "actual_embedding_used must be true"),
        ("dense", "smoke_fallback_used", True, "smoke_fallback_used must be false"),
        ("ekell", "actual_embedding_used", False, "actual_embedding_used must be true"),
        ("ekell", "smoke_fallback_used", True, "smoke_fallback_used must be false"),
    ],
)
def test_complete_freeze_requires_real_embedding_flag_values(
    tmp_path: Path,
    block: str,
    field: str,
    value: bool,
    match: str,
) -> None:
    experiment, payload, methods = _complete_freeze_payload(tmp_path)
    payload["indexes"][block][field] = value

    with pytest.raises(FormalConfigError, match=match):
        validate_freeze_manifest(
            payload,
            experiment_manifest_path=experiment,
            experiment_raw=yaml.safe_load(experiment.read_text(encoding="utf-8")),
            require_complete=True,
            expected_runner_bundle_checksum="c" * 64,
            expected_corpus_checksum="a" * 64,
            expected_prediction_schema_checksum="e" * 64,
            method_config_paths=methods,
            repository_root=tmp_path,
        )


def test_comparison_formal_validates_dense() -> None:
    from external_baselines.common.formal_config_validator import validate_dense_config_for_real_run

    with pytest.raises(FormalConfigError):
        validate_dense_config_for_real_run(
            {"dense_rag": {"backend": "smoke_hash_embedding"}},
            allow_placeholders=False,
            validation_stage="dry_run",
        )


def test_comparison_formal_validates_hybrid() -> None:
    from external_baselines.common.formal_config_validator import validate_hybrid_config_for_real_run

    with pytest.raises(FormalConfigError):
        validate_hybrid_config_for_real_run(
            {"hybrid_rag": {"rrf_k": 0}, "dense_rag": {"backend": "text2vec"}},
            allow_placeholders=False,
            validation_stage="dry_run",
        )


def test_comparison_rejects_dense_placeholder_model_version() -> None:
    from external_baselines.common.formal_config_validator import validate_dense_config_for_real_run

    with pytest.raises(FormalConfigError, match="model_version|placeholder"):
        validate_dense_config_for_real_run(
            {
                "dense_rag": {
                    "backend": "text2vec",
                    "model_name": "BAAI/bge-m3",
                    "model_version": "REQUIRED_BEFORE_REAL_INDEX_BUILD",
                    "dimension": 1024,
                    "normalize_embeddings": True,
                    "reject_smoke": True,
                    "index_path": "outputs/indexes/dense/x",
                }
            },
            allow_placeholders=False,
            validation_stage="template",
        )


def test_comparison_rejects_hybrid_smoke_backend() -> None:
    from external_baselines.common.formal_config_validator import validate_hybrid_config_for_real_run

    with pytest.raises(FormalConfigError):
        validate_hybrid_config_for_real_run(
            {
                "hybrid_rag": {
                    "rrf_k": 60,
                    "top_k": 5,
                    "candidate_pool": 20,
                    "lexical_weight": 1.0,
                    "dense_weight": 1.0,
                    "reject_smoke": True,
                },
                "dense_rag": {
                    "backend": "smoke_hash_embedding",
                    "model_name": "x",
                    "model_version": "y",
                    "dimension": 8,
                    "normalize_embeddings": True,
                    "reject_smoke": True,
                    "index_path": "outputs/indexes/dense/x",
                },
            },
            allow_placeholders=False,
            validation_stage="template",
        )


def test_comparison_rejects_missing_method(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text("llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n", encoding="utf-8")
    method = tmp_path / "direct.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "t",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "paper_final": True,
                "freeze_status": "provisional",
                "bundle": "bundle",
                "base_config": str(ROOT / "configs/default.yaml"),
                "shared_model_config": str(shared),
                "require_bundle_checksum": True,
                "require_external_schema": True,
                "require_complete_case_match": True,
                "fail_on_schema_error": True,
                "fail_on_duplicate_case_id": True,
                "fail_on_missing_case": True,
                "fail_on_extra_case": True,
                "comparison_suite_methods": ["direct_llm", "bm25_rag"],
                "main_table_methods": ["direct_llm"],
                "methods": [{"method_id": "direct_llm", "config": str(method), "enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="comparison_suite|five|exact|missing"):
        validate_experiment_manifest(manifest, validation_stage="dry_run", method_set="comparison_suite")


def test_comparison_rejects_extra_enhanced_method(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text("llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n", encoding="utf-8")
    methods = []
    for mid in list(comparison_suite_methods()) + ["ekell_style_enhanced"]:
        p = tmp_path / f"{mid}.yaml"
        p.write_text(f"method_id: {mid}\n", encoding="utf-8")
        methods.append({"method_id": mid, "config": str(p), "enabled": True})
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "t",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "paper_final": True,
                "freeze_status": "provisional",
                "bundle": "bundle",
                "base_config": str(ROOT / "configs/default.yaml"),
                "shared_model_config": str(shared),
                "require_bundle_checksum": True,
                "require_external_schema": True,
                "require_complete_case_match": True,
                "fail_on_schema_error": True,
                "fail_on_duplicate_case_id": True,
                "fail_on_missing_case": True,
                "fail_on_extra_case": True,
                "comparison_suite_methods": list(comparison_suite_methods()) + ["ekell_style_enhanced"],
                "main_table_methods": ["direct_llm", "bm25_rag", "ekell_style_controlled_shared_llm"],
                "methods": methods,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="enhanced|exact|comparison_suite"):
        validate_experiment_manifest(manifest, validation_stage="dry_run", method_set="comparison_suite")


def test_comparison_requires_exact_five_methods() -> None:
    assert list(comparison_suite_methods()) == [
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ]


def test_hybrid_and_dense_index_identity_must_match() -> None:
    from external_baselines.common.fairness import CrossMethodFairnessError, validate_cross_method_fairness

    with pytest.raises(CrossMethodFairnessError):
        validate_cross_method_fairness(
            {
                "dense_rag": {
                    "paper_final": True,
                    "llm": {"provider": "siliconflow", "model": "m", "model_version": "v"},
                    "dense_rag": {"backend": "text2vec", "model_name": "bge", "model_version": "v1", "dimension": 8},
                    "dense_index_checksum": "aaa",
                },
                "hybrid_rag": {
                    "paper_final": True,
                    "llm": {"provider": "siliconflow", "model": "m", "model_version": "v"},
                    "dense_rag": {"backend": "text2vec", "model_name": "bge", "model_version": "v1", "dimension": 8},
                    "dense_index_checksum": "bbb",
                },
            }
        )


def _custom_prompt_freeze(tmp_path: Path) -> tuple[Path, dict, Path, dict[str, str]]:
    from external_baselines.ekell_style.prompt_identity import EKELL_REQUIRED_PROMPTS

    prompt_dir = tmp_path / "custom_prompts"
    prompt_dir.mkdir(parents=True)
    for name in EKELL_REQUIRED_PROMPTS:
        (prompt_dir / name).write_text(f"custom {name}\n", encoding="utf-8")
    shared = tmp_path / "shared_prompt.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: model\n  model_version: version\n",
        encoding="utf-8",
    )
    methods: dict[str, str] = {}
    for mid in comparison_suite_methods():
        method = tmp_path / f"prompt_{mid}.yaml"
        body: dict = {"method_id": mid}
        if mid == "ekell_style_controlled_shared_llm":
            body["ekell_style"] = {"prompt_dir": str(prompt_dir)}
        method.write_text(yaml.safe_dump(body), encoding="utf-8")
        methods[mid] = str(method)
    evidence = tmp_path / "selected_prompt.json"
    evidence.write_text('{"selected": true}\n', encoding="utf-8")
    raw = {
        "shared_model_config": str(shared),
        "methods": [
            {"method_id": mid, "config": path} for mid, path in methods.items()
        ],
    }
    experiment = tmp_path / "prompt_experiment.yaml"
    experiment.write_text(yaml.safe_dump(raw), encoding="utf-8")
    payload = build_freeze_manifest_payload(
        experiment_manifest_path=experiment,
        experiment_raw=raw,
        selected_dev_run=resolve_path_reference(
            evidence.name,
            context=PathContext(repository_root=tmp_path),
            policy="repository_relative",
            expected_kind="file",
        ),
        method_config_paths=methods,
        embedding={
            "backend": "text2vec",
            "model_name": "example/bge",
            "model_version": "v-test",
            "dimension": 8,
            "normalize_embeddings": True,
        },
    )
    return experiment, payload, prompt_dir, methods


def test_complete_freeze_uses_merged_ekell_prompt_dir(tmp_path: Path) -> None:
    _experiment, payload, prompt_dir, _methods = _custom_prompt_freeze(tmp_path)
    bundle = payload["ekell_prompt_bundle"]
    assert bundle["declared_prompt_dir"] == str(prompt_dir).replace("\\", "/")
    assert payload["prompt_tree_sha256"] == bundle["prompt_tree_sha256"]
    assert set(bundle["required_prompt_files"]) == {
        "stepwise_projection.txt",
        "stepwise_intersection.txt",
        "stepwise_union.txt",
        "stepwise_negation.txt",
        "final_kg_grounded_response.txt",
    }


def test_complete_freeze_accepts_custom_prompt_dir(tmp_path: Path) -> None:
    experiment, payload, _prompt_dir, methods = _custom_prompt_freeze(tmp_path)
    validate_freeze_manifest(
        payload,
        experiment_manifest_path=experiment,
        experiment_raw=yaml.safe_load(experiment.read_text(encoding="utf-8")),
        require_complete=False,
        method_config_paths=methods,
        repository_root=tmp_path,
    )


def test_prompt_tree_hash_includes_additional_prompt_files(tmp_path: Path) -> None:
    from external_baselines.common.path_resolution import PathContext
    from external_baselines.ekell_style.prompt_identity import validate_and_hash_prompt_bundle

    _experiment, first, prompt_dir, _methods = _custom_prompt_freeze(tmp_path)
    (prompt_dir / "additional_context.txt").write_text("extra\n", encoding="utf-8")
    second = _custom_prompt_freeze(tmp_path / "second")[1]
    modified = validate_and_hash_prompt_bundle(
        prompt_dir,
        path_context=PathContext(repository_root=Path(__file__).resolve().parents[1]),
    )
    assert modified["prompt_tree_sha256"] != first["prompt_tree_sha256"]
    assert second["prompt_tree_sha256"] == first["prompt_tree_sha256"]


def test_formal_preflight_rejects_prompt_file_modified_after_freeze(tmp_path: Path) -> None:
    from external_baselines.common.decision_suite_preflight import _validate_ekell_prompts

    _experiment, payload, prompt_dir, _methods = _custom_prompt_freeze(tmp_path)
    target = prompt_dir / "stepwise_projection.txt"
    target.write_text("modified prompt\n", encoding="utf-8")
    report = _validate_ekell_prompts(
        prompt_dir,
        freeze=payload,
        formal=True,
        declared_prompt_dir=str(prompt_dir),
    )
    assert report["ok"] is False
    assert "ekell_prompt_hash_mismatch:stepwise_projection.txt" in report["errors"]


def test_legacy_prompt_hash_only_freeze_requires_regeneration(tmp_path: Path) -> None:
    _experiment, payload, _prompt_dir, _methods = _custom_prompt_freeze(tmp_path)
    payload.pop("ekell_prompt_bundle")
    with pytest.raises(
        FormalConfigError,
        match="legacy_freeze_prompt_identity_requires_regeneration",
    ):
        validate_freeze_manifest(
            payload,
            experiment_manifest_path=tmp_path / "missing.yaml",
            experiment_raw={},
            require_complete=True,
        )
