"""Formal runner validation, freeze completeness, and comparison formal rules."""

from __future__ import annotations

import json
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
from external_baselines.method_registry import comparison_suite_methods

ROOT = Path(__file__).resolve().parents[1]


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
        selected_dev_run=evidence,
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
        "corpus_checksum": "c1",
        "embedding": {"model_version": "rev1"},
    }
    with pytest.raises(FormalConfigError, match="corpus_checksum"):
        validate_frozen_runtime_inputs(
            freeze,
            bundle={"corpus_manifest": {"aggregate_sha256": "c2"}},
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
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
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
        "experiment_manifest_sha256": sha256_file(manifest),
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
    freeze_body["experiment_manifest_sha256"] = sha256_file(manifest)
    freeze.write_text(json.dumps(freeze_body), encoding="utf-8")
    result = validate_experiment_manifest(manifest, validation_stage="formal", method_set="main_table")
    assert result["valid"] is True


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
