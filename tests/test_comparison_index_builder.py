"""Fail-closed official index-builder tests using offline persisted indexes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from external_baselines.common.checksums import sha256_file
from external_baselines.dense_rag.pipeline import build_dense_index
from external_baselines.ekell_style.embedding_backends import create_embedding_backend
from external_baselines.ekell_style.kg_loader import fire_kg_checksum, load_kg_strict
from external_baselines.ekell_style.vector_index import VectorIndex
from external_baselines.interop.bundle import (
    load_runner_bundle,
    runner_bundle_corpus_aggregate_sha256,
)
from scripts import build_comparison_indexes as builder
from tests.test_decision_comparison_suite import _make_runner_bundle
from tests.test_dense_real_index import FakeEmbeddingModel


def _shared_llm() -> dict[str, Any]:
    return {
        "llm": {
            "provider": "siliconflow",
            "model": "deepseek-ai/DeepSeek-R1-Test",
            "model_version": "revision-test",
            "allow_model_env_override": False,
            "api_key_env": "TEST_API_KEY",
            "base_url_env": "TEST_BASE_URL",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "seed": 7,
            "enable_thinking": False,
        }
    }


def _dense_block(index_path: Path) -> dict[str, Any]:
    return {
        "backend": "text2vec",
        "model_name": "example/bge",
        "model_version": "v-test",
        "dimension": 8,
        "batch_size": 2,
        "normalize_embeddings": True,
        "index_path": str(index_path),
        "reject_smoke": True,
    }


def _write_prompt_dir(root: Path) -> Path:
    from external_baselines.ekell_style.prompt_identity import EKELL_REQUIRED_PROMPTS

    root.mkdir()
    for name in EKELL_REQUIRED_PROMPTS:
        (root / name).write_text(f"controlled prompt {name}\n", encoding="utf-8")
    return root


def _write_experiment(
    tmp_path: Path,
    bundle: Path | None,
    *,
    build_indexes: bool,
) -> tuple[Path, dict[str, Path]]:
    configs = tmp_path / "configs"
    configs.mkdir()
    base = configs / "base.yaml"
    shared = configs / "shared.yaml"
    base.write_text("{}\n", encoding="utf-8")
    shared.write_text(yaml.safe_dump(_shared_llm(), sort_keys=False), encoding="utf-8")
    dense_index = tmp_path / "dense_index"
    ekell_index = tmp_path / "ekell_index"
    prompt_dir = _write_prompt_dir(tmp_path / "prompts")
    dense = _dense_block(dense_index)
    hybrid = {
        "lexical_method": "bm25",
        "dense_method": dense["backend"],
        "dense_model_name": dense["model_name"],
        "dense_model_version": dense["model_version"],
        "dimension": dense["dimension"],
        "normalize_embeddings": True,
        "top_k": 2,
        "candidate_pool": 4,
        "rrf_k": 60,
        "lexical_weight": 1.0,
        "dense_weight": 1.0,
        "reject_smoke": True,
    }
    method_payloads: dict[str, dict[str, Any]] = {
        "direct_llm": {"method_id": "direct_llm"},
        "bm25_rag": {"method_id": "bm25_rag"},
        "dense_rag": {"method_id": "dense_rag", "dense_rag": dense},
        "hybrid_rag": {
            "method_id": "hybrid_rag",
            "dense_rag": dict(dense),
            "hybrid_rag": hybrid,
        },
        "ekell_style_controlled_shared_llm": {
            "method_id": "ekell_style_controlled_shared_llm",
            "ekell_style": {
                "prompt_dir": str(prompt_dir),
                "dense_entity_retrieval": False,
                "hybrid_subgraph_ranking": False,
                "reranker": False,
                "self_consistency": False,
                "structured_verification": False,
            },
            "ekell_vector": {
                **_dense_block(ekell_index),
                "batch_size": 2,
            },
        },
    }
    paths: dict[str, Path] = {}
    methods = []
    for method_id, payload in method_payloads.items():
        path = configs / f"{method_id}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        paths[method_id] = path
        methods.append(
            {
                "method_id": method_id,
                "config": path.relative_to(tmp_path).as_posix(),
                "enabled": True,
            }
        )
    manifest_payload = {
        "schema_version": "firebench-interop-v1",
        "experiment_id": "builder-test",
        "track": "A_shared_outcome",
        "run_mode": "formal",
        "freeze_status": "provisional",
        "paper_final": True,
        "require_bundle_checksum": True,
        "require_external_schema": True,
        "require_complete_case_match": True,
        "fail_on_schema_error": True,
        "fail_on_duplicate_case_id": True,
        "fail_on_missing_case": True,
        "fail_on_extra_case": True,
        "bundle": bundle.relative_to(tmp_path).as_posix() if bundle else None,
        "base_config": base.relative_to(tmp_path).as_posix(),
        "shared_model_config": shared.relative_to(tmp_path).as_posix(),
        "main_table_methods": [
            "direct_llm",
            "bm25_rag",
            "ekell_style_controlled_shared_llm",
        ],
        "comparison_suite_methods": list(method_payloads),
        "supplemental_methods": ["dense_rag", "hybrid_rag"],
        "methods": methods,
        "output": str(tmp_path / "predictions.jsonl"),
        "run_manifest": str(tmp_path / "run_manifest.json"),
    }
    manifest = tmp_path / "experiment.yaml"
    manifest.write_text(yaml.safe_dump(manifest_payload, sort_keys=False), encoding="utf-8")
    paths.update(
        {
            "manifest": manifest,
            "dense_index": dense_index,
            "ekell_index": ekell_index,
        }
    )

    if bundle is not None and build_indexes:
        loaded = load_runner_bundle(bundle, formal=True)
        corpus_dir = Path(loaded["corpus_dir"])
        corpus_checksum = runner_bundle_corpus_aggregate_sha256(loaded, required=True)
        evidence = corpus_dir / "evidence_chunks.jsonl"
        build_dense_index(
            evidence,
            model_name=dense["model_name"],
            model_version=dense["model_version"],
            backend=dense["backend"],
            dim=dense["dimension"],
            cache_path=dense_index,
            embedding_model=FakeEmbeddingModel(8),
            batch_size=2,
            normalize_embeddings=True,
            paper_final=True,
            reject_smoke=True,
            corpus_checksum=corpus_checksum,
        )
        kg = load_kg_strict(corpus_dir)
        backend = create_embedding_backend(
            "text2vec",
            model_name=dense["model_name"],
            model_version=dense["model_version"],
            dimension=8,
            model=FakeEmbeddingModel(8),
            paper_final=True,
            reject_smoke=True,
        )
        index = VectorIndex.from_kg(
            kg,
            backend,
            corpus_checksum=corpus_checksum,
            kg_checksum=fire_kg_checksum(kg),
            paper_final=True,
            reject_smoke=True,
            normalize_embeddings=True,
        )
        index.save_directory(ekell_index)
    return manifest, paths


def _invoke(manifest: Path, tmp_path: Path) -> tuple[int, dict[str, Any]]:
    import external_baselines.common.experiment_manifest as experiment_manifest_module
    import external_baselines.common.formal_config_validator as formal_validator

    output = tmp_path / "report.json"
    with (
        patch.object(builder, "ROOT", tmp_path),
        patch.object(experiment_manifest_module, "REPOSITORY_ROOT", tmp_path),
        patch.object(formal_validator, "ROOT_REL", tmp_path),
    ):
        try:
            builder.main(
                [
                    "--experiment-manifest",
                    str(manifest),
                    "--validate-only",
                    "--output",
                    str(output),
                ]
            )
        except SystemExit as exc:
            code = int(exc.code)
        else:
            code = 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is (code == 0)
    return code, report


def _invoke_build(manifest: Path, output: Path) -> tuple[int, dict[str, Any]]:
    import external_baselines.common.experiment_manifest as experiment_manifest_module
    import external_baselines.common.formal_config_validator as formal_validator

    repository_root = manifest.parent
    with (
        patch.object(builder, "ROOT", repository_root),
        patch.object(experiment_manifest_module, "REPOSITORY_ROOT", repository_root),
        patch.object(formal_validator, "ROOT_REL", repository_root),
    ):
        try:
            builder.main(
                [
                    "--experiment-manifest",
                    str(manifest),
                    "--output",
                    str(output),
                ]
            )
        except SystemExit as exc:
            code = int(exc.code)
        else:
            code = 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is (code == 0)
    return code, report


def _valid_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    bundle = _make_runner_bundle(tmp_path)
    return _write_experiment(tmp_path, bundle, build_indexes=True)


def _mutate_yaml(path: Path, section: str | None, field: str, value: Any) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = payload if section is None else payload[section]
    target[field] = value
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_validate_only_success_exits_zero(tmp_path: Path) -> None:
    manifest, _ = _valid_fixture(tmp_path)
    code, report = _invoke(manifest, tmp_path)
    assert code == 0
    assert report["ok"] is True
    assert report["errors"] == []


def test_validate_only_missing_bundle_exits_one(tmp_path: Path) -> None:
    manifest, _ = _write_experiment(tmp_path, None, build_indexes=False)
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert "bundle_placeholder_or_missing" in report["errors"]


def test_validate_only_bundle_checksum_failure_exits_one(tmp_path: Path, monkeypatch) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, _ = _write_experiment(tmp_path, bundle, build_indexes=False)
    kg_loaded = False

    def forbidden_kg(*_args, **_kwargs):
        nonlocal kg_loaded
        kg_loaded = True
        raise AssertionError("KG must not load after Bundle checksum failure")

    monkeypatch.setattr(builder, "validate_bundle_checksum", lambda _bundle: {"ok": False})
    monkeypatch.setattr(builder, "load_kg_strict", forbidden_kg)
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert "runner_bundle_file_checksum_validation_failed" in report["errors"]
    assert kg_loaded is False


def test_validate_only_dense_placeholder_exits_one(tmp_path: Path) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    _mutate_yaml(paths["dense_rag"], "dense_rag", "model_version", "REPLACE_ME")
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert report["experiment_validation"]["ok"] is False
    assert any("model_version" in error and "placeholder" in error for error in report["errors"])


def test_validate_only_dense_index_missing_exits_one(tmp_path: Path) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, _ = _write_experiment(tmp_path, bundle, build_indexes=False)
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert "dense_index_path_missing" in report["errors"]


def test_validate_only_dense_validation_failure_exits_one(tmp_path: Path) -> None:
    manifest, paths = _valid_fixture(tmp_path)
    (paths["dense_index"] / "embeddings.npy").write_bytes(b"corrupt")
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert any(error.startswith("dense_index_validation_failed:") for error in report["errors"])


def test_validate_only_ekell_placeholder_exits_one(tmp_path: Path) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    _mutate_yaml(
        paths["ekell_style_controlled_shared_llm"],
        "ekell_vector",
        "model_version",
        "REPLACE_ME",
    )
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert report["experiment_validation"]["ok"] is False
    assert any("model_version" in error and "placeholder" in error for error in report["errors"])


def test_validate_only_ekell_index_missing_exits_one(tmp_path: Path) -> None:
    manifest, paths = _valid_fixture(tmp_path)
    shutil.rmtree(paths["ekell_index"])
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert "ekell_index_path_missing" in report["errors"]


def test_validate_only_ekell_validation_failure_exits_one(tmp_path: Path) -> None:
    manifest, paths = _valid_fixture(tmp_path)
    (paths["ekell_index"] / "embeddings.npy").write_bytes(b"corrupt")
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert any(error.startswith("ekell_index_validation_failed:") for error in report["errors"])


def test_validate_only_hybrid_without_valid_dense_exits_one(tmp_path: Path) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, _ = _write_experiment(tmp_path, bundle, build_indexes=False)
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert "hybrid_dense_dependency_missing" in report["errors"]


def test_validate_only_report_ok_matches_exit_code(tmp_path: Path) -> None:
    manifest, _ = _valid_fixture(tmp_path)
    success_code, success = _invoke(manifest, tmp_path)
    assert success["ok"] is (success_code == 0)
    (tmp_path / "runner_bundle" / "manifest.json").unlink()
    failure_code, failure = _invoke(manifest, tmp_path)
    assert failure["ok"] is (failure_code == 0)


def _assert_builder_rejects_invalid_exact_config_type(
    tmp_path: Path,
    section: str | None,
    field: str,
    value: Any,
    message: str,
) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    target = manifest if section is None else paths["dense_rag"]
    _mutate_yaml(target, section, field, value)
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert any(message in error for error in report["errors"])


def test_builder_rejects_string_paper_final(tmp_path: Path) -> None:
    _assert_builder_rejects_invalid_exact_config_type(
        tmp_path, None, "paper_final", "true", "exact boolean"
    )


def test_builder_rejects_string_reject_smoke(tmp_path: Path) -> None:
    _assert_builder_rejects_invalid_exact_config_type(
        tmp_path, "dense_rag", "reject_smoke", "false", "exact boolean"
    )


def test_builder_rejects_integer_reject_smoke(tmp_path: Path) -> None:
    _assert_builder_rejects_invalid_exact_config_type(
        tmp_path, "dense_rag", "reject_smoke", 1, "exact boolean"
    )


def test_builder_rejects_string_dimension(tmp_path: Path) -> None:
    _assert_builder_rejects_invalid_exact_config_type(
        tmp_path, "dense_rag", "dimension", "8", "exact YAML integer"
    )


def test_builder_rejects_float_dimension(tmp_path: Path) -> None:
    _assert_builder_rejects_invalid_exact_config_type(
        tmp_path, "dense_rag", "dimension", 8.0, "exact YAML integer"
    )


def test_builder_rejects_numeric_model_version(tmp_path: Path) -> None:
    _assert_builder_rejects_invalid_exact_config_type(
        tmp_path, "dense_rag", "model_version", 1, "exact YAML string"
    )


def test_builder_rejects_boolean_backend(tmp_path: Path) -> None:
    _assert_builder_rejects_invalid_exact_config_type(
        tmp_path, "dense_rag", "backend", True, "exact YAML string"
    )


def test_builder_rejects_placeholder_model_version(tmp_path: Path) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    _mutate_yaml(
        paths["dense_rag"],
        "dense_rag",
        "model_version",
        "REQUIRED_BEFORE_REAL_INDEX_BUILD",
    )
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert report["experiment_validation"]["ok"] is False
    assert any("model_version" in error and "placeholder" in error for error in report["errors"])


def test_builder_does_not_truthiness_coerce_false_string(tmp_path: Path) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    _mutate_yaml(paths["dense_rag"], "dense_rag", "reject_smoke", "false")
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert any("reject_smoke must be an exact boolean" in error for error in report["errors"])


def test_dense_source_identity_matches_real_fixture(tmp_path: Path) -> None:
    manifest, paths = _valid_fixture(tmp_path)
    code, report = _invoke(manifest, tmp_path)
    assert code == 0
    evidence = tmp_path / "runner_bundle" / "corpus" / "evidence_chunks.jsonl"
    assert report["indexes"]["dense"]["evidence_source_checksum"] == sha256_file(evidence)
    assert paths["dense_index"].is_dir()


def test_builder_runs_experiment_gate_before_embedding_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import external_baselines.dense_rag.pipeline as dense_pipeline
    import external_baselines.ekell_style.embedding_backends as embedding_backends

    bundle = _make_runner_bundle(tmp_path)
    manifest, _paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    events: list[str] = []

    def reject_experiment(*_args, **_kwargs):
        events.append("experiment_gate")
        raise ValueError("gate-rejected")

    def forbidden_embedding(*_args, **_kwargs):
        events.append("embedding_backend")
        raise AssertionError("embedding backend initialized before experiment gate")

    monkeypatch.setattr(builder, "validate_experiment_manifest", reject_experiment)
    monkeypatch.setattr(dense_pipeline, "build_dense_index", forbidden_embedding)
    monkeypatch.setattr(
        embedding_backends,
        "create_embedding_backend",
        forbidden_embedding,
    )

    code, report = _invoke_build(manifest, tmp_path / "gate-report.json")

    assert code == 1
    assert events == ["experiment_gate"]
    assert report["experiment_validation"]["ok"] is False


def test_builder_runs_experiment_gate_before_index_directory_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    monkeypatch.setattr(
        builder,
        "validate_experiment_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("gate-rejected")),
    )

    code, _report = _invoke_build(manifest, tmp_path / "directory-gate-report.json")

    assert code == 1
    assert not paths["dense_index"].exists()
    assert not paths["ekell_index"].exists()


def test_builder_reports_experiment_gate_failure(tmp_path: Path) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    _mutate_yaml(manifest, None, "run_mode", "smoke")

    code, report = _invoke_build(manifest, tmp_path / "invalid-experiment-report.json")

    assert code == 1
    assert report["experiment_validation"]["stage"] == "index_build_candidate"
    assert report["experiment_validation"]["ok"] is False
    assert any(
        error.startswith("experiment_index_build_candidate_validation_failed:")
        for error in report["errors"]
    )
    assert not paths["dense_index"].exists()
    assert not paths["ekell_index"].exists()
