"""Comparison readiness and five-method fairness tests (no network/API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from external_baselines.common.comparison_readiness import assess_comparison_readiness
from external_baselines.common.fairness import CrossMethodFairnessError, validate_cross_method_fairness
from external_baselines.common.formal_config_validator import FormalConfigError, validate_experiment_manifest
from external_baselines.common.freeze_manifest import validate_freeze_manifest

ROOT = Path(__file__).resolve().parents[1]


def _shared_llm() -> dict:
    return {
        "provider": "siliconflow",
        "model": "m",
        "model_version": "v",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 100,
        "seed": 7,
    }


def _emb() -> dict:
    return {"backend": "text2vec", "model_name": "BAAI/bge-m3", "model_version": "rev1", "dimension": 1024}


def test_five_methods_share_llm_identity() -> None:
    llm = _shared_llm()
    configs = {
        mid: {"paper_final": True, "llm": dict(llm), "bundle_checksum": "b1", "corpus_checksum": "c1"}
        for mid in (
            "direct_llm",
            "bm25_rag",
            "dense_rag",
            "hybrid_rag",
            "ekell_style_controlled_shared_llm",
        )
    }
    configs["dense_rag"]["dense_rag"] = _emb()
    configs["hybrid_rag"]["dense_rag"] = _emb()
    configs["ekell_style_controlled_shared_llm"]["ekell_vector"] = _emb()
    report = validate_cross_method_fairness(configs)
    assert report["shared_llm"] is True


def test_five_methods_share_bundle_and_corpus() -> None:
    llm = _shared_llm()
    configs = {
        mid: {
            "paper_final": True,
            "llm": dict(llm),
            "bundle_checksum": "bundle-x",
            "corpus_checksum": "corpus-y",
        }
        for mid in ("direct_llm", "bm25_rag", "dense_rag")
    }
    configs["dense_rag"]["dense_rag"] = _emb()
    report = validate_cross_method_fairness(configs)
    assert report["shared_bundle"] is True
    assert report["shared_corpus"] is True


def test_dense_hybrid_ekell_share_embedding_identity() -> None:
    llm = _shared_llm()
    emb = _emb()
    configs = {
        "dense_rag": {"paper_final": True, "llm": llm, "dense_rag": emb},
        "hybrid_rag": {"paper_final": True, "llm": llm, "dense_rag": emb},
        "ekell_style_controlled_shared_llm": {"paper_final": True, "llm": llm, "ekell_vector": emb},
    }
    report = validate_cross_method_fairness(configs)
    assert report["shared_embedding_identity"] is True


def test_hybrid_requires_dense_index_checksum() -> None:
    llm = _shared_llm()
    emb = _emb()
    with pytest.raises(CrossMethodFairnessError):
        validate_cross_method_fairness(
            {
                "dense_rag": {
                    "paper_final": True,
                    "llm": llm,
                    "dense_rag": emb,
                    "dense_index_checksum": "aaa",
                },
                "hybrid_rag": {
                    "paper_final": True,
                    "llm": llm,
                    "dense_rag": emb,
                    "dense_index_checksum": "bbb",
                },
            }
        )


def test_different_index_types_do_not_fail_embedding_fairness() -> None:
    """Dense evidence index vs E-KELL KG index may differ; embedding identity must still match."""
    llm = _shared_llm()
    emb = _emb()
    report = validate_cross_method_fairness(
        {
            "dense_rag": {
                "paper_final": True,
                "llm": llm,
                "dense_rag": emb,
                "dense_index_checksum": "dense-idx",
            },
            "ekell_style_controlled_shared_llm": {
                "paper_final": True,
                "llm": llm,
                "ekell_vector": emb,
                "ekell_index_checksum": "ekell-idx",
            },
        }
    )
    assert report["shared_embedding_identity"] is True


def test_comparison_readiness_reports_each_method(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n  api_key_env: SILICONFLOW_API_KEY\n",
        encoding="utf-8",
    )
    for name in ("direct", "bm25", "dense", "hybrid", "ekell"):
        (tmp_path / f"{name}.yaml").write_text(f"method_id: {name}\n", encoding="utf-8")
    dense = tmp_path / "dense.yaml"
    dense.write_text(
        "dense_rag:\n  backend: text2vec\n  model_name: BAAI/bge-m3\n"
        "  model_version: REQUIRED_BEFORE_REAL_INDEX_BUILD\n  dimension: 1024\n"
        "  index_path: outputs/indexes/dense/missing/\n",
        encoding="utf-8",
    )
    hybrid = tmp_path / "hybrid.yaml"
    hybrid.write_text("hybrid_rag:\n  rrf_k: 60\n", encoding="utf-8")
    ekell = tmp_path / "ekell.yaml"
    ekell.write_text(
        "ekell_vector:\n  backend: text2vec\n  model_name: BAAI/bge-m3\n"
        "  model_version: REQUIRED_BEFORE_REAL_INDEX_BUILD\n  index_path: missing/\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "m.yaml"
    methods = [
        {"method_id": "direct_llm", "config": str(tmp_path / "direct.yaml"), "enabled": True},
        {"method_id": "bm25_rag", "config": str(tmp_path / "bm25.yaml"), "enabled": True},
        {"method_id": "dense_rag", "config": str(dense), "enabled": True},
        {"method_id": "hybrid_rag", "config": str(hybrid), "enabled": True},
        {
            "method_id": "ekell_style_controlled_shared_llm",
            "config": str(ekell),
            "enabled": True,
        },
    ]
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "t",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "freeze_status": "provisional",
                "paper_final": True,
                "bundle": "bundle",
                "base_config": str(ROOT / "configs/default.yaml"),
                "shared_model_config": str(shared),
                "comparison_suite_methods": [m["method_id"] for m in methods],
                "methods": methods,
            }
        ),
        encoding="utf-8",
    )
    report = assess_comparison_readiness(experiment_manifest=manifest, method_set="comparison_suite")
    assert set(report["methods"]) == {m["method_id"] for m in methods}
    assert "reasons" in report["methods"]["dense_rag"]
    assert report["comparison_ready"] is False


def test_dense_missing_index_blocks_comparison(tmp_path: Path) -> None:
    report = assess_comparison_readiness(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        method_set="comparison_suite",
    )
    dense = report["methods"].get("dense_rag") or {}
    assert dense.get("ready") is False
    assert any("index" in r or "placeholder" in r for r in dense.get("reasons") or [])


def test_hybrid_dependency_failure_is_explicit(tmp_path: Path) -> None:
    report = assess_comparison_readiness(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        method_set="comparison_suite",
    )
    hybrid = report["methods"]["hybrid_rag"]
    assert hybrid["ready"] is False
    assert "dense_dependency_not_ready" in hybrid["reasons"] or "bm25_dependency_not_ready" in hybrid["reasons"]


def test_ekell_missing_kg_blocks_comparison() -> None:
    report = assess_comparison_readiness(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        method_set="comparison_suite",
    )
    ekell = report["methods"]["ekell_style_controlled_shared_llm"]
    assert ekell["ready"] is False
    assert ekell["reasons"]


def test_api_env_presence_never_exposes_secret() -> None:
    report = assess_comparison_readiness(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        method_set="comparison_suite",
    )
    blob = json.dumps(report)
    assert "sk-" not in blob.lower()
    for value in (report.get("api_env_presence") or {}).values():
        assert value in {"present", "missing", "unknown"} or isinstance(value, str)
        assert "sk-" not in str(value)


def test_complete_fixture_is_comparison_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-not-a-real-key")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    evidence = corpus / "evidence_chunks.jsonl"
    evidence.write_text(
        '{"chunk_id":"c1","text":"hose","source_id":"s1"}\n'
        '{"chunk_id":"c2","text":"smoke","source_id":"s2"}\n',
        encoding="utf-8",
    )
    for name in ("entities.jsonl", "relations.jsonl", "triples.jsonl"):
        (corpus / name).write_text("{}\n", encoding="utf-8")

    from external_baselines.dense_rag.pipeline import build_dense_index
    from external_baselines.ekell_style.kg_loader import load_kg
    from external_baselines.ekell_style.vector_index import VectorIndex
    from external_baselines.retrieval.embedding_backends import create_embedding_backend

    class FakeEmbeddingModel:
        def __init__(self, dim: int = 8) -> None:
            self.dim = dim

        def encode(self, texts, show_progress_bar=False):  # noqa: ARG002
            out = []
            for text in texts:
                vec = [0.0] * self.dim
                for i, ch in enumerate(str(text).encode("utf-8")):
                    vec[i % self.dim] += (ch % 13) / 13.0
                out.append(vec)
            return out

    dense_idx = tmp_path / "dense_idx"
    build_dense_index(
        evidence,
        model_name="BAAI/bge-m3",
        model_version="rev1",
        backend="text2vec",
        dim=8,
        cache_path=dense_idx,
        embedding_model=FakeEmbeddingModel(8),
        reject_smoke=True,
    )
    ekell_idx = tmp_path / "ekell_idx"
    kg = load_kg(corpus)
    backend = create_embedding_backend(
        "text2vec",
        model_name="BAAI/bge-m3",
        model_version="rev1",
        dimension=8,
        model=FakeEmbeddingModel(8),
        reject_smoke=True,
    )
    VectorIndex.from_kg(kg, backend, reject_smoke=True).save_directory(ekell_idx)

    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n  api_key_env: SILICONFLOW_API_KEY\n"
        f"paths:\n  corpus_dir: {corpus.as_posix()}\n",
        encoding="utf-8",
    )
    base = tmp_path / "base.yaml"
    base.write_text(f"paths:\n  corpus_dir: {corpus.as_posix()}\n", encoding="utf-8")
    method_cfgs = {
        "direct_llm": "method_id: direct_llm\n",
        "bm25_rag": "method_id: bm25_rag\n",
        "dense_rag": (
            "dense_rag:\n  backend: text2vec\n  model_name: BAAI/bge-m3\n"
            f"  model_version: rev1\n  dimension: 8\n  index_path: {dense_idx.as_posix()}\n"
            "  normalize_embeddings: true\n  reject_smoke: true\n"
        ),
        "hybrid_rag": (
            "hybrid_rag:\n  lexical_method: bm25\n  rrf_k: 60\n  lexical_weight: 1.0\n"
            "  dense_weight: 1.0\n  top_k: 5\n  candidate_pool: 20\n  reject_smoke: true\n"
            "dense_rag:\n  backend: text2vec\n  model_name: BAAI/bge-m3\n"
            f"  model_version: rev1\n  dimension: 8\n  index_path: {dense_idx.as_posix()}\n"
            "  normalize_embeddings: true\n  reject_smoke: true\n"
        ),
        "ekell_style_controlled_shared_llm": (
            "ekell_vector:\n  backend: text2vec\n  model_name: BAAI/bge-m3\n"
            f"  model_version: rev1\n  dimension: 8\n  index_path: {ekell_idx.as_posix()}\n"
            "  normalize_embeddings: true\n  reject_smoke: true\n"
            "ekell_style:\n  prompt_dir: configs/prompts/controlled\n"
        ),
    }
    methods = []
    for mid, body in method_cfgs.items():
        path = tmp_path / f"{mid}.yaml"
        path.write_text(body, encoding="utf-8")
        methods.append({"method_id": mid, "config": str(path), "enabled": True})
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "ready",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "freeze_status": "provisional",
                "paper_final": True,
                "bundle": "bundle",
                "base_config": str(base),
                "shared_model_config": str(shared),
                "comparison_suite_methods": list(method_cfgs),
                "methods": methods,
            }
        ),
        encoding="utf-8",
    )
    resources = tmp_path / "resources.yaml"
    resources.write_text(
        yaml.safe_dump(
            {
                "main_project": {
                    "repo_path": str(ROOT),
                    "expected_branch": "main",
                    "runner_bundle_path": str(tmp_path / "bundle"),
                },
                "status": {"main_project_v1_ready": False},
            }
        ),
        encoding="utf-8",
    )
    report = assess_comparison_readiness(
        experiment_manifest=manifest,
        resources_path=resources,
        method_set="comparison_suite",
    )
    not_ready = {mid: m for mid, m in report["methods"].items() if not m.get("ready")}
    assert not not_ready, not_ready
    # overall comparison_ready still false without main_project_v1_ready / real bundle
    assert report["comparison_ready"] is False
    assert "bundle" in report


def test_dry_run_accepts_frozen(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
        encoding="utf-8",
    )
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
        ),
        encoding="utf-8",
    )
    result = validate_experiment_manifest(manifest, validation_stage="dry_run")
    assert result["valid"] is True


def test_formal_requires_selected_dev_evidence(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
        encoding="utf-8",
    )
    method = tmp_path / "direct.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "freeze_id": "x",
                "freeze_status": "frozen",
                "selected_dev_run_evidence": "missing/selected.json",
                "embedding": {"model_version": "rev1"},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "t",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "paper_final": True,
                "freeze_status": "frozen",
                "freeze_manifest": str(freeze),
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
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="selected_dev_run_evidence"):
        validate_experiment_manifest(manifest, validation_stage="formal")


def test_formal_rejects_config_checksum_mismatch(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "freeze_id": "x",
                "freeze_status": "frozen",
                "selected_dev_run_evidence": str(evidence),
                "experiment_manifest_sha256": "deadbeef",
                "embedding": {"model_version": "rev1"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="experiment_manifest_sha256"):
        validate_freeze_manifest(
            freeze,
            experiment_manifest_path=shared,
            experiment_raw={"shared_model_config": str(shared)},
        )


def test_formal_rejects_prompt_checksum_mismatch(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "freeze_id": "x",
                "freeze_status": "frozen",
                "selected_dev_run_evidence": str(evidence),
                "prompt_tree_sha256": "not-the-real-hash",
                "embedding": {"model_version": "rev1"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="prompt_tree_sha256"):
        validate_freeze_manifest(
            freeze,
            experiment_manifest_path=tmp_path / "m.yaml",
            experiment_raw={},
        )


def test_formal_rejects_embedding_version_mismatch(tmp_path: Path) -> None:
    evidence = tmp_path / "selected.json"
    evidence.write_text("{}", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "freeze_id": "x",
                "freeze_status": "frozen",
                "selected_dev_run_evidence": str(evidence),
                "embedding": {"model_version": "REQUIRED_BEFORE_REAL_INDEX_BUILD"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="embedding.model_version"):
        validate_freeze_manifest(
            freeze,
            experiment_manifest_path=tmp_path / "m.yaml",
            experiment_raw={},
        )


def test_formal_accepts_complete_frozen_configuration(tmp_path: Path) -> None:
    from external_baselines.common.checksums import sha256_file
    from external_baselines.common.freeze_manifest import prompt_tree_checksum

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
    freeze.write_text(
        json.dumps(
            {
                "freeze_id": "controlled_comparison_v1",
                "freeze_status": "frozen",
                "selected_dev_run_evidence": str(evidence),
                "experiment_manifest_sha256": sha256_file(manifest),
                "shared_model_config_sha256": sha256_file(shared),
                "prompt_tree_sha256": prompt_tree_checksum("configs/prompts/controlled"),
                "embedding": {"backend": "text2vec", "model_name": "BAAI/bge-m3", "model_version": "rev1"},
            }
        ),
        encoding="utf-8",
    )
    payload["freeze_manifest"] = str(freeze)
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    # refresh checksum after adding freeze_manifest path
    freeze.write_text(
        json.dumps(
            {
                "freeze_id": "controlled_comparison_v1",
                "freeze_status": "frozen",
                "selected_dev_run_evidence": str(evidence),
                "experiment_manifest_sha256": sha256_file(manifest),
                "shared_model_config_sha256": sha256_file(shared),
                "prompt_tree_sha256": prompt_tree_checksum("configs/prompts/controlled"),
                "embedding": {"backend": "text2vec", "model_name": "BAAI/bge-m3", "model_version": "rev1"},
            }
        ),
        encoding="utf-8",
    )
    result = validate_experiment_manifest(manifest, validation_stage="formal")
    assert result["valid"] is True
