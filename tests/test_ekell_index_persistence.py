"""E-KELL KG vector index directory persistence tests (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baselines.common.checksums import sha256_file
from external_baselines.ekell_style.embedding_backends import create_embedding_backend
from external_baselines.ekell_style.kg_loader import FireKG
from external_baselines.ekell_style.vector_index import (
    VectorIndex,
    VectorIndexError,
    ekell_index_identity_checksum,
)
from external_baselines.ekell_style.vector_retriever import VectorRetriever


class FakeEmbeddingModel:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.load_count = 1
        self.encode_call_count = 0

    def encode(self, texts, show_progress_bar=False):  # noqa: ARG002
        self.encode_call_count += 1
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for i, ch in enumerate(str(text).encode("utf-8")):
                vec[i % self.dim] += (ch % 13) / 13.0
            out.append(vec)
        return out


def _backend(fake: FakeEmbeddingModel | None = None):
    fake = fake or FakeEmbeddingModel(8)
    return create_embedding_backend(
        "text2vec",
        model_name="fake/bge",
        model_version="v-test",
        dimension=8,
        model=fake,
        reject_smoke=True,
    ), fake


def _tiny_kg() -> FireKG:
    return FireKG(
        entities=[{"entity_id": "e1", "name": "hose"}],
        relations=[{"relation_id": "r1", "name": "used_for"}],
        triples=[{"head": "hose", "relation": "used_for", "tail": "fire", "source_id": "t1"}],
        evidence_chunks=[{"chunk_id": "c1", "text": "fire hose near exit", "source_id": "s1"}],
    )


def _build_ekell_index(tmp_path: Path) -> Path:
    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index_dir = tmp_path / "ekell_idx"
    index.save_directory(index_dir)
    return index_dir


def _read_manifest(index_dir: Path) -> dict:
    return json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(index_dir: Path, manifest: dict) -> None:
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _rewrite_ekell_embeddings(index_dir: Path, arr) -> None:  # noqa: ANN001
    import numpy as np

    emb_path = index_dir / "embeddings.npy"
    np.save(emb_path, arr.astype("float32"))
    manifest = _read_manifest(index_dir)
    manifest["embeddings_checksum"] = sha256_file(emb_path)
    manifest["index_checksum"] = ekell_index_identity_checksum(
        backend=manifest["backend"],
        model_name=manifest["model_name"],
        model_version=manifest["model_version"],
        dimension=manifest["dimension"],
        normalize_embeddings=manifest["normalize_embeddings"],
        document_count=manifest["document_count"],
        documents_checksum=manifest["documents_checksum"],
        embeddings_checksum=manifest["embeddings_checksum"],
        kg_checksum=manifest["kg_checksum"],
        corpus_checksum=manifest["corpus_checksum"],
    )
    _write_manifest(index_dir, manifest)


def test_ekell_index_save_directory(tmp_path: Path) -> None:
    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    manifest = index.save_directory(tmp_path / "ekell_idx")
    assert (tmp_path / "ekell_idx" / "documents.jsonl").is_file()
    assert (tmp_path / "ekell_idx" / "embeddings.npy").is_file()
    assert (tmp_path / "ekell_idx" / "index_manifest.json").is_file()
    assert manifest["index_type"] == "ekell_kg_vector_index"
    assert manifest["normalize_embeddings"] is True
    assert manifest["actual_embedding_used"] is True
    assert manifest["smoke_fallback_used"] is False


def test_ekell_build_records_explicit_normalize_embeddings(tmp_path: Path) -> None:
    backend, _ = _backend()
    index = VectorIndex.from_kg(
        _tiny_kg(),
        backend,
        reject_smoke=True,
        normalize_embeddings=False,
    )
    manifest = index.save_directory(tmp_path / "ekell_idx")

    assert manifest["normalize_embeddings"] is False


def test_ekell_build_rejects_string_normalize_embeddings() -> None:
    backend, _ = _backend()

    with pytest.raises(VectorIndexError, match="ekell_index_normalize_embeddings_must_be_bool"):
        VectorIndex.from_kg(
            _tiny_kg(),
            backend,
            reject_smoke=True,
            normalize_embeddings="true",  # type: ignore[arg-type]
        )


def test_ekell_build_actually_normalizes_vectors() -> None:
    import numpy as np

    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True, normalize_embeddings=True)
    norms = np.linalg.norm(np.asarray(index.vectors, dtype="float32"), axis=1)

    assert np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5)


def test_ekell_index_checksum_changes_with_normalize_embeddings() -> None:
    payload = {
        "backend": "text2vec",
        "model_name": "fake/bge",
        "model_version": "v-test",
        "dimension": 8,
        "document_count": 2,
        "documents_checksum": "1" * 64,
        "embeddings_checksum": "2" * 64,
        "kg_checksum": "3" * 64,
        "corpus_checksum": "4" * 64,
    }
    assert ekell_index_identity_checksum(normalize_embeddings=True, **payload) != ekell_index_identity_checksum(
        normalize_embeddings=False,
        **payload,
    )


def test_ekell_save_directory_rejects_missing_normalization_metadata(tmp_path: Path) -> None:
    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index.metadata.pop("normalize_embeddings")

    with pytest.raises(VectorIndexError, match="normalize_embeddings"):
        index.save_directory(tmp_path / "ekell_idx")


def test_ekell_index_load_directory(tmp_path: Path) -> None:
    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index.save_directory(tmp_path / "ekell_idx")
    loaded = VectorIndex.load_directory(
        tmp_path / "ekell_idx",
        expected_backend="text2vec",
        expected_model_name="fake/bge",
        expected_model_version="v-test",
        require_real_embedding=True,
    )
    assert len(loaded.documents) == len(index.documents)
    assert loaded.metadata["index_checksum"] == index.metadata["index_checksum"]


def test_ekell_index_rejects_documents_checksum_mismatch(tmp_path: Path) -> None:
    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index_dir = tmp_path / "ekell_idx"
    index.save_directory(index_dir)
    docs = index_dir / "documents.jsonl"
    docs.write_text(
        '{"document_id":"x","text":"tampered","source_id":null,"citation":null,"metadata":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(VectorIndexError, match="documents"):
        VectorIndex.load_directory(index_dir, require_real_embedding=True)


def test_ekell_index_rejects_embeddings_checksum_mismatch(tmp_path: Path) -> None:
    import numpy as np

    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index_dir = tmp_path / "ekell_idx"
    index.save_directory(index_dir)
    np.save(index_dir / "embeddings.npy", np.zeros((len(index.vectors), 8), dtype="float32"))
    with pytest.raises(VectorIndexError, match="embeddings"):
        VectorIndex.load_directory(index_dir, require_real_embedding=True)


def test_ekell_index_rejects_index_checksum_mismatch(tmp_path: Path) -> None:
    import json

    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index_dir = tmp_path / "ekell_idx"
    index.save_directory(index_dir)
    manifest_path = index_dir / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_checksum"] = "deadbeef"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VectorIndexError, match="index_checksum"):
        VectorIndex.load_directory(index_dir, require_real_embedding=True)


def test_ekell_index_rejects_model_version_mismatch(tmp_path: Path) -> None:
    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index.save_directory(tmp_path / "ekell_idx")
    with pytest.raises(VectorIndexError, match="model_version"):
        VectorIndex.load_directory(
            tmp_path / "ekell_idx",
            expected_model_version="other",
            require_real_embedding=True,
        )


def test_ekell_index_rejects_kg_checksum_mismatch(tmp_path: Path) -> None:
    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index.save_directory(tmp_path / "ekell_idx")
    with pytest.raises(VectorIndexError, match="kg_checksum"):
        VectorIndex.load_directory(
            tmp_path / "ekell_idx",
            expected_kg_checksum="not-the-kg",
            require_real_embedding=True,
        )


def test_ekell_pipeline_loads_configured_index(tmp_path: Path) -> None:
    backend, fake = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True)
    index_dir = tmp_path / "ekell_idx"
    index.save_directory(index_dir)
    retriever = VectorRetriever.from_index_directory(
        index_dir,
        backend,
        reject_smoke=True,
        expected_dimension=8,
    )
    hits = retriever.retrieve("fire hose", top_k=2)
    assert hits
    assert fake.encode_call_count >= 1


def test_ekell_formal_rejects_missing_runtime() -> None:
    from external_baselines.ekell_style.full_pipeline import run_ekell_full_pipeline

    with pytest.raises(RuntimeError, match="EKELLRuntime"):
        run_ekell_full_pipeline(
            {"scenario_id": "s1", "scenario_text": "fire"},
            config={
                "paper_final": True,
                "ekell_vector": {"reject_smoke": True, "backend": "text2vec", "model_name": "x", "model_version": "y"},
                "llm": {"provider": "heuristic", "model": "unit-test-heuristic", "model_version": "unit-test"},
            },
            method="ekell_style_controlled_shared_llm",
            track="controlled_shared_llm",
        )


def test_ekell_does_not_rebuild_index_per_case(tmp_path: Path) -> None:
    from external_baselines.common.method_runtime import clear_runtime_cache, prepare_ekell_runtime
    from external_baselines.ekell_style.full_pipeline import run_controlled_shared_llm

    clear_runtime_cache()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "entities.jsonl").write_text('{"entity_id":"e1","name":"hose"}\n', encoding="utf-8")
    (corpus / "relations.jsonl").write_text('{"relation_id":"r1","name":"used_for"}\n', encoding="utf-8")
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"used_for","tail":"fire","source_id":"t1"}\n',
        encoding="utf-8",
    )
    (corpus / "evidence_chunks.jsonl").write_text(
        '{"chunk_id":"c1","text":"fire hose near exit","source_id":"s1"}\n',
        encoding="utf-8",
    )
    fake = FakeEmbeddingModel(8)
    backend, _ = _backend(fake)
    kg = FireKG(
        entities=[{"entity_id": "e1", "name": "hose"}],
        relations=[{"relation_id": "r1", "name": "used_for"}],
        triples=[{"head": "hose", "relation": "used_for", "tail": "fire", "source_id": "t1"}],
        evidence_chunks=[{"chunk_id": "c1", "text": "fire hose near exit", "source_id": "s1"}],
    )
    index = VectorIndex.from_kg(kg, backend, reject_smoke=False)
    index_dir = tmp_path / "ekell_idx"
    index.save_directory(index_dir)

    config = {
        "paper_final": False,
        "paths": {"corpus_dir": str(corpus)},
        "ekell_vector": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(index_dir),
            "reject_smoke": False,
            "injected_model": FakeEmbeddingModel(8),
        },
        "ekell_style": {"prompt_dir": "configs/prompts/controlled"},
        "llm": {"provider": "heuristic", "model": "h", "model_version": "h"},
        "scenario_parser": {"use_llm": False},
    }
    runtime = prepare_ekell_runtime(config)
    loads_before = runtime.audit.index_load_count
    for i in range(3):
        run_controlled_shared_llm(
            {"scenario_id": f"s{i}", "scenario_text": "hose fire"},
            config=config,
            runtime=runtime,
        )
    assert runtime.audit.index_load_count == loads_before
    assert runtime.audit.case_count == 3


def test_ekell_freeze_integrity_accepts_valid_index(tmp_path: Path) -> None:
    index_dir = _build_ekell_index(tmp_path)

    result = VectorIndex.validate_directory_for_freeze(
        index_dir,
        expected_backend="text2vec",
        expected_model_name="fake/bge",
        expected_model_version="v-test",
        expected_dimension=8,
    )

    assert result["index_checksum"]
    assert result["index_manifest_sha256"]
    assert result["kg_checksum"]
    assert result["corpus_checksum"]
    assert result["actual_embedding_used"] is True
    assert result["smoke_fallback_used"] is False


def test_ekell_freeze_validator_rejects_normalization_mismatch(tmp_path: Path) -> None:
    index_dir = _build_ekell_index(tmp_path)

    with pytest.raises(VectorIndexError, match="ekell_index_normalize_embeddings_mismatch"):
        VectorIndex.validate_directory_for_freeze(
            index_dir,
            expected_normalize_embeddings=False,
        )


def test_ekell_query_uses_matching_normalization_policy() -> None:
    backend, _ = _backend()
    index = VectorIndex.from_kg(_tiny_kg(), backend, reject_smoke=True, normalize_embeddings=True)
    seen: dict[str, float] = {}

    def _record_query(query_vector, **_kwargs):  # noqa: ANN001
        import math

        seen["norm"] = math.sqrt(sum(float(value) * float(value) for value in query_vector))
        return []

    index.search = _record_query  # type: ignore[method-assign]
    retriever = VectorRetriever(index, backend, reject_smoke=True)
    retriever.retrieve("fire hose", top_k=1)

    assert seen["norm"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field",
    [
        "index_checksum",
        "kg_checksum",
        "corpus_checksum",
    ],
)
def test_ekell_freeze_integrity_requires_required_checksums(tmp_path: Path, field: str) -> None:
    index_dir = _build_ekell_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest.pop(field)
    _write_manifest(index_dir, manifest)

    with pytest.raises(VectorIndexError, match=field):
        VectorIndex.validate_directory_for_freeze(index_dir)


def test_ekell_freeze_integrity_rejects_tampered_documents_jsonl(tmp_path: Path) -> None:
    index_dir = _build_ekell_index(tmp_path)
    docs = index_dir / "documents.jsonl"
    docs.write_text(docs.read_text(encoding="utf-8").replace("fire hose", "tampered hose", 1), encoding="utf-8")

    with pytest.raises(VectorIndexError, match="documents_file_checksum"):
        VectorIndex.validate_directory_for_freeze(index_dir)


def test_ekell_freeze_integrity_rejects_tampered_embeddings_same_shape(tmp_path: Path) -> None:
    import numpy as np

    index_dir = _build_ekell_index(tmp_path)
    emb = index_dir / "embeddings.npy"
    arr = np.load(emb)
    arr[0, 0] = arr[0, 0] + 0.125
    np.save(emb, arr.astype("float32"))

    with pytest.raises(VectorIndexError, match="embeddings_checksum"):
        VectorIndex.validate_directory_for_freeze(index_dir)


def test_ekell_freeze_integrity_rejects_index_checksum_mismatch(tmp_path: Path) -> None:
    index_dir = _build_ekell_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["index_checksum"] = "1" * 64
    _write_manifest(index_dir, manifest)

    with pytest.raises(VectorIndexError, match="index_checksum"):
        VectorIndex.validate_directory_for_freeze(index_dir)


def test_ekell_freeze_integrity_rejects_string_boolean_metadata(tmp_path: Path) -> None:
    index_dir = _build_ekell_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["smoke_fallback_used"] = "false"
    _write_manifest(index_dir, manifest)

    with pytest.raises(VectorIndexError, match="smoke_fallback_used"):
        VectorIndex.validate_directory_for_freeze(index_dir)


def test_ekell_strict_validator_rejects_non_unit_vectors_when_normalized(tmp_path: Path) -> None:
    import numpy as np

    index_dir = _build_ekell_index(tmp_path)
    arr = np.load(index_dir / "embeddings.npy")
    arr[0] *= 2.0
    _rewrite_ekell_embeddings(index_dir, arr)

    with pytest.raises(VectorIndexError, match="unit_norm"):
        VectorIndex.validate_directory_for_freeze(index_dir)


@pytest.mark.parametrize(("value", "match"), [(float("nan"), "non_finite"), (float("inf"), "non_finite")])
def test_ekell_strict_validator_rejects_non_finite_embedding(
    tmp_path: Path,
    value: float,
    match: str,
) -> None:
    import numpy as np

    index_dir = _build_ekell_index(tmp_path)
    arr = np.load(index_dir / "embeddings.npy")
    arr[0, 0] = value
    _rewrite_ekell_embeddings(index_dir, arr)

    with pytest.raises(VectorIndexError, match=match):
        VectorIndex.validate_directory_for_freeze(index_dir)


def test_ekell_strict_validator_rejects_zero_vector(tmp_path: Path) -> None:
    import numpy as np

    index_dir = _build_ekell_index(tmp_path)
    arr = np.load(index_dir / "embeddings.npy")
    arr[0] = 0.0
    _rewrite_ekell_embeddings(index_dir, arr)

    with pytest.raises(VectorIndexError, match="ekell_index_zero_embedding_vector"):
        VectorIndex.validate_directory_for_freeze(index_dir)
