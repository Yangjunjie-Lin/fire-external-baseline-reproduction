"""E-KELL KG vector index directory persistence tests (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baselines.common.checksums import sha256_file
from external_baselines.ekell_style.embedding_backends import create_embedding_backend
from external_baselines.ekell_style.kg_loader import FireKG, load_kg_strict
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


def _strict_kg_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "strict_kg"
    corpus.mkdir()
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose"}\n', encoding="utf-8"
    )
    (corpus / "relations.jsonl").write_text(
        '{"relation_id":"r1","name":"used_for"}\n', encoding="utf-8"
    )
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"used_for","tail":"fire"}\n',
        encoding="utf-8",
    )
    (corpus / "evidence_chunks.jsonl").write_text(
        '{"chunk_id":"c1","text":"hose evidence","source_id":"s1"}\n',
        encoding="utf-8",
    )
    return corpus


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


@pytest.mark.parametrize(
    "filename",
    [
        "entities.jsonl",
        "relations.jsonl",
        "triples.jsonl",
        "evidence_chunks.jsonl",
    ],
)
def test_strict_kg_rejects_missing_required_file(
    tmp_path: Path,
    filename: str,
) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / filename).unlink()
    with pytest.raises(ValueError, match=rf"kg_jsonl_missing:{filename}"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_empty_file(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "relations.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="kg_jsonl_empty:relations.jsonl"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_non_object_jsonl_record(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '["not", "an", "object"]\n', encoding="utf-8"
    )
    with pytest.raises(
        ValueError,
        match="kg_jsonl_record_must_be_object:entities.jsonl:line_1",
    ):
        load_kg_strict(corpus)


def test_strict_kg_reports_original_line_number(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '\n{"head":"ok","relation":"r","tail":"ok"}\n[1, 2]\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="kg_jsonl_record_must_be_object:triples.jsonl:line_3",
    ):
        load_kg_strict(corpus)


@pytest.mark.parametrize("field", ["head", "relation", "tail"])
def test_strict_kg_rejects_triple_missing_required_field(
    tmp_path: Path,
    field: str,
) -> None:
    corpus = _strict_kg_dir(tmp_path)
    triple = {"head": "hose", "relation": "used_for", "tail": "fire"}
    del triple[field]
    (corpus / "triples.jsonl").write_text(
        json.dumps(triple) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=rf"triple_{field}_missing"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_evidence_missing_text(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "evidence_chunks.jsonl").write_text(
        '{"chunk_id":"c1","source_id":"s1"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="evidence_text_missing"):
        load_kg_strict(corpus)


def _write_strict_row(corpus: Path, filename: str, row: dict) -> None:
    (corpus / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_strict_kg_rejects_numeric_evidence_text(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "evidence_chunks.jsonl",
        {"chunk_id": "c1", "text": 456, "source_id": "s1"},
    )
    with pytest.raises(ValueError, match="evidence_text_must_be_string"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_boolean_evidence_text(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "evidence_chunks.jsonl",
        {"chunk_id": "c1", "text": True, "source_id": "s1"},
    )
    with pytest.raises(ValueError, match="evidence_text_must_be_string"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_list_evidence_text(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "evidence_chunks.jsonl",
        {"chunk_id": "c1", "text": ["not", "text"], "source_id": "s1"},
    )
    with pytest.raises(ValueError, match="evidence_text_must_be_string"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_semantic_text_surrounding_whitespace(
    tmp_path: Path,
) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "evidence_chunks.jsonl",
        {"chunk_id": "c1", "text": " padded evidence ", "source_id": "s1"},
    )
    with pytest.raises(
        ValueError,
        match="evidence_text_must_not_have_surrounding_whitespace",
    ):
        load_kg_strict(corpus)


def test_strict_kg_rejects_numeric_relation_label(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "triples.jsonl",
        {"head": 1, "relation": 2, "tail": 3},
    )
    with pytest.raises(ValueError, match="triple_relation_must_be_string"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_float_head_id(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "triples.jsonl",
        {"head": 1.5, "relation": "used_for", "tail": 3},
    )
    with pytest.raises(ValueError, match="triple_head_identifier_invalid_type"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_boolean_tail_id(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "triples.jsonl",
        {"head": 1, "relation": "used_for", "tail": False},
    )
    with pytest.raises(ValueError, match="triple_tail_identifier_invalid_type"):
        load_kg_strict(corpus)


def test_strict_kg_accepts_integer_entity_id(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "entities.jsonl",
        {"entity_id": 1, "name": "hose"},
    )
    kg = load_kg_strict(corpus)
    assert kg.entities[0]["entity_id"] == 1


def test_strict_kg_accepts_integer_chunk_id(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "evidence_chunks.jsonl",
        {"chunk_id": 1, "text": "hose evidence", "citation": "manual"},
    )
    kg = load_kg_strict(corpus)
    assert kg.evidence_chunks[0]["chunk_id"] == 1


def test_strict_kg_requires_string_source_or_citation(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    _write_strict_row(
        corpus,
        "evidence_chunks.jsonl",
        {"chunk_id": "c1", "text": "hose evidence", "source_id": 789},
    )
    with pytest.raises(ValueError, match="evidence_source_or_citation_must_be_string"):
        load_kg_strict(corpus)


def test_strict_kg_error_contains_file_line_and_field(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "evidence_chunks.jsonl").write_text(
        '\n{"chunk_id":"c1","text":123,"source_id":"s1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=(
            "kg_schema_invalid:evidence_chunks.jsonl:line_2:"
            "evidence_text_must_be_string"
        ),
    ):
        load_kg_strict(corpus)


def test_strict_kg_rejects_duplicate_chunk_id(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "evidence_chunks.jsonl").write_text(
        '\n'.join(
            [
                '{"chunk_id":"c1","text":"first","source_id":"s1"}',
                '{"chunk_id":"c1","text":"second","source_id":"s2"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kg_duplicate_evidence_chunk_id:c1"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_duplicate_entity_id(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '\n'.join(
            [
                '{"entity_id":1,"name":"hose"}',
                '{"entity_id":1,"name":"hydrant"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kg_duplicate_entity_id:1"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_duplicate_relation_id(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "relations.jsonl").write_text(
        '\n'.join(
            [
                '{"relation_id":"r1","name":"used_for"}',
                '{"relation_id":"r1","name":"supports"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kg_duplicate_relation_id:r1"):
        load_kg_strict(corpus)


def test_strict_kg_rejects_duplicate_triple(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '\n'.join(
            [
                '{"head":"hose","relation":"used_for","tail":"fire"}',
                '{"head":"hose","relation":"used_for","tail":"fire"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=r"kg_duplicate_triple_provenance:hose\|used_for\|fire\|provenance_sha256=",
    ):
        load_kg_strict(corpus)


def test_duplicate_explicit_triple_id_is_rejected(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '\n'.join(
            [
                '{"triple_id":"t1","head":"hose","relation":"used_for","tail":"fire","source_id":"s1"}',
                '{"triple_id":"t1","head":"hydrant","relation":"used_for","tail":"fire","source_id":"s2"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kg_duplicate_triple_id:t1"):
        load_kg_strict(corpus)


def test_same_fact_same_provenance_is_rejected(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '\n'.join(
            [
                '{"head":"hose","relation":"used_for","tail":"fire","source_id":"s1"}',
                '{"head":"hose","relation":"used_for","tail":"fire","source_id":"s1"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=r"kg_duplicate_triple_provenance:hose\|used_for\|fire\|provenance_sha256=",
    ):
        load_kg_strict(corpus)


def test_same_fact_different_source_is_allowed(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '\n'.join(
            [
                '{"head":"hose","relation":"used_for","tail":"fire","source_id":"s1"}',
                '{"head":"hose","relation":"used_for","tail":"fire","source_id":"s2"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    kg = load_kg_strict(corpus)
    assert len(kg.triples) == 2


def test_same_fact_different_citation_is_allowed(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '\n'.join(
            [
                '{"head":"hose","relation":"used_for","tail":"fire","citation":"manual_a"}',
                '{"head":"hose","relation":"used_for","tail":"fire","citation":"manual_b"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    kg = load_kg_strict(corpus)
    assert len(kg.triples) == 2


def test_same_fact_different_source_chunk_is_allowed(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        "\n".join(
            [
                '{"head":"hose","relation":"used_for","tail":"fire","source_id":"s1","source_chunk_id":"c1"}',
                '{"head":"hose","relation":"used_for","tail":"fire","source_id":"s1","source_chunk_id":"c2"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    kg = load_kg_strict(corpus)
    assert len(kg.triples) == 2


def test_same_fact_evidence_reference_alias_is_same_provenance(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        "\n".join(
            [
                '{"head":"hose","relation":"used_for","tail":"fire","chunk_id":"c1"}',
                '{"head":"hose","relation":"used_for","tail":"fire","source_chunk_id":"c1"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kg_duplicate_triple_provenance"):
        load_kg_strict(corpus)


def test_same_fact_evidence_reference_list_order_is_same_provenance(
    tmp_path: Path,
) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        "\n".join(
            [
                '{"head":"hose","relation":"used_for","tail":"fire","source_chunk_ids":["c1","c2"]}',
                '{"head":"hose","relation":"used_for","tail":"fire","source_chunk_ids":["c2","c1"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kg_duplicate_triple_provenance"):
        load_kg_strict(corpus)


def test_duplicate_error_preserves_filename_and_original_line(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        "\n"
        '{"head":"hose","relation":"used_for","tail":"fire"}\n'
        "\n"
        '{"head":"hose","relation":"used_for","tail":"fire"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=r"triples.jsonl:line_4:first_line_2",
    ):
        load_kg_strict(corpus)


def test_entity_id_rejects_surrounding_whitespace(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1 ","name":"hose"}\n', encoding="utf-8"
    )
    with pytest.raises(
        ValueError,
        match="entity_identifier_must_not_have_surrounding_whitespace",
    ):
        load_kg_strict(corpus)


def test_relation_id_rejects_surrounding_whitespace(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "relations.jsonl").write_text(
        '{"relation_id":" r1","name":"used_for"}\n', encoding="utf-8"
    )
    with pytest.raises(
        ValueError,
        match="relation_identifier_must_not_have_surrounding_whitespace",
    ):
        load_kg_strict(corpus)


def test_triple_id_rejects_surrounding_whitespace(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '{"triple_id":"t1 ","head":"hose","relation":"used_for","tail":"fire"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="triple_identifier_must_not_have_surrounding_whitespace",
    ):
        load_kg_strict(corpus)


def test_chunk_id_rejects_surrounding_whitespace(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "evidence_chunks.jsonl").write_text(
        '{"chunk_id":"c1 ","text":"hose evidence","source_id":"s1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="evidence_chunk_identifier_must_not_have_surrounding_whitespace",
    ):
        load_kg_strict(corpus)


def test_triple_head_rejects_surrounding_whitespace(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose ","relation":"used_for","tail":"fire"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="triple_head_identifier_must_not_have_surrounding_whitespace",
    ):
        load_kg_strict(corpus)


def test_triple_tail_rejects_surrounding_whitespace(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"used_for","tail":" fire"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="triple_tail_identifier_must_not_have_surrounding_whitespace",
    ):
        load_kg_strict(corpus)


def test_identifier_rejects_control_characters(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e\\u00071","name":"hose"}\n', encoding="utf-8"
    )
    with pytest.raises(
        ValueError,
        match="entity_identifier_contains_control_character",
    ):
        load_kg_strict(corpus)


def test_identifier_rejects_interior_newline(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '{"head":"ho\\nse","relation":"used_for","tail":"fire"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="triple_head_identifier_contains_control_character",
    ):
        load_kg_strict(corpus)


def test_identifier_accepts_exact_integer(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":123,"name":"hose"}\n', encoding="utf-8"
    )
    kg = load_kg_strict(corpus)
    assert kg.entities[0]["entity_id"] == 123


def test_identifier_rejects_boolean(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":true,"name":"hose"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="entity_identifier_invalid_type"):
        load_kg_strict(corpus)


def test_identifier_rejects_float(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":1.0,"name":"hose"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="entity_identifier_invalid_type"):
        load_kg_strict(corpus)


def test_strict_identity_matches_runtime_identity(tmp_path: Path) -> None:
    from external_baselines.ekell_style.kg_loader import (
        entity_id,
        evidence_chunk_id,
        triple_id,
    )

    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"E1","name":"hose"}\n{"entity_id":42,"name":"hydrant"}\n',
        encoding="utf-8",
    )
    (corpus / "triples.jsonl").write_text(
        '{"triple_id":"T1","head":"hose","relation":"used_for","tail":"fire","source_id":"s1"}\n',
        encoding="utf-8",
    )
    kg = load_kg_strict(corpus)
    from external_baselines.ekell_style.kg_loader import _strict_identity

    for entity in kg.entities:
        assert _strict_identity("entities", entity)[1] == entity_id(entity)
    for triple in kg.triples:
        assert _strict_identity("triples", triple)[1] == triple_id(triple)
    for chunk in kg.evidence_chunks:
        assert _strict_identity("evidence_chunks", chunk)[1] == evidence_chunk_id(chunk)


def test_entity_alias_list_accepts_exact_strings(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":["fire hose","water line"]}\n',
        encoding="utf-8",
    )
    kg = load_kg_strict(corpus)
    assert kg.entities[0]["aliases"] == ["fire hose", "water line"]


def test_entity_alias_list_rejects_integer(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":["fire hose",123]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="entity_alias_must_be_string"):
        load_kg_strict(corpus)


def test_entity_alias_list_rejects_boolean(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":[true]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="entity_alias_must_be_string"):
        load_kg_strict(corpus)


def test_entity_alias_list_rejects_object(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":[{"x":1}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="entity_alias_must_be_string"):
        load_kg_strict(corpus)


def test_entity_alias_rejects_surrounding_whitespace(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":["fire hose "]}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="entity_alias_must_not_have_surrounding_whitespace",
    ):
        load_kg_strict(corpus)


def test_entity_alias_scalar_field_rejects_non_string(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":123}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="entity_alias_must_be_string"):
        load_kg_strict(corpus)


def test_entity_alias_legacy_delimited_string_is_accepted(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":"fire hose, water line"}\n',
        encoding="utf-8",
    )
    kg = load_kg_strict(corpus)
    assert kg.entities[0]["aliases"] == "fire hose, water line"


def test_entity_alias_legacy_delimited_empty_item_is_rejected(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":"fire hose,,water line"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="entity_alias_legacy_delimited_item_empty",
    ):
        load_kg_strict(corpus)


def test_triple_evidence_must_be_string(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"used_for","tail":"fire","evidence":123}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="triple_evidence_must_be_string"):
        load_kg_strict(corpus)


def test_triple_description_must_be_string(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"used_for","tail":"fire","description":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="triple_description_must_be_string"):
        load_kg_strict(corpus)


def test_triple_text_must_be_string(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"used_for","tail":"fire","text":["a","b"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="triple_text_must_be_string"):
        load_kg_strict(corpus)


def test_triple_content_must_be_string(tmp_path: Path) -> None:
    corpus = _strict_kg_dir(tmp_path)
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"used_for","tail":"fire","content":{"x":1}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="triple_content_must_be_string"):
        load_kg_strict(corpus)


def test_valid_strict_kg_does_not_require_lossy_string_coercion(
    tmp_path: Path,
) -> None:
    from external_baselines.ekell_style.kg_loader import (
        entity_aliases,
        triple_to_text,
    )

    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":["fire hose"]}\n',
        encoding="utf-8",
    )
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"used_for","tail":"fire","evidence":"manual section 2"}\n',
        encoding="utf-8",
    )
    kg = load_kg_strict(corpus)
    aliases = entity_aliases(kg.entities[0])
    assert "fire hose" in aliases
    assert all(type(a) is str for a in aliases)
    text = triple_to_text(kg.triples[0])
    assert text == "hose --used_for--> fire. manual section 2"


def test_lenient_triple_to_text_ignores_non_string_evidence() -> None:
    from external_baselines.ekell_style.kg_loader import triple_to_text

    text = triple_to_text(
        {"head": "hose", "relation": "used_for", "tail": "fire", "evidence": 123}
    )
    assert text == "hose --used_for--> fire"


def test_lenient_entity_aliases_skip_non_string_elements() -> None:
    from external_baselines.ekell_style.kg_loader import entity_aliases

    aliases = entity_aliases(
        {"entity_id": "e1", "name": "hose", "aliases": ["fire hose", 123, True]}
    )
    assert "fire hose" in aliases
    assert "123" not in aliases
    assert "True" not in aliases


def test_repository_dev_firekg_corpus_strict_compatibility() -> None:
    corpus = Path(__file__).resolve().parents[1] / "data" / "corpus"
    kg = load_kg_strict(corpus)
    assert kg.counts() == {
        "entities": 3,
        "relations": 2,
        "triples": 2,
        "evidence_chunks": 2,
    }


def test_strict_firekg_audit_reports_structure_only() -> None:
    from scripts.audit.audit_strict_firekg import audit_strict_firekg

    corpus = Path(__file__).resolve().parents[1] / "data" / "corpus"
    report = audit_strict_firekg(corpus)
    assert report["ok"] is True
    assert report["strict_loader_executed"] is True
    assert report["strict_loader_error"] is None
    assert report["counts"]["triples"] == 2
    assert report["duplicate_statistics"]["triples"] == 0
    assert report["schema_errors"] == []


def test_strict_firekg_audit_preserves_rejected_file_and_line(tmp_path: Path) -> None:
    from scripts.audit.audit_strict_firekg import audit_strict_firekg

    corpus = _strict_kg_dir(tmp_path)
    (corpus / "entities.jsonl").write_text(
        '{"entity_id":"e1","name":"hose","aliases":[123]}\n',
        encoding="utf-8",
    )
    report = audit_strict_firekg(corpus)
    assert report["ok"] is False
    assert report["rejected_field_type_statistics"]["entities"] == {
        "entity_alias_must_be_string": 1
    }
    assert report["schema_errors"][0]["file"] == "entities.jsonl"
    assert report["schema_errors"][0]["line"] == 1
