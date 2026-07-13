"""Dense real-backend index tests with injected fake embedding model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baselines.common.checksums import sha256_file
from external_baselines.dense_rag.pipeline import DenseRetriever, build_dense_index
from external_baselines.retrieval.dense_index import (
    DenseIndexError,
    load_dense_index,
    validate_dense_index_integrity_for_freeze,
)
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


def _evidence(tmp_path: Path) -> Path:
    path = tmp_path / "evidence_chunks.jsonl"
    path.write_text(
        '{"chunk_id":"c1","text":"alpha fire hose","source_id":"s1"}\n'
        '{"chunk_id":"c2","text":"beta smoke alarm","source_id":"s2"}\n'
        '{"chunk_id":"c3","text":"gamma evacuation route","source_id":"s3"}\n',
        encoding="utf-8",
    )
    return path


def _build_real_dense_index(tmp_path: Path) -> Path:
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
        corpus_checksum=sha256_file(evidence),
    )
    return index_dir


def _read_manifest(index_dir: Path) -> dict:
    return json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(index_dir: Path, manifest: dict) -> None:
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_dense_real_backend_builds_index_with_injected_model(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "idx"
    index = build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=FakeEmbeddingModel(8),
        reject_smoke=True,
    )
    assert index.backend == "text2vec"
    assert index.build_manifest.get("actual_embedding_used") is True
    assert (index_dir / "documents.jsonl").is_file()
    assert (index_dir / "embeddings.npy").is_file()
    assert (index_dir / "index_manifest.json").is_file()


def test_dense_index_persists_documents_and_numpy_embeddings(tmp_path: Path) -> None:
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
    payload = load_dense_index(index_dir)
    assert len(payload["documents"]) == 3
    assert len(payload["embeddings"]) == 3
    assert payload["dimension"] == 8


def test_dense_index_load_validates_checksums(tmp_path: Path) -> None:
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
    # Corrupt embeddings checksum by rewriting documents without updating manifest.
    docs = (index_dir / "documents.jsonl").read_text(encoding="utf-8")
    (index_dir / "documents.jsonl").write_text(docs + '{"chunk_id":"cX","text":"extra"}\n', encoding="utf-8")
    with pytest.raises(DenseIndexError):
        load_dense_index(index_dir)


def test_dense_query_uses_same_embedding_backend(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "idx"
    fake = FakeEmbeddingModel(8)
    index = build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=fake,
        reject_smoke=True,
    )
    emb = create_embedding_backend(
        "text2vec",
        model_name="fake/bge",
        model_version="v-test",
        dimension=8,
        reject_smoke=True,
        model=fake,
    )
    retriever = DenseRetriever(index, embedding_backend=emb)
    hits = retriever.retrieve("alpha fire", top_k=2)
    assert hits
    assert hits[0].metadata.get("embedding_model") == "fake/bge"
    assert hits[0].metadata.get("dense_rank") == 1


def test_dense_rejects_model_version_mismatch(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "idx"
    build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v1",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=FakeEmbeddingModel(8),
        reject_smoke=True,
    )
    with pytest.raises(DenseIndexError, match="model_version"):
        load_dense_index(index_dir, expected_model_version="v2")


def test_dense_rejects_corpus_checksum_mismatch(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "idx"
    build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v1",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=FakeEmbeddingModel(8),
        reject_smoke=True,
        corpus_checksum="abc",
    )
    with pytest.raises(DenseIndexError, match="corpus_checksum"):
        load_dense_index(index_dir, expected_corpus_checksum="zzz")


def test_dense_rejects_smoke_in_real_mode() -> None:
    with pytest.raises(Exception):
        create_embedding_backend("smoke_hash_embedding", reject_smoke=True, dimension=8)


def test_dense_accepts_dimension_field() -> None:
    from external_baselines.retrieval.embedding_backends import resolve_dimension

    assert resolve_dimension({"dimension": 1024}) == 1024
    assert resolve_dimension({"dim": 64}) == 64


def test_dense_freeze_integrity_accepts_valid_index(tmp_path: Path) -> None:
    index_dir = _build_real_dense_index(tmp_path)

    result = validate_dense_index_integrity_for_freeze(
        index_dir,
        expected_backend="text2vec",
        expected_model_name="fake/bge",
        expected_model_version="v-test",
        expected_dimension=8,
    )

    assert result["index_checksum"]
    assert result["index_manifest_sha256"]
    assert result["actual_embedding_used"] is True
    assert result["smoke_fallback_used"] is False


@pytest.mark.parametrize(
    "field",
    [
        "index_checksum",
        "documents_file_checksum",
        "embeddings_checksum",
    ],
)
def test_dense_freeze_integrity_requires_required_checksums(tmp_path: Path, field: str) -> None:
    index_dir = _build_real_dense_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest.pop(field)
    _write_manifest(index_dir, manifest)

    with pytest.raises(DenseIndexError, match=field):
        validate_dense_index_integrity_for_freeze(index_dir)


def test_dense_freeze_integrity_rejects_tampered_documents_jsonl(tmp_path: Path) -> None:
    index_dir = _build_real_dense_index(tmp_path)
    docs = index_dir / "documents.jsonl"
    docs.write_text(docs.read_text(encoding="utf-8").replace("alpha", "tampered", 1), encoding="utf-8")

    with pytest.raises(DenseIndexError, match="documents_file_checksum"):
        validate_dense_index_integrity_for_freeze(index_dir)


def test_dense_freeze_integrity_rejects_tampered_embeddings_same_shape(tmp_path: Path) -> None:
    import numpy as np

    index_dir = _build_real_dense_index(tmp_path)
    emb = index_dir / "embeddings.npy"
    arr = np.load(emb)
    arr[0, 0] = arr[0, 0] + 0.125
    np.save(emb, arr.astype("float32"))

    with pytest.raises(DenseIndexError, match="embeddings_checksum"):
        validate_dense_index_integrity_for_freeze(index_dir)


def test_dense_freeze_integrity_rejects_semantic_documents_checksum_mismatch(tmp_path: Path) -> None:
    index_dir = _build_real_dense_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["documents_checksum"] = "0" * 64
    _write_manifest(index_dir, manifest)

    with pytest.raises(DenseIndexError, match="documents_checksum"):
        validate_dense_index_integrity_for_freeze(index_dir)


def test_dense_freeze_integrity_rejects_index_checksum_mismatch(tmp_path: Path) -> None:
    index_dir = _build_real_dense_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["index_checksum"] = "1" * 64
    _write_manifest(index_dir, manifest)

    with pytest.raises(DenseIndexError, match="index_checksum"):
        validate_dense_index_integrity_for_freeze(index_dir)


def test_dense_freeze_integrity_rejects_invalid_checksum_format(tmp_path: Path) -> None:
    index_dir = _build_real_dense_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["embeddings_checksum"] = "deadbeef"
    _write_manifest(index_dir, manifest)

    with pytest.raises(DenseIndexError, match="embeddings_checksum"):
        validate_dense_index_integrity_for_freeze(index_dir)


def test_dense_freeze_integrity_requires_actual_embedding_true(tmp_path: Path) -> None:
    index_dir = _build_real_dense_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["actual_embedding_used"] = False
    _write_manifest(index_dir, manifest)

    with pytest.raises(DenseIndexError, match="actual_embedding_used"):
        validate_dense_index_integrity_for_freeze(index_dir)


def test_dense_freeze_integrity_requires_smoke_fallback_false(tmp_path: Path) -> None:
    index_dir = _build_real_dense_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["smoke_fallback_used"] = True
    _write_manifest(index_dir, manifest)

    with pytest.raises(DenseIndexError, match="smoke_fallback_used"):
        validate_dense_index_integrity_for_freeze(index_dir)
