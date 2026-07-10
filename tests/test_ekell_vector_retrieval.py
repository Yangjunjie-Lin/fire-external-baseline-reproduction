from __future__ import annotations

import pytest

from external_baselines.ekell_style.embedding_backends import (
    EmbeddingBackendError,
    HashEmbeddingBackend,
)
from external_baselines.ekell_style.kg_loader import FireKG
from external_baselines.ekell_style.vector_index import VectorIndex
from external_baselines.ekell_style.vector_retriever import VectorRetriever


def _kg() -> FireKG:
    return FireKG(
        triples=[
            {
                "triple_id": "t-fire-smoke",
                "head": "electrical fire",
                "relation": "produces",
                "tail": "smoke",
                "source_chunk_ids": ["c1"],
            },
            {
                "triple_id": "t-water-cooling",
                "head": "water",
                "relation": "provides",
                "tail": "cooling",
                "source_chunk_ids": ["c2"],
            },
        ],
        evidence_chunks=[
            {
                "chunk_id": "c1",
                "text": "Electrical fire can produce dense smoke.",
                "source_id": "manual",
            }
        ],
    )


def test_smoke_index_build_and_query(tmp_path):
    backend = HashEmbeddingBackend(dimension=64)
    index = VectorIndex.from_kg(_kg(), backend)
    assert index.metadata["actual_embedding_used"] is False
    assert index.metadata["smoke_fallback_used"] is True
    for key in (
        "embedding_model",
        "model_version",
        "dimension",
        "corpus_checksum",
        "kg_checksum",
        "build_timestamp",
        "backend",
        "index_checksum",
        "package_versions",
    ):
        assert index.metadata[key]

    path = tmp_path / "index.json"
    index.save(path)
    loaded = VectorIndex.load(path)
    contexts = VectorRetriever(loaded, backend).retrieve(
        "electrical fire smoke", top_k=2
    )
    assert contexts
    assert set(contexts[0]) == {
        "context_id",
        "text",
        "source_id",
        "citation",
        "score",
        "metadata",
    }
    assert contexts[0]["metadata"]["provenance_id"]


@pytest.mark.parametrize("guard", ["paper_final", "reject_smoke"])
def test_paper_final_or_reject_smoke_refuses_hash_backend(guard):
    backend = HashEmbeddingBackend()
    with pytest.raises(EmbeddingBackendError):
        VectorIndex.from_kg(_kg(), backend, **{guard: True})
