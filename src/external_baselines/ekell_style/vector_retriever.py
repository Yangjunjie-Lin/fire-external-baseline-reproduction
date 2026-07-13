from __future__ import annotations

"""E-KELL-native vector retriever (independent of dense/hybrid RAG)."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from external_baselines.common.schema import RetrievedContext
from external_baselines.common.text_utils import compact_text
from external_baselines.ekell_style.embedding_backends import (
    EmbeddingBackend,
    EmbeddingBackendError,
    l2_normalize_vector,
    validate_embedding_backend,
)
from external_baselines.ekell_style.kg_loader import FireKG
from external_baselines.ekell_style.vector_index import VectorIndex


class VectorRetriever:
    def __init__(
        self,
        index: VectorIndex,
        backend: EmbeddingBackend,
        *,
        max_context_chars: int = 1200,
        paper_final: bool = False,
        reject_smoke: bool = False,
    ) -> None:
        validate_embedding_backend(backend, paper_final=paper_final, reject_smoke=reject_smoke)
        if (paper_final or reject_smoke) and (
            index.metadata.get("smoke_fallback_used")
            or not index.metadata.get("actual_embedding_used")
        ):
            raise EmbeddingBackendError("A smoke-built vector index is forbidden for this run.")
        if index.metadata.get("embedding_model") != backend.model_name:
            raise ValueError("Index and query embedding model names differ.")
        index_dim = int(index.metadata.get("dimension") or 0)
        if backend.dimension in (0, None):
            backend.dimension = index_dim
        elif int(backend.dimension) != index_dim:
            raise ValueError("Index and query embedding dimensions differ.")
        self.index = index
        self.backend = backend
        self.max_context_chars = max_context_chars

    @classmethod
    def from_kg(
        cls,
        kg: FireKG,
        backend: EmbeddingBackend,
        *,
        paper_final: bool = False,
        reject_smoke: bool = False,
        max_context_chars: int = 1200,
        **index_kwargs: Any,
    ) -> "VectorRetriever":
        index = VectorIndex.from_kg(
            kg,
            backend,
            paper_final=paper_final,
            reject_smoke=reject_smoke,
            **index_kwargs,
        )
        return cls(
            index,
            backend,
            max_context_chars=max_context_chars,
            paper_final=paper_final,
            reject_smoke=reject_smoke,
        )

    @classmethod
    def from_index_directory(
        cls,
        index_dir: str | Path,
        backend: EmbeddingBackend,
        *,
        paper_final: bool = False,
        reject_smoke: bool = False,
        max_context_chars: int = 1200,
        expected_kg_checksum: str | None = None,
        expected_corpus_checksum: str | None = None,
        expected_dimension: int | None = None,
    ) -> "VectorRetriever":
        index = VectorIndex.load_directory(
            index_dir,
            expected_backend=backend.backend,
            expected_model_name=backend.model_name,
            expected_model_version=backend.model_version,
            expected_dimension=expected_dimension or (backend.dimension or None),
            expected_kg_checksum=expected_kg_checksum,
            expected_corpus_checksum=expected_corpus_checksum,
            require_real_embedding=bool(paper_final or reject_smoke),
        )
        # Align backend dimension with loaded index when backend was lazy (dim=0).
        if backend.dimension in (0, None):
            backend.dimension = int(index.metadata.get("dimension") or 0)
        return cls(
            index,
            backend,
            max_context_chars=max_context_chars,
            paper_final=paper_final,
            reject_smoke=reject_smoke,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return serializable dictionaries matching ``RetrievedContext``."""

        if not query.strip() or top_k <= 0:
            return []
        query_vector = self.backend.embed_query(query)
        normalize = self.index.metadata.get("normalize_embeddings")
        if type(normalize) is not bool:
            raise EmbeddingBackendError("ekell_index_normalize_embeddings_must_be_bool")
        if normalize:
            query_vector = l2_normalize_vector(query_vector)
        results = self.index.search(query_vector, top_k=top_k, min_score=min_score)
        contexts: list[dict[str, Any]] = []
        for rank, (document, score) in enumerate(results, start=1):
            metadata = {
                **document.metadata,
                "retrieval_backend": self.index.metadata["backend"],
                "embedding_model": self.index.metadata["embedding_model"],
                "model_version": self.index.metadata["model_version"],
                "vector_score": round(float(score), 8),
                "rank": rank,
                "index_checksum": self.index.metadata["index_checksum"],
                "actual_embedding_used": self.index.metadata["actual_embedding_used"],
                "smoke_fallback_used": self.index.metadata["smoke_fallback_used"],
            }
            context = RetrievedContext(
                context_id=document.document_id,
                text=compact_text(document.text, self.max_context_chars),
                source_id=document.source_id,
                citation=document.citation or document.source_id or document.document_id,
                score=round(float(score), 8),
                metadata=metadata,
            )
            contexts.append(asdict(context))
        return contexts


EKELLVectorRetriever = VectorRetriever
