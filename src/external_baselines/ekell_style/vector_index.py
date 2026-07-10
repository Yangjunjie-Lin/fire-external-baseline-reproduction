from __future__ import annotations

"""Auditable in-process vector index for E-KELL KG segments."""

import json
import math
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from external_baselines.common.checksums import sha256_json
from external_baselines.ekell_style.embedding_backends import (
    EmbeddingBackend,
    embedding_package_versions,
    validate_embedding_backend,
)
from external_baselines.ekell_style.kg_loader import (
    FireKG,
    evidence_chunk_id,
    evidence_citation,
    evidence_source_id,
    evidence_text,
    triple_id,
    triple_parts,
    triple_to_text,
)


class VectorIndexError(RuntimeError):
    pass


@dataclass
class VectorDocument:
    document_id: str
    text: str
    source_id: str | None = None
    citation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def kg_segments(kg: FireKG) -> list[VectorDocument]:
    """Materialize triple texts and evidence chunks with stable provenance IDs."""

    documents: list[VectorDocument] = []
    for index, row in enumerate(kg.triples):
        tid = triple_id(row, index)
        head, relation, tail = triple_parts(row)
        source_id = str(row.get("source_id") or row.get("source") or "kg_triple")
        chunk_ids = _source_chunk_ids(row)
        documents.append(
            VectorDocument(
                document_id=tid,
                text=triple_to_text(row),
                source_id=source_id,
                citation=str(row.get("citation") or source_id or tid),
                metadata={
                    "kind": "kg_triple",
                    "provenance_id": tid,
                    "triple_id": tid,
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "source_chunk_ids": chunk_ids,
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"text", "content", "evidence"}
                    },
                },
            )
        )
    for index, row in enumerate(kg.evidence_chunks):
        chunk_id = evidence_chunk_id(row, index)
        text = evidence_text(row)
        if not text.strip():
            continue
        documents.append(
            VectorDocument(
                document_id=chunk_id,
                text=text,
                source_id=evidence_source_id(row),
                citation=evidence_citation(row),
                metadata={
                    "kind": "evidence_chunk",
                    "provenance_id": chunk_id,
                    "source_chunk_ids": [chunk_id],
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"text", "content", "chunk", "body"}
                    },
                },
            )
        )
    return documents


def _source_chunk_ids(row: dict[str, Any]) -> list[str]:
    value = (
        row.get("source_chunk_ids")
        or row.get("evidence_chunk_ids")
        or row.get("chunk_ids")
        or row.get("source_chunk_id")
        or row.get("chunk_id")
        or []
    )
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _coerce_document(value: VectorDocument | dict[str, Any], index: int) -> VectorDocument:
    if isinstance(value, VectorDocument):
        return value
    text = str(value.get("text") or value.get("content") or "")
    document_id = str(
        value.get("document_id")
        or value.get("context_id")
        or value.get("triple_id")
        or value.get("chunk_id")
        or value.get("id")
        or f"segment_{index}"
    )
    excluded = {"document_id", "context_id", "text", "content", "source_id", "citation", "metadata"}
    return VectorDocument(
        document_id=document_id,
        text=text,
        source_id=str(value["source_id"]) if value.get("source_id") is not None else None,
        citation=str(value["citation"]) if value.get("citation") is not None else None,
        metadata={**dict(value.get("metadata") or {}), **{k: v for k, v in value.items() if k not in excluded}},
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise VectorIndexError("Query and index embedding dimensions differ.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class VectorIndex:
    """Small exact cosine index with complete reproducibility metadata."""

    def __init__(
        self,
        documents: Sequence[VectorDocument],
        vectors: Sequence[Sequence[float]],
        metadata: dict[str, Any],
    ) -> None:
        self.documents = list(documents)
        self.vectors = [[float(value) for value in vector] for vector in vectors]
        self.metadata = dict(metadata)
        if len(self.documents) != len(self.vectors):
            raise VectorIndexError("Document and vector counts differ.")
        dimension = int(self.metadata.get("dimension") or 0)
        if self.vectors and any(len(vector) != dimension for vector in self.vectors):
            raise VectorIndexError("Stored vector dimensions do not match index metadata.")

    @classmethod
    def build(
        cls,
        documents_or_kg: FireKG | Iterable[VectorDocument | dict[str, Any]],
        backend: EmbeddingBackend,
        *,
        corpus_checksum: str | None = None,
        kg_checksum: str | None = None,
        paper_final: bool = False,
        reject_smoke: bool = False,
        build_timestamp: str | None = None,
    ) -> "VectorIndex":
        validate_embedding_backend(backend, paper_final=paper_final, reject_smoke=reject_smoke)
        if isinstance(documents_or_kg, FireKG):
            kg = documents_or_kg
            documents = kg_segments(kg)
            canonical_kg = {
                "entities": kg.entities,
                "relations": kg.relations,
                "triples": kg.triples,
                "evidence_chunks": kg.evidence_chunks,
            }
            computed_kg_checksum = sha256_json(canonical_kg)
            computed_corpus_checksum = sha256_json(
                {"triples": kg.triples, "evidence_chunks": kg.evidence_chunks}
            )
        else:
            documents = [
                _coerce_document(value, index)
                for index, value in enumerate(documents_or_kg)
            ]
            payload = [document.to_dict() for document in documents]
            computed_kg_checksum = sha256_json(payload)
            computed_corpus_checksum = computed_kg_checksum
        vectors = backend.embed_documents([document.text for document in documents])
        if documents and not vectors:
            raise VectorIndexError("Embedding backend returned no vectors.")
        dimension = len(vectors[0]) if vectors else int(backend.dimension)
        if any(len(vector) != dimension for vector in vectors):
            raise VectorIndexError("Embedding backend returned inconsistent dimensions.")
        backend.dimension = dimension
        package_versions = {
            "python": platform.python_version(),
            **embedding_package_versions(),
        }
        metadata = {
            "embedding_model": backend.model_name,
            "model_version": backend.model_version,
            "dimension": dimension,
            "corpus_checksum": corpus_checksum or computed_corpus_checksum,
            "kg_checksum": kg_checksum or computed_kg_checksum,
            "build_timestamp": build_timestamp or datetime.now(timezone.utc).isoformat(),
            "backend": backend.backend,
            "package_versions": package_versions,
            "actual_embedding_used": bool(backend.actual_embedding_used),
            "smoke_fallback_used": bool(backend.smoke_fallback_used),
            "document_count": len(documents),
        }
        checksum_payload = {
            "metadata": metadata,
            "documents": [document.to_dict() for document in documents],
            "vectors": vectors,
        }
        metadata["index_checksum"] = sha256_json(checksum_payload)
        return cls(documents, vectors, metadata)

    @classmethod
    def from_kg(
        cls, kg: FireKG, backend: EmbeddingBackend, **kwargs: Any
    ) -> "VectorIndex":
        return cls.build(kg, backend, **kwargs)

    def search(
        self, query_vector: Sequence[float], *, top_k: int = 5, min_score: float | None = None
    ) -> list[tuple[VectorDocument, float]]:
        if top_k <= 0:
            return []
        ranked = sorted(
            (
                (document, _cosine(query_vector, vector))
                for document, vector in zip(self.documents, self.vectors)
            ),
            key=lambda item: (-item[1], item[0].document_id),
        )
        if min_score is not None:
            ranked = [item for item in ranked if item[1] >= min_score]
        return ranked[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "documents": [document.to_dict() for document in self.documents],
            "vectors": self.vectors,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, verify_checksum: bool = True) -> "VectorIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = dict(payload["metadata"])
        documents = [VectorDocument(**document) for document in payload["documents"]]
        vectors = payload["vectors"]
        if verify_checksum:
            expected = metadata.get("index_checksum")
            checksum_metadata = {k: v for k, v in metadata.items() if k != "index_checksum"}
            actual = sha256_json(
                {
                    "metadata": checksum_metadata,
                    "documents": [document.to_dict() for document in documents],
                    "vectors": vectors,
                }
            )
            if not expected or actual != expected:
                raise VectorIndexError("Vector index checksum verification failed.")
        return cls(documents, vectors, metadata)
