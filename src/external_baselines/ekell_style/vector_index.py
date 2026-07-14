from __future__ import annotations

"""Auditable in-process vector index for E-KELL KG segments."""

import json
import math
import platform
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from external_baselines.common.checksums import sha256_file, sha256_json
from external_baselines.common.io import read_json, read_jsonl
from external_baselines.ekell_style.embedding_backends import (
    EmbeddingBackend,
    embedding_package_versions,
    l2_normalize_vector,
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


SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def require_ekell_formal_embedding_manifest(manifest: dict[str, Any]) -> None:
    if "actual_embedding_used" not in manifest:
        raise VectorIndexError("actual_embedding_used_missing")
    if manifest["actual_embedding_used"] is not True:
        raise VectorIndexError("actual_embedding_used_must_be_true")
    if "smoke_fallback_used" not in manifest:
        raise VectorIndexError("smoke_fallback_used_missing")
    if manifest["smoke_fallback_used"] is not False:
        raise VectorIndexError("smoke_fallback_used_must_be_false")


@dataclass
class VectorDocument:
    document_id: str
    text: str
    source_id: str | None = None
    citation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ekell_documents_semantic_checksum(documents: Sequence[VectorDocument]) -> str:
    return sha256_json([document.to_dict() for document in documents])


def ekell_index_identity_checksum(
    *,
    backend: str,
    model_name: str,
    model_version: str,
    dimension: int,
    normalize_embeddings: bool,
    document_count: int,
    documents_checksum: str,
    embeddings_checksum: str,
    kg_checksum: str,
    corpus_checksum: str,
) -> str:
    if type(normalize_embeddings) is not bool:
        raise TypeError("ekell_index_normalize_embeddings_must_be_bool")
    return sha256_json(
        {
            "backend": backend,
            "model_name": model_name,
            "model_version": model_version,
            "dimension": dimension,
            "normalize_embeddings": normalize_embeddings,
            "document_count": document_count,
            "documents_checksum": documents_checksum,
            "embeddings_checksum": embeddings_checksum,
            "kg_checksum": kg_checksum,
            "corpus_checksum": corpus_checksum,
        }
    )


def _require_plain_file(path: Path, *, code: str) -> None:
    if not path.is_file():
        raise VectorIndexError(code)


def _require_manifest_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VectorIndexError("ekell_index_manifest_must_be_object")
    return value


def _require_exact_nonempty_string(manifest: dict[str, Any], field: str) -> str:
    value = manifest.get(field)
    if type(value) is not str or not value:
        raise VectorIndexError(f"ekell_index_{field}_must_be_nonempty_string")
    return value


def _require_positive_int(manifest: dict[str, Any], field: str) -> int:
    value = manifest.get(field)
    if type(value) is not int or value <= 0:
        raise VectorIndexError(f"ekell_index_{field}_must_be_positive_int")
    return value


def _require_exact_bool(manifest: dict[str, Any], field: str) -> bool:
    value = manifest.get(field)
    if type(value) is not bool:
        raise VectorIndexError(f"ekell_index_{field}_must_be_bool")
    return value


def _read_exact_bool(mapping: dict[str, Any], field: str, *, default: bool | None = None) -> bool:
    value = mapping.get(field)
    if value is None and default is not None:
        return default
    if type(value) is not bool:
        raise VectorIndexError(f"ekell_index_{field}_must_be_bool")
    return value


def _require_sha256(manifest: dict[str, Any], field: str) -> str:
    value = manifest.get(field)
    if type(value) is not str or not SHA256_HEX_RE.fullmatch(value):
        raise VectorIndexError(f"ekell_index_{field}_invalid")
    return value


def _read_vector_documents_strict(path: Path) -> list[VectorDocument]:
    documents: list[VectorDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise VectorIndexError(f"ekell_index_documents_invalid_json:line_{line_no}") from exc
            if not isinstance(value, dict):
                raise VectorIndexError(f"ekell_index_documents_record_must_be_object:line_{line_no}")
            try:
                documents.append(VectorDocument(**value))
            except TypeError as exc:
                raise VectorIndexError(f"ekell_index_documents_invalid_record:line_{line_no}") from exc
    if not documents:
        raise VectorIndexError("ekell_index_documents_empty")
    return documents


def _validate_ekell_embedding_values(
    arr: Any,
    *,
    normalize_embeddings: bool,
    chunk_rows: int = 4096,
) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise VectorIndexError("numpy is required to validate E-KELL embeddings (.npy).") from exc

    row_count = int(arr.shape[0])
    for start in range(0, row_count, chunk_rows):
        block = np.asarray(arr[start : start + chunk_rows])
        if not np.isfinite(block).all():
            raise VectorIndexError("ekell_index_embeddings_non_finite")
        norms = np.linalg.norm(block, axis=1)
        if (norms <= 1e-12).any():
            raise VectorIndexError("ekell_index_zero_embedding_vector")
        if normalize_embeddings and not np.allclose(
            norms,
            1.0,
            rtol=1e-4,
            atol=1e-5,
        ):
            raise VectorIndexError("ekell_index_normalized_embeddings_unit_norm_mismatch")


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
        normalize_embeddings: bool = True,
        build_timestamp: str | None = None,
    ) -> "VectorIndex":
        if type(normalize_embeddings) is not bool:
            raise VectorIndexError("ekell_index_normalize_embeddings_must_be_bool")
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
        if normalize_embeddings:
            vectors = [l2_normalize_vector(vector) for vector in vectors]
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
            "normalize_embeddings": normalize_embeddings,
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

    def save_directory(self, index_dir: str | Path) -> dict[str, Any]:
        """Persist as documents.jsonl + embeddings.npy + index_manifest.json."""
        import os
        import tempfile

        from external_baselines.common.io import ensure_dir, write_jsonl

        index_dir = Path(index_dir)
        ensure_dir(index_dir)
        docs_path = index_dir / "documents.jsonl"
        emb_path = index_dir / "embeddings.npy"
        manifest_path = index_dir / "index_manifest.json"

        documents_payload = [document.to_dict() for document in self.documents]
        write_jsonl(docs_path, documents_payload)
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise VectorIndexError("numpy is required to persist E-KELL embeddings (.npy).") from exc
        np.save(emb_path, np.asarray(self.vectors, dtype="float32"))

        docs_file_checksum = sha256_file(docs_path)
        embeddings_checksum = sha256_file(emb_path)
        documents_checksum = ekell_documents_semantic_checksum(self.documents)
        meta = dict(self.metadata)
        dimension = int(meta.get("dimension") or (len(self.vectors[0]) if self.vectors else 0))
        index_checksum = ekell_index_identity_checksum(
            backend=str(meta.get("backend")),
            model_name=str(meta.get("embedding_model") or meta.get("model_name")),
            model_version=str(meta.get("model_version")),
            dimension=dimension,
            normalize_embeddings=_read_exact_bool(meta, "normalize_embeddings"),
            document_count=len(self.documents),
            documents_checksum=documents_checksum,
            embeddings_checksum=embeddings_checksum,
            kg_checksum=str(meta.get("kg_checksum")),
            corpus_checksum=str(meta.get("corpus_checksum")),
        )
        manifest = {
            "index_type": "ekell_kg_vector_index",
            "backend": meta.get("backend"),
            "model_name": meta.get("embedding_model") or meta.get("model_name"),
            "model_version": meta.get("model_version"),
            "dimension": dimension,
            "normalize_embeddings": _read_exact_bool(meta, "normalize_embeddings"),
            "document_count": len(self.documents),
            "kg_checksum": meta.get("kg_checksum"),
            "corpus_checksum": meta.get("corpus_checksum"),
            "documents_checksum": documents_checksum,
            "documents_file_checksum": docs_file_checksum,
            "embeddings_checksum": embeddings_checksum,
            "index_checksum": index_checksum,
            "actual_embedding_used": _read_exact_bool(meta, "actual_embedding_used"),
            "smoke_fallback_used": _read_exact_bool(meta, "smoke_fallback_used"),
            "package_versions": meta.get("package_versions") or {},
            "index_dir": str(index_dir).replace("\\", "/"),
        }
        fd, tmp_name = tempfile.mkstemp(prefix=manifest_path.name, dir=str(index_dir))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(manifest_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        self.metadata = {**meta, **manifest}
        return manifest

    @classmethod
    def validate_directory_for_freeze(
        cls,
        index_dir: str | Path,
        *,
        expected_backend: str | None = None,
        expected_model_name: str | None = None,
        expected_model_version: str | None = None,
        expected_dimension: int | None = None,
        expected_kg_checksum: str | None = None,
        expected_corpus_checksum: str | None = None,
        expected_normalize_embeddings: bool | None = None,
    ) -> dict[str, Any]:
        index_dir = Path(index_dir)
        if index_dir.is_file():
            if index_dir.suffix.lower() == ".json":
                raise VectorIndexError("legacy_ekell_json_forbidden_in_formal")
            raise VectorIndexError("ekell_index_path_not_directory")
        if not index_dir.is_dir():
            raise VectorIndexError("ekell_index_path_not_directory")

        docs_path = index_dir / "documents.jsonl"
        emb_path = index_dir / "embeddings.npy"
        manifest_path = index_dir / "index_manifest.json"
        _require_plain_file(docs_path, code="ekell_index_documents_missing")
        _require_plain_file(emb_path, code="ekell_index_embeddings_missing")
        _require_plain_file(manifest_path, code="ekell_index_manifest_missing")

        manifest = _require_manifest_object(read_json(manifest_path))
        index_type = _require_exact_nonempty_string(manifest, "index_type")
        if index_type != "ekell_kg_vector_index":
            raise VectorIndexError("ekell_index_type_mismatch")
        backend = _require_exact_nonempty_string(manifest, "backend")
        model_name = _require_exact_nonempty_string(manifest, "model_name")
        model_version = _require_exact_nonempty_string(manifest, "model_version")
        dimension = _require_positive_int(manifest, "dimension")
        document_count = _require_positive_int(manifest, "document_count")
        normalize_embeddings = _require_exact_bool(manifest, "normalize_embeddings")
        kg_checksum = _require_sha256(manifest, "kg_checksum")
        corpus_checksum = _require_sha256(manifest, "corpus_checksum")
        documents_checksum = _require_sha256(manifest, "documents_checksum")
        documents_file_checksum = _require_sha256(manifest, "documents_file_checksum")
        embeddings_checksum = _require_sha256(manifest, "embeddings_checksum")
        index_checksum = _require_sha256(manifest, "index_checksum")
        actual_embedding_used = _require_exact_bool(manifest, "actual_embedding_used")
        smoke_fallback_used = _require_exact_bool(manifest, "smoke_fallback_used")
        if actual_embedding_used is not True:
            raise VectorIndexError("actual_embedding_used_must_be_true")
        if smoke_fallback_used is not False:
            raise VectorIndexError("smoke_fallback_used_must_be_false")

        if expected_backend and backend != str(expected_backend):
            raise VectorIndexError("backend mismatch.")
        if expected_model_name and model_name != str(expected_model_name):
            raise VectorIndexError("model_name mismatch.")
        if expected_model_version and model_version != str(expected_model_version):
            raise VectorIndexError("model_version mismatch.")
        if expected_dimension is not None and dimension != int(expected_dimension):
            raise VectorIndexError("dimension mismatch.")
        if expected_kg_checksum and kg_checksum != str(expected_kg_checksum):
            raise VectorIndexError("kg_checksum mismatch.")
        if expected_corpus_checksum and corpus_checksum != str(expected_corpus_checksum):
            raise VectorIndexError("corpus_checksum mismatch.")
        if expected_normalize_embeddings is not None:
            if type(expected_normalize_embeddings) is not bool:
                raise VectorIndexError("ekell_index_expected_normalize_embeddings_must_be_bool")
            if normalize_embeddings is not expected_normalize_embeddings:
                raise VectorIndexError("ekell_index_normalize_embeddings_mismatch")

        documents = _read_vector_documents_strict(docs_path)
        if len(documents) != document_count:
            raise VectorIndexError("document_count does not match documents.jsonl length.")
        actual_documents_file_checksum = sha256_file(docs_path)
        if actual_documents_file_checksum != documents_file_checksum:
            raise VectorIndexError("documents_file_checksum mismatch.")
        actual_embeddings_checksum = sha256_file(emb_path)
        if actual_embeddings_checksum != embeddings_checksum:
            raise VectorIndexError("embeddings_checksum mismatch.")
        recomputed_documents_checksum = ekell_documents_semantic_checksum(documents)
        if recomputed_documents_checksum != documents_checksum:
            raise VectorIndexError("documents_checksum mismatch.")

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise VectorIndexError("numpy is required to validate E-KELL embeddings (.npy).") from exc
        arr = np.load(emb_path, mmap_mode="r")
        if len(arr.shape) != 2:
            raise VectorIndexError("embeddings.npy must be a two-dimensional array.")
        row_count, dim = int(arr.shape[0]), int(arr.shape[1])
        if row_count <= 0 or dim <= 0:
            raise VectorIndexError("embeddings.npy must have positive row count and dimension.")
        if row_count != document_count:
            raise VectorIndexError("embeddings.npy row count does not match documents.")
        if dim != dimension:
            raise VectorIndexError("Manifest dimension does not match embeddings.npy.")
        _validate_ekell_embedding_values(arr, normalize_embeddings=normalize_embeddings)

        recomputed_index_checksum = ekell_index_identity_checksum(
            backend=backend,
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            normalize_embeddings=normalize_embeddings,
            document_count=document_count,
            documents_checksum=recomputed_documents_checksum,
            embeddings_checksum=actual_embeddings_checksum,
            kg_checksum=kg_checksum,
            corpus_checksum=corpus_checksum,
        )
        if recomputed_index_checksum != index_checksum:
            raise VectorIndexError("index_checksum mismatch.")

        return {
            "index_type": index_type,
            "index_dir": str(index_dir).replace("\\", "/"),
            "backend": backend,
            "model_name": model_name,
            "model_version": model_version,
            "dimension": dimension,
            "normalize_embeddings": normalize_embeddings,
            "document_count": document_count,
            "kg_checksum": kg_checksum,
            "corpus_checksum": corpus_checksum,
            "documents_checksum": recomputed_documents_checksum,
            "documents_file_checksum": actual_documents_file_checksum,
            "embeddings_checksum": actual_embeddings_checksum,
            "index_checksum": recomputed_index_checksum,
            "index_manifest_sha256": sha256_file(manifest_path),
            "actual_embedding_used": actual_embedding_used,
            "smoke_fallback_used": smoke_fallback_used,
        }

    @classmethod
    def validate_directory(
        cls,
        index_dir: str | Path,
        *,
        load_embeddings: bool = True,
        expected_backend: str | None = None,
        expected_model_name: str | None = None,
        expected_model_version: str | None = None,
        expected_dimension: int | None = None,
        expected_kg_checksum: str | None = None,
        expected_corpus_checksum: str | None = None,
        require_real_embedding: bool = False,
    ) -> dict[str, Any]:
        index_dir = Path(index_dir)
        if index_dir.is_file():
            if index_dir.suffix.lower() == ".json":
                raise VectorIndexError("legacy_ekell_json_forbidden_in_formal")
            raise VectorIndexError("ekell_index_path_not_directory")
        if not index_dir.is_dir():
            raise VectorIndexError("ekell_index_path_not_directory")
        if load_embeddings:
            loaded = cls.load_directory(
                index_dir,
                expected_backend=expected_backend,
                expected_model_name=expected_model_name,
                expected_model_version=expected_model_version,
                expected_dimension=expected_dimension,
                expected_kg_checksum=expected_kg_checksum,
                expected_corpus_checksum=expected_corpus_checksum,
                require_real_embedding=require_real_embedding,
            )
            return dict(loaded.metadata)
        docs_path = index_dir / "documents.jsonl"
        emb_path = index_dir / "embeddings.npy"
        manifest_path = index_dir / "index_manifest.json"
        if not manifest_path.is_file():
            raise VectorIndexError("ekell_index_manifest_missing")
        if not docs_path.is_file():
            raise VectorIndexError("ekell_index_documents_missing")
        if not emb_path.is_file():
            raise VectorIndexError("ekell_index_embeddings_missing")
        manifest = read_json(manifest_path)
        documents = read_jsonl(docs_path)
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise VectorIndexError("numpy is required to validate E-KELL embeddings (.npy).") from exc
        arr = np.load(emb_path, mmap_mode="r")
        row_count, dim = int(arr.shape[0]), int(arr.shape[1])
        if int(manifest.get("document_count") or 0) != len(documents):
            raise VectorIndexError("document_count does not match documents.jsonl length.")
        if row_count != len(documents):
            raise VectorIndexError("embeddings.npy row count does not match documents.")
        if int(manifest.get("dimension") or 0) != dim:
            raise VectorIndexError("Manifest dimension does not match embeddings.npy.")
        if require_real_embedding:
            require_ekell_formal_embedding_manifest(manifest)
        if expected_backend and str(manifest.get("backend")) != str(expected_backend):
            raise VectorIndexError("backend mismatch.")
        if expected_model_name and str(manifest.get("model_name")) != str(expected_model_name):
            raise VectorIndexError("model_name mismatch.")
        if expected_model_version and str(manifest.get("model_version")) != str(expected_model_version):
            raise VectorIndexError("model_version mismatch.")
        if expected_dimension is not None and int(manifest.get("dimension") or 0) != int(expected_dimension):
            raise VectorIndexError("dimension mismatch.")
        return dict(manifest)

    @classmethod
    def load_directory(
        cls,
        index_dir: str | Path,
        *,
        expected_backend: str | None = None,
        expected_model_name: str | None = None,
        expected_model_version: str | None = None,
        expected_dimension: int | None = None,
        expected_kg_checksum: str | None = None,
        expected_corpus_checksum: str | None = None,
        require_real_embedding: bool = False,
    ) -> "VectorIndex":
        index_dir = Path(index_dir)
        docs_path = index_dir / "documents.jsonl"
        emb_path = index_dir / "embeddings.npy"
        manifest_path = index_dir / "index_manifest.json"
        for path in (docs_path, emb_path, manifest_path):
            if not path.is_file():
                raise VectorIndexError(f"E-KELL index missing required file: {path}")

        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise VectorIndexError("index_manifest.json must be an object.")
        documents_raw = read_jsonl(docs_path)
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise VectorIndexError("numpy is required to load E-KELL embeddings (.npy).") from exc
        arr = np.load(emb_path)
        vectors = [[float(x) for x in row] for row in arr]
        documents = [VectorDocument(**doc) if isinstance(doc, dict) else doc for doc in documents_raw]

        if int(manifest.get("document_count") or 0) != len(documents):
            raise VectorIndexError("document_count does not match documents.jsonl length.")
        if len(vectors) != len(documents):
            raise VectorIndexError("embeddings.npy row count does not match documents.")
        if not vectors:
            raise VectorIndexError("E-KELL index has zero embeddings.")
        dim = len(vectors[0])
        if any(len(row) != dim for row in vectors):
            raise VectorIndexError("Inconsistent embedding row dimensions.")
        if int(manifest.get("dimension") or 0) != dim:
            raise VectorIndexError("Manifest dimension does not match embeddings.npy.")

        documents_checksum = ekell_documents_semantic_checksum(documents)
        if documents_checksum != manifest.get("documents_checksum"):
            raise VectorIndexError("documents_checksum mismatch.")
        if manifest.get("documents_file_checksum") and sha256_file(docs_path) != manifest.get(
            "documents_file_checksum"
        ):
            raise VectorIndexError("documents_file_checksum mismatch.")
        if sha256_file(emb_path) != manifest.get("embeddings_checksum"):
            raise VectorIndexError("embeddings_checksum mismatch.")
        normalize_embeddings = _require_exact_bool(dict(manifest), "normalize_embeddings")
        _validate_ekell_embedding_values(arr, normalize_embeddings=normalize_embeddings)

        recomputed = ekell_index_identity_checksum(
            backend=str(manifest.get("backend")),
            model_name=str(manifest.get("model_name")),
            model_version=str(manifest.get("model_version")),
            dimension=dim,
            normalize_embeddings=normalize_embeddings,
            document_count=len(documents),
            documents_checksum=documents_checksum,
            embeddings_checksum=str(manifest.get("embeddings_checksum")),
            kg_checksum=str(manifest.get("kg_checksum")),
            corpus_checksum=str(manifest.get("corpus_checksum")),
        )
        if manifest.get("index_checksum") and recomputed != manifest.get("index_checksum"):
            raise VectorIndexError("index_checksum mismatch.")

        if expected_backend and str(manifest.get("backend")) != str(expected_backend):
            raise VectorIndexError(
                f"backend mismatch: index={manifest.get('backend')!r} expected={expected_backend!r}"
            )
        if expected_model_name and str(manifest.get("model_name")) != str(expected_model_name):
            raise VectorIndexError(
                f"model_name mismatch: index={manifest.get('model_name')!r} expected={expected_model_name!r}"
            )
        if expected_model_version and str(manifest.get("model_version")) != str(expected_model_version):
            raise VectorIndexError(
                f"model_version mismatch: index={manifest.get('model_version')!r} "
                f"expected={expected_model_version!r}"
            )
        if expected_dimension is not None and int(manifest.get("dimension") or 0) != int(expected_dimension):
            raise VectorIndexError("dimension mismatch.")
        if expected_kg_checksum and str(manifest.get("kg_checksum")) != str(expected_kg_checksum):
            raise VectorIndexError("kg_checksum mismatch.")
        if expected_corpus_checksum and str(manifest.get("corpus_checksum")) != str(expected_corpus_checksum):
            raise VectorIndexError("corpus_checksum mismatch.")
        if require_real_embedding:
            require_ekell_formal_embedding_manifest(manifest)

        metadata = {
            **manifest,
            "embedding_model": manifest.get("model_name"),
            "index_checksum": manifest.get("index_checksum"),
        }
        return cls(documents, vectors, metadata)
