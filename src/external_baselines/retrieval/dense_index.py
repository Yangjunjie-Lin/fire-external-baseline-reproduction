"""Dense evidence index: documents.jsonl + embeddings.npy + index_manifest.json."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from external_baselines.common.checksums import sha256_file, sha256_json
from external_baselines.common.io import ensure_dir, read_json, read_jsonl, write_jsonl
from external_baselines.common.schema import RetrievedContext
from external_baselines.common.text_utils import compact_text
from external_baselines.retrieval.embedding_backends import (
    EmbeddingBackend,
    embedding_package_versions,
    validate_embedding_backend,
)


class DenseIndexError(RuntimeError):
    """Raised when a dense index cannot be built, loaded, or queried safely."""


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _model_slug(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in model_name).strip("-") or "model"


def default_dense_index_dir(
    *,
    model_name: str,
    corpus_checksum: str,
    index_root: str | Path = "outputs/indexes/dense",
) -> Path:
    return Path(index_root) / _model_slug(model_name) / corpus_checksum


def _load_documents(evidence_path: Path) -> list[dict[str, Any]]:
    docs_raw = read_jsonl(evidence_path)
    documents: list[dict[str, Any]] = []
    for i, doc in enumerate(docs_raw):
        text = str(doc.get("text") or doc.get("content") or doc.get("chunk") or doc.get("body") or "").strip()
        if not text:
            continue
        cid = str(doc.get("chunk_id") or doc.get("id") or doc.get("source_id") or f"chunk_{i}")
        documents.append(
            {
                "chunk_id": cid,
                "text": text,
                "source_id": doc.get("source_id") or doc.get("source") or doc.get("document_id"),
                "citation": doc.get("citation") or doc.get("url"),
                "metadata": {k: v for k, v in doc.items() if k not in {"text", "content", "chunk", "body"}},
            }
        )
    return documents


def _write_npy(path: Path, matrix: list[list[float]]) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise DenseIndexError("numpy is required to persist dense embeddings (.npy).") from exc
    arr = np.asarray(matrix, dtype="float32")
    ensure_dir(path.parent)
    np.save(path, arr)


def _read_npy(path: Path) -> list[list[float]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise DenseIndexError("numpy is required to load dense embeddings (.npy).") from exc
    arr = np.load(path)
    return [[float(x) for x in row] for row in arr]


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def build_dense_index(
    evidence_path: str | Path,
    embedding_backend: EmbeddingBackend,
    index_dir: str | Path,
    *,
    batch_size: int = 16,
    normalize_embeddings: bool = True,
    corpus_checksum: str | None = None,
    paper_final: bool = False,
    reject_smoke: bool = False,
) -> dict[str, Any]:
    """Build a dense evidence index directory. Returns the index manifest."""
    validate_embedding_backend(
        embedding_backend, paper_final=paper_final, reject_smoke=reject_smoke
    )
    evidence_path = Path(evidence_path)
    index_dir = Path(index_dir)
    documents = _load_documents(evidence_path)
    if not documents:
        raise DenseIndexError(f"No evidence documents found in {evidence_path}")

    texts = [str(d["text"]) for d in documents]
    embeddings: list[list[float]] = []
    bs = max(1, int(batch_size))
    for start in range(0, len(texts), bs):
        batch = texts[start : start + bs]
        vectors = embedding_backend.encode(batch)
        if len(vectors) != len(batch):
            raise DenseIndexError("Embedding batch size mismatch.")
        if normalize_embeddings:
            vectors = [_l2_normalize(v) for v in vectors]
        embeddings.extend(vectors)

    if len(embeddings) != len(documents):
        raise DenseIndexError("Document/embedding count mismatch after encoding.")
    dim = len(embeddings[0])
    if any(len(v) != dim for v in embeddings):
        raise DenseIndexError("Inconsistent embedding dimensions.")
    if embedding_backend.dimension not in (0, dim):
        raise DenseIndexError(
            f"Backend dimension {embedding_backend.dimension} != observed {dim}."
        )
    embedding_backend.dimension = dim

    evidence_checksum = sha256_file(evidence_path)
    corpus_checksum = corpus_checksum or evidence_checksum
    docs_checksum = sha256_json({"chunk_ids": [d["chunk_id"] for d in documents]})

    ensure_dir(index_dir)
    docs_path = index_dir / "documents.jsonl"
    emb_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "index_manifest.json"

    # Atomic-ish write: temp files then replace.
    write_jsonl(docs_path, documents)
    _write_npy(emb_path, embeddings)
    emb_checksum = _file_sha256(emb_path)
    docs_file_checksum = _file_sha256(docs_path)

    index_checksum = sha256_json(
        {
            "backend": embedding_backend.backend,
            "model_name": embedding_backend.model_name,
            "model_version": embedding_backend.model_version,
            "dimension": dim,
            "document_count": len(documents),
            "documents_checksum": docs_checksum,
            "embeddings_checksum": emb_checksum,
            "corpus_checksum": corpus_checksum,
            "evidence_source_checksum": evidence_checksum,
        }
    )
    manifest = {
        "index_type": "dense_evidence_index",
        "backend": embedding_backend.backend,
        "model_name": embedding_backend.model_name,
        "model_version": embedding_backend.model_version,
        "dimension": dim,
        "normalize_embeddings": bool(normalize_embeddings),
        "document_count": len(documents),
        "corpus_checksum": corpus_checksum,
        "documents_checksum": docs_checksum,
        "documents_file_checksum": docs_file_checksum,
        "embeddings_checksum": emb_checksum,
        "evidence_source_checksum": evidence_checksum,
        "index_checksum": index_checksum,
        "package_versions": embedding_package_versions(),
        "actual_embedding_used": bool(embedding_backend.actual_embedding_used),
        "smoke_fallback_used": bool(embedding_backend.smoke_fallback_used),
        "index_dir": str(index_dir).replace("\\", "/"),
        "evidence_path": str(evidence_path).replace("\\", "/"),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def load_dense_index(
    index_dir: str | Path,
    *,
    expected_model_name: str | None = None,
    expected_model_version: str | None = None,
    expected_corpus_checksum: str | None = None,
    expected_backend: str | None = None,
    expected_dimension: int | None = None,
) -> dict[str, Any]:
    """Load and validate a dense index directory. Hard-fails on mismatch."""
    index_dir = Path(index_dir)
    docs_path = index_dir / "documents.jsonl"
    emb_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "index_manifest.json"
    for path in (docs_path, emb_path, manifest_path):
        if not path.is_file():
            raise DenseIndexError(f"Dense index missing required file: {path}")

    manifest = read_json(manifest_path)
    documents = read_jsonl(docs_path)
    embeddings = _read_npy(emb_path)

    if int(manifest.get("document_count") or 0) != len(documents):
        raise DenseIndexError("document_count does not match documents.jsonl length.")
    if len(embeddings) != len(documents):
        raise DenseIndexError("embeddings.npy row count does not match documents.")
    if not embeddings:
        raise DenseIndexError("Dense index has zero embeddings.")
    dim = len(embeddings[0])
    if any(len(row) != dim for row in embeddings):
        raise DenseIndexError("Inconsistent embedding row dimensions.")
    if int(manifest.get("dimension") or 0) != dim:
        raise DenseIndexError("Manifest dimension does not match embeddings.npy.")

    docs_checksum = sha256_json({"chunk_ids": [d.get("chunk_id") for d in documents]})
    if docs_checksum != manifest.get("documents_checksum"):
        raise DenseIndexError("documents checksum mismatch.")
    if _file_sha256(emb_path) != manifest.get("embeddings_checksum"):
        raise DenseIndexError("embeddings checksum mismatch.")

    if expected_model_name and str(manifest.get("model_name")) != str(expected_model_name):
        raise DenseIndexError(
            f"model_name mismatch: index={manifest.get('model_name')!r} expected={expected_model_name!r}"
        )
    if expected_model_version and str(manifest.get("model_version")) != str(expected_model_version):
        raise DenseIndexError(
            f"model_version mismatch: index={manifest.get('model_version')!r} expected={expected_model_version!r}"
        )
    if expected_corpus_checksum and str(manifest.get("corpus_checksum")) != str(expected_corpus_checksum):
        raise DenseIndexError("corpus_checksum mismatch.")
    if expected_backend and str(manifest.get("backend")) != str(expected_backend):
        raise DenseIndexError(
            f"backend mismatch: index={manifest.get('backend')!r} expected={expected_backend!r}"
        )
    if expected_dimension is not None and int(manifest.get("dimension") or 0) != int(expected_dimension):
        raise DenseIndexError("dimension mismatch.")

    return {
        "index_dir": index_dir,
        "documents": documents,
        "embeddings": embeddings,
        "manifest": manifest,
        "dimension": dim,
        "checksum": manifest.get("index_checksum"),
        "backend": manifest.get("backend"),
        "model_name": manifest.get("model_name"),
        "model_version": manifest.get("model_version"),
    }


def query_dense_index(
    index: dict[str, Any],
    query: str,
    embedding_backend: EmbeddingBackend,
    *,
    top_k: int = 5,
    normalize_embeddings: bool = True,
) -> list[RetrievedContext]:
    """Embed query with the same backend and rank by cosine similarity."""
    if embedding_backend.backend != index.get("backend") and not (
        embedding_backend.smoke_fallback_used and str(index.get("backend", "")).startswith("smoke")
    ):
        # Allow smoke backends with smoke indexes; otherwise require identity.
        if not embedding_backend.smoke_fallback_used:
            if str(embedding_backend.model_name) != str(index.get("model_name")):
                raise DenseIndexError("Query embedding model_name does not match index.")
            if str(embedding_backend.model_version) != str(index.get("model_version")):
                raise DenseIndexError("Query embedding model_version does not match index.")

    q = embedding_backend.embed_query(query)
    if normalize_embeddings or bool((index.get("manifest") or {}).get("normalize_embeddings", True)):
        q = _l2_normalize(q)
    if len(q) != int(index["dimension"]):
        raise DenseIndexError("Query embedding dimension does not match index.")

    scored: list[tuple[float, int]] = []
    for i, emb in enumerate(index["embeddings"]):
        scored.append((_cosine(q, emb), i))
    scored.sort(key=lambda x: (-x[0], x[1]))

    contexts: list[RetrievedContext] = []
    manifest = index.get("manifest") or {}
    for rank, (score, idx) in enumerate(scored[:top_k], start=1):
        doc = index["documents"][idx]
        contexts.append(
            RetrievedContext(
                context_id=str(doc["chunk_id"]),
                text=compact_text(doc["text"], 1000),
                source_id=str(doc["source_id"]) if doc.get("source_id") is not None else None,
                citation=str(doc.get("citation") or doc.get("source_id") or doc["chunk_id"]),
                score=round(float(score), 6),
                metadata={
                    **(doc.get("metadata") or {}),
                    "retrieval_backend": "dense",
                    "embedding_backend": index.get("backend"),
                    "embedding_model": index.get("model_name"),
                    "embedding_model_version": index.get("model_version"),
                    "dense_rank": rank,
                    "dense_score": round(float(score), 6),
                    "index_checksum": index.get("checksum"),
                    "corpus_checksum": manifest.get("corpus_checksum"),
                },
            )
        )
    return contexts


def is_directory_index(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() or (path.suffix == "" and not path.exists()) or (
        path.exists() and path.is_dir()
    )
