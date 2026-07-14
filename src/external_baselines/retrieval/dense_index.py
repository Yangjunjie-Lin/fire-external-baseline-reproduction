"""Dense evidence index: documents.jsonl + embeddings.npy + index_manifest.json."""

from __future__ import annotations

import json
import os
import re
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
    l2_normalize_vector,
    require_exact_embedding_evidence_flags,
    validate_embedding_backend,
)


class DenseIndexError(RuntimeError):
    """Raised when a dense index cannot be built, loaded, or queried safely."""


SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


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


def dense_documents_semantic_checksum(documents: list[dict[str, Any]]) -> str:
    """Canonical semantic checksum shared by Dense index build and validation."""
    return sha256_json(
        [
            {
                "chunk_id": d.get("chunk_id"),
                "text": d.get("text"),
                "source_id": d.get("source_id"),
                "citation": d.get("citation"),
            }
            for d in documents
        ]
    )


def dense_index_identity_checksum(
    *,
    backend: str,
    model_name: str,
    model_version: str,
    dimension: int,
    normalize_embeddings: bool,
    document_count: int,
    documents_checksum: str,
    embeddings_checksum: str,
    corpus_checksum: str,
    evidence_source_checksum: str,
) -> str:
    """Canonical Dense persisted-index checksum payload."""
    if type(normalize_embeddings) is not bool:
        raise TypeError("dense_index_normalize_embeddings_must_be_bool")
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
            "corpus_checksum": corpus_checksum,
            "evidence_source_checksum": evidence_source_checksum,
        }
    )


def _require_plain_file(path: Path, *, code: str) -> None:
    if not path.is_file():
        raise DenseIndexError(code)


def _require_manifest_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DenseIndexError("dense_index_manifest_must_be_object")
    return value


def _require_exact_nonempty_string(manifest: dict[str, Any], field: str) -> str:
    value = manifest.get(field)
    if type(value) is not str or not value:
        raise DenseIndexError(f"dense_index_{field}_must_be_nonempty_string")
    return value


def _require_positive_int(manifest: dict[str, Any], field: str) -> int:
    value = manifest.get(field)
    if type(value) is not int or value <= 0:
        raise DenseIndexError(f"dense_index_{field}_must_be_positive_int")
    return value


def _require_exact_bool(manifest: dict[str, Any], field: str) -> bool:
    value = manifest.get(field)
    if type(value) is not bool:
        raise DenseIndexError(f"dense_index_{field}_must_be_bool")
    return value


def _require_sha256(manifest: dict[str, Any], field: str) -> str:
    value = manifest.get(field)
    if type(value) is not str or not SHA256_HEX_RE.fullmatch(value):
        raise DenseIndexError(f"dense_index_{field}_invalid")
    return value


def _read_dense_documents_strict(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DenseIndexError(f"dense_index_documents_invalid_json:line_{line_no}") from exc
            if not isinstance(value, dict):
                raise DenseIndexError(f"dense_index_documents_record_must_be_object:line_{line_no}")
            documents.append(value)
    if not documents:
        raise DenseIndexError("dense_index_documents_empty")
    return documents


def _validate_dense_embedding_values(
    arr: Any,
    *,
    normalize_embeddings: bool,
    chunk_rows: int = 4096,
) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise DenseIndexError("numpy is required to validate dense embeddings (.npy).") from exc

    row_count = int(arr.shape[0])
    for start in range(0, row_count, chunk_rows):
        block = np.asarray(arr[start : start + chunk_rows])
        if not np.isfinite(block).all():
            raise DenseIndexError("dense_index_embeddings_non_finite")
        norms = np.linalg.norm(block, axis=1)
        if (norms <= 1e-12).any():
            raise DenseIndexError("dense_index_zero_embedding_vector")
        if normalize_embeddings and not np.allclose(
            norms,
            1.0,
            rtol=1e-4,
            atol=1e-5,
        ):
            raise DenseIndexError("dense_index_normalized_embeddings_unit_norm_mismatch")


def require_dense_formal_embedding_manifest(manifest: dict[str, Any]) -> None:
    """Strict formal embedding evidence checks (no defaults or loose bool coercion)."""
    if "actual_embedding_used" not in manifest:
        raise DenseIndexError("actual_embedding_used_missing")
    if manifest["actual_embedding_used"] is not True:
        raise DenseIndexError("actual_embedding_used_must_be_true")
    if "smoke_fallback_used" not in manifest:
        raise DenseIndexError("smoke_fallback_used_missing")
    if manifest["smoke_fallback_used"] is not False:
        raise DenseIndexError("smoke_fallback_used_must_be_false")


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
    if type(normalize_embeddings) is not bool:
        raise DenseIndexError("dense_index_normalize_embeddings_must_be_bool")
    validate_embedding_backend(
        embedding_backend, paper_final=paper_final, reject_smoke=reject_smoke
    )
    actual_embedding_used, smoke_fallback_used = require_exact_embedding_evidence_flags(
        embedding_backend
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
            vectors = [l2_normalize_vector(v) for v in vectors]
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
    docs_checksum = dense_documents_semantic_checksum(documents)

    ensure_dir(index_dir)
    docs_path = index_dir / "documents.jsonl"
    emb_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "index_manifest.json"

    # Atomic-ish write: temp files then replace.
    write_jsonl(docs_path, documents)
    _write_npy(emb_path, embeddings)
    emb_checksum = _file_sha256(emb_path)
    docs_file_checksum = _file_sha256(docs_path)

    index_checksum = dense_index_identity_checksum(
        backend=embedding_backend.backend,
        model_name=embedding_backend.model_name,
        model_version=embedding_backend.model_version,
        dimension=dim,
        normalize_embeddings=normalize_embeddings,
        document_count=len(documents),
        documents_checksum=docs_checksum,
        embeddings_checksum=emb_checksum,
        corpus_checksum=corpus_checksum,
        evidence_source_checksum=evidence_checksum,
    )
    manifest = {
        "index_type": "dense_evidence_index",
        "backend": embedding_backend.backend,
        "model_name": embedding_backend.model_name,
        "model_version": embedding_backend.model_version,
        "dimension": dim,
        "normalize_embeddings": normalize_embeddings,
        "document_count": len(documents),
        "corpus_checksum": corpus_checksum,
        "documents_checksum": docs_checksum,
        "documents_file_checksum": docs_file_checksum,
        "embeddings_checksum": emb_checksum,
        "evidence_source_checksum": evidence_checksum,
        "index_checksum": index_checksum,
        "package_versions": embedding_package_versions(),
        "actual_embedding_used": actual_embedding_used,
        "smoke_fallback_used": smoke_fallback_used,
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

    docs_checksum = dense_documents_semantic_checksum(documents)
    if docs_checksum != manifest.get("documents_checksum"):
        raise DenseIndexError("documents_checksum mismatch (semantic document content).")
    if manifest.get("documents_file_checksum") and _file_sha256(docs_path) != manifest.get(
        "documents_file_checksum"
    ):
        raise DenseIndexError("documents_file_checksum mismatch.")
    if _file_sha256(emb_path) != manifest.get("embeddings_checksum"):
        raise DenseIndexError("embeddings_checksum mismatch.")
    normalize_embeddings = _require_exact_bool(dict(manifest), "normalize_embeddings")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise DenseIndexError("numpy is required to validate dense embeddings (.npy).") from exc
    _validate_dense_embedding_values(
        np.load(emb_path, mmap_mode="r"),
        normalize_embeddings=normalize_embeddings,
    )
    expected_index = dense_index_identity_checksum(
        backend=str(manifest.get("backend")),
        model_name=str(manifest.get("model_name")),
        model_version=str(manifest.get("model_version")),
        dimension=dim,
        normalize_embeddings=normalize_embeddings,
        document_count=len(documents),
        documents_checksum=docs_checksum,
        embeddings_checksum=str(manifest.get("embeddings_checksum")),
        corpus_checksum=str(manifest.get("corpus_checksum")),
        evidence_source_checksum=str(manifest.get("evidence_source_checksum")),
    )
    if manifest.get("index_checksum") and expected_index != manifest.get("index_checksum"):
        raise DenseIndexError("index_checksum mismatch.")
    if bool(manifest.get("smoke_fallback_used")) and (
        expected_backend and str(expected_backend).lower() not in {"smoke", "hash", "smoke_hash_embedding"}
    ):
        raise DenseIndexError("smoke_fallback_used=true is forbidden for real dense index loads.")
    if expected_backend and str(expected_backend).lower() not in {
        "smoke",
        "hash",
        "smoke_hash_embedding",
        "deterministic_hash_smoke",
    }:
        require_dense_formal_embedding_manifest(manifest)

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


def validate_dense_index_integrity_for_freeze(
    index_dir: str | Path,
    *,
    expected_backend: str | None = None,
    expected_model_name: str | None = None,
    expected_model_version: str | None = None,
    expected_dimension: int | None = None,
    expected_corpus_checksum: str | None = None,
    expected_evidence_source_checksum: str | None = None,
    expected_normalize_embeddings: bool | None = None,
) -> dict[str, Any]:
    """Strict persisted Dense index validation for freeze-candidate/complete freeze."""
    index_dir = Path(index_dir)
    if index_dir.is_file():
        if index_dir.suffix.lower() == ".json":
            raise DenseIndexError("legacy_dense_json_forbidden_in_formal")
        raise DenseIndexError("dense_index_path_not_directory")
    if not index_dir.is_dir():
        raise DenseIndexError("dense_index_path_not_directory")

    docs_path = index_dir / "documents.jsonl"
    emb_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "index_manifest.json"
    _require_plain_file(docs_path, code="dense_index_documents_missing")
    _require_plain_file(emb_path, code="dense_index_embeddings_missing")
    _require_plain_file(manifest_path, code="dense_index_manifest_missing")

    manifest = _require_manifest_object(read_json(manifest_path))
    index_type = _require_exact_nonempty_string(manifest, "index_type")
    if index_type != "dense_evidence_index":
        raise DenseIndexError("dense_index_type_mismatch")
    backend = _require_exact_nonempty_string(manifest, "backend")
    model_name = _require_exact_nonempty_string(manifest, "model_name")
    model_version = _require_exact_nonempty_string(manifest, "model_version")
    dimension = _require_positive_int(manifest, "dimension")
    document_count = _require_positive_int(manifest, "document_count")
    normalize_embeddings = _require_exact_bool(manifest, "normalize_embeddings")
    corpus_checksum = _require_sha256(manifest, "corpus_checksum")
    documents_checksum = _require_sha256(manifest, "documents_checksum")
    documents_file_checksum = _require_sha256(manifest, "documents_file_checksum")
    embeddings_checksum = _require_sha256(manifest, "embeddings_checksum")
    evidence_source_checksum = _require_sha256(manifest, "evidence_source_checksum")
    index_checksum = _require_sha256(manifest, "index_checksum")
    actual_embedding_used = _require_exact_bool(manifest, "actual_embedding_used")
    smoke_fallback_used = _require_exact_bool(manifest, "smoke_fallback_used")
    if actual_embedding_used is not True:
        raise DenseIndexError("actual_embedding_used_must_be_true")
    if smoke_fallback_used is not False:
        raise DenseIndexError("smoke_fallback_used_must_be_false")

    if expected_backend and backend != str(expected_backend):
        raise DenseIndexError("backend mismatch.")
    if expected_model_name and model_name != str(expected_model_name):
        raise DenseIndexError("model_name mismatch.")
    if expected_model_version and model_version != str(expected_model_version):
        raise DenseIndexError("model_version mismatch.")
    if expected_dimension is not None and dimension != int(expected_dimension):
        raise DenseIndexError("dimension mismatch.")
    if expected_corpus_checksum is not None:
        if (
            type(expected_corpus_checksum) is not str
            or not SHA256_HEX_RE.fullmatch(expected_corpus_checksum)
        ):
            raise DenseIndexError("dense_index_expected_corpus_checksum_invalid")
        if corpus_checksum != expected_corpus_checksum:
            raise DenseIndexError("index_corpus_checksum_contract_requires_rebuild")
    if expected_evidence_source_checksum is not None:
        if (
            type(expected_evidence_source_checksum) is not str
            or not SHA256_HEX_RE.fullmatch(expected_evidence_source_checksum)
        ):
            raise DenseIndexError("dense_index_expected_evidence_source_checksum_invalid")
        if evidence_source_checksum != expected_evidence_source_checksum:
            raise DenseIndexError("dense_index_evidence_source_checksum_mismatch")
    if expected_normalize_embeddings is not None:
        if type(expected_normalize_embeddings) is not bool:
            raise DenseIndexError("dense_index_expected_normalize_embeddings_must_be_bool")
        if normalize_embeddings is not expected_normalize_embeddings:
            raise DenseIndexError("dense_index_normalize_embeddings_mismatch")

    documents = _read_dense_documents_strict(docs_path)
    if len(documents) != document_count:
        raise DenseIndexError("document_count does not match documents.jsonl length.")
    actual_documents_file_checksum = _file_sha256(docs_path)
    if actual_documents_file_checksum != documents_file_checksum:
        raise DenseIndexError("documents_file_checksum mismatch.")
    actual_embeddings_checksum = _file_sha256(emb_path)
    if actual_embeddings_checksum != embeddings_checksum:
        raise DenseIndexError("embeddings_checksum mismatch.")
    recomputed_documents_checksum = dense_documents_semantic_checksum(documents)
    if recomputed_documents_checksum != documents_checksum:
        raise DenseIndexError("documents_checksum mismatch (semantic document content).")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise DenseIndexError("numpy is required to validate dense embeddings (.npy).") from exc
    arr = np.load(emb_path, mmap_mode="r")
    if len(arr.shape) != 2:
        raise DenseIndexError("embeddings.npy must be a two-dimensional array.")
    row_count, dim = int(arr.shape[0]), int(arr.shape[1])
    if row_count <= 0 or dim <= 0:
        raise DenseIndexError("embeddings.npy must have positive row count and dimension.")
    if row_count != document_count:
        raise DenseIndexError("embeddings.npy row count does not match documents.")
    if dim != dimension:
        raise DenseIndexError("Manifest dimension does not match embeddings.npy.")
    _validate_dense_embedding_values(arr, normalize_embeddings=normalize_embeddings)

    recomputed_index_checksum = dense_index_identity_checksum(
        backend=backend,
        model_name=model_name,
        model_version=model_version,
        dimension=dimension,
        normalize_embeddings=normalize_embeddings,
        document_count=document_count,
        documents_checksum=recomputed_documents_checksum,
        embeddings_checksum=actual_embeddings_checksum,
        corpus_checksum=corpus_checksum,
        evidence_source_checksum=evidence_source_checksum,
    )
    if recomputed_index_checksum != index_checksum:
        raise DenseIndexError("index_checksum mismatch.")

    manifest_sha = _file_sha256(manifest_path)
    return {
        "index_type": index_type,
        "index_dir": str(index_dir).replace("\\", "/"),
        "backend": backend,
        "model_name": model_name,
        "model_version": model_version,
        "dimension": dimension,
        "normalize_embeddings": normalize_embeddings,
        "document_count": document_count,
        "corpus_checksum": corpus_checksum,
        "documents_checksum": recomputed_documents_checksum,
        "documents_file_checksum": actual_documents_file_checksum,
        "embeddings_checksum": actual_embeddings_checksum,
        "evidence_source_checksum": evidence_source_checksum,
        "index_checksum": recomputed_index_checksum,
        "index_manifest_sha256": manifest_sha,
        "actual_embedding_used": actual_embedding_used,
        "smoke_fallback_used": smoke_fallback_used,
    }


def validate_dense_index_directory(
    index_dir: str | Path,
    *,
    load_embeddings: bool = True,
    expected_model_name: str | None = None,
    expected_model_version: str | None = None,
    expected_corpus_checksum: str | None = None,
    expected_backend: str | None = None,
    expected_dimension: int | None = None,
    require_explicit_embedding_evidence: bool = False,
) -> dict[str, Any]:
    """Validate a persisted dense index directory without building or migrating."""
    index_dir = Path(index_dir)
    if index_dir.is_file():
        if index_dir.suffix.lower() == ".json":
            raise DenseIndexError("legacy_dense_json_forbidden_in_formal")
        raise DenseIndexError("dense_index_path_not_directory")
    if not index_dir.is_dir():
        raise DenseIndexError("dense_index_path_not_directory")

    docs_path = index_dir / "documents.jsonl"
    emb_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.is_file():
        raise DenseIndexError("dense_index_manifest_missing")
    if not docs_path.is_file():
        raise DenseIndexError("dense_index_documents_missing")
    if not emb_path.is_file():
        raise DenseIndexError("dense_index_embeddings_missing")

    if load_embeddings:
        payload = load_dense_index(
            index_dir,
            expected_model_name=expected_model_name,
            expected_model_version=expected_model_version,
            expected_corpus_checksum=expected_corpus_checksum,
            expected_backend=expected_backend,
            expected_dimension=expected_dimension,
        )
        if require_explicit_embedding_evidence:
            require_dense_formal_embedding_manifest(dict(payload.get("manifest") or {}))
        return payload

    manifest = read_json(manifest_path)
    documents = read_jsonl(docs_path)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise DenseIndexError("numpy is required to validate dense embeddings (.npy).") from exc
    arr = np.load(emb_path, mmap_mode="r")
    row_count, dim = int(arr.shape[0]), int(arr.shape[1])
    if int(manifest.get("document_count") or 0) != len(documents):
        raise DenseIndexError("document_count does not match documents.jsonl length.")
    if row_count != len(documents):
        raise DenseIndexError("embeddings.npy row count does not match documents.")
    if int(manifest.get("dimension") or 0) != dim:
        raise DenseIndexError("Manifest dimension does not match embeddings.npy.")
    if require_explicit_embedding_evidence:
        require_dense_formal_embedding_manifest(manifest)
    elif bool(manifest.get("smoke_fallback_used")):
        raise DenseIndexError("smoke_fallback_used=true is forbidden for formal dense indexes.")
    elif manifest.get("actual_embedding_used") is not True:
        raise DenseIndexError("actual_embedding_used must be true for formal dense indexes.")
    if expected_model_name and str(manifest.get("model_name")) != str(expected_model_name):
        raise DenseIndexError("model_name mismatch.")
    if expected_model_version and str(manifest.get("model_version")) != str(expected_model_version):
        raise DenseIndexError("model_version mismatch.")
    if expected_backend and str(manifest.get("backend")) != str(expected_backend):
        raise DenseIndexError("backend mismatch.")
    if expected_dimension is not None and int(manifest.get("dimension") or 0) != int(expected_dimension):
        raise DenseIndexError("dimension mismatch.")
    if expected_corpus_checksum and str(manifest.get("corpus_checksum")) != str(expected_corpus_checksum):
        raise DenseIndexError("corpus_checksum mismatch.")
    return {
        "index_dir": index_dir,
        "documents": documents,
        "embeddings": None,
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
    manifest_normalize = (index.get("manifest") or {}).get("normalize_embeddings")
    if type(manifest_normalize) is bool:
        normalize_query = manifest_normalize
    elif type(normalize_embeddings) is bool:
        normalize_query = normalize_embeddings
    else:
        raise DenseIndexError("dense_index_normalize_embeddings_must_be_bool")
    if normalize_query:
        q = l2_normalize_vector(q)
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
