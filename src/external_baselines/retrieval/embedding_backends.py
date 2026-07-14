"""Shared embedding backends for Dense, Hybrid, and E-KELL vector retrieval."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding backend cannot satisfy the requested run."""


def require_exact_embedding_evidence_flags(
    backend: "EmbeddingBackend",
) -> tuple[bool, bool]:
    """Return provenance flags only when the backend supplies exact booleans."""
    actual_embedding_used = getattr(backend, "actual_embedding_used", None)
    smoke_fallback_used = getattr(backend, "smoke_fallback_used", None)
    if type(actual_embedding_used) is not bool:
        raise EmbeddingBackendError("embedding_backend_actual_embedding_used_must_be_bool")
    if type(smoke_fallback_used) is not bool:
        raise EmbeddingBackendError("embedding_backend_smoke_fallback_used_must_be_bool")
    return actual_embedding_used, smoke_fallback_used


def _as_vectors(values: Any) -> list[list[float]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise EmbeddingBackendError("Embedding backend returned a non-sequence value.")
    if values and isinstance(values[0], (int, float)):
        values = [values]
    vectors = [[float(value) for value in vector] for vector in values]
    if any(not vector for vector in vectors):
        raise EmbeddingBackendError("Embedding backend returned an empty vector.")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise EmbeddingBackendError("embedding_backend_returned_non_finite_value")
    return vectors


def l2_normalize_vector(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalized copy, preserving zero vectors for validators to reject."""
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise EmbeddingBackendError("embedding_backend_returned_non_finite_value")
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


class EmbeddingBackend(ABC):
    """Minimal text2vec-compatible embedding contract with auditable metadata."""

    backend: str
    model_name: str
    model_version: str
    dimension: int
    actual_embedding_used: bool
    smoke_fallback_used: bool

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts in input order."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.encode([text])[0]

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "embedding_model": self.model_name,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "actual_embedding_used": self.actual_embedding_used,
            "smoke_fallback_used": self.smoke_fallback_used,
        }


@dataclass
class Text2VecEmbeddingBackend(EmbeddingBackend):
    """Lazy text2vec wrapper. Inject ``model`` in tests to avoid downloads."""

    model_name: str
    model_version: str = "unspecified"
    model: Any | None = None
    dimension: int = 0
    backend: str = "text2vec"
    actual_embedding_used: bool = True
    smoke_fallback_used: bool = False

    _load_count: int = 0

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be explicitly configured.")
        # Injected models (tests) are already loaded; real SentenceModel is lazy.
        if self.model is not None:
            self._load_count = 1

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        try:
            from text2vec import SentenceModel  # type: ignore
        except ImportError as exc:
            raise EmbeddingBackendError(
                "text2vec is required for the real embedding backend."
            ) from exc
        self.model = SentenceModel(self.model_name)
        self._load_count += 1
        return self.model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        batch = [str(text) for text in texts]
        if not batch:
            return []
        model = self._ensure_model()
        encoder = getattr(model, "encode", None)
        if not callable(encoder):
            raise EmbeddingBackendError("Configured text2vec model has no encode method.")
        try:
            values = encoder(batch, show_progress_bar=False)
        except TypeError:
            values = encoder(batch)
        vectors = _as_vectors(values)
        if len(vectors) != len(batch):
            raise EmbeddingBackendError("Embedding count does not match input count.")
        observed = len(vectors[0])
        if any(len(vector) != observed for vector in vectors):
            raise EmbeddingBackendError("Embedding dimensions are inconsistent.")
        if self.dimension not in (0, observed):
            raise EmbeddingBackendError(
                f"Configured dimension {self.dimension} differs from observed {observed}."
            )
        self.dimension = observed
        return vectors


@dataclass
class HashEmbeddingBackend(EmbeddingBackend):
    """Deterministic feature-hash embedding reserved for smoke tests."""

    dimension: int = 128
    model_name: str = "deterministic-hash-smoke"
    model_version: str = "sha256-v1"
    backend: str = "deterministic_hash_smoke"
    actual_embedding_used: bool = False
    smoke_fallback_used: bool = True

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(str(text)) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


SmokeHashEmbeddingBackend = HashEmbeddingBackend

SMOKE_BACKEND_NAMES = frozenset(
    {
        "smoke",
        "hash",
        "hash_smoke",
        "deterministic_hash",
        "deterministic_hash_smoke",
        "smoke_hash",
        "smoke_hash_embedding",
    }
)


def create_embedding_backend(
    backend: str,
    *,
    model_name: str | None = None,
    model_version: str = "unspecified",
    dimension: int = 128,
    paper_final: bool = False,
    reject_smoke: bool = False,
    model: Any | None = None,
) -> EmbeddingBackend:
    """Create a controlled backend. Does not download models until encode/init without inject."""

    normalized = backend.strip().casefold().replace("-", "_")
    if normalized in SMOKE_BACKEND_NAMES:
        if paper_final or reject_smoke:
            raise EmbeddingBackendError(
                "Smoke hash embeddings are forbidden when paper_final=true or reject_smoke=true."
            )
        return HashEmbeddingBackend(
            dimension=dimension,
            model_name=model_name or "deterministic-hash-smoke",
            model_version=model_version if model_version != "unspecified" else "sha256-v1",
            backend="smoke_hash_embedding" if "smoke_hash" in normalized else "deterministic_hash_smoke",
        )
    if normalized in {"text2vec", "sentence_model", "text2vec_compatible"}:
        return Text2VecEmbeddingBackend(
            model_name=model_name or "",
            model_version=model_version,
            model=model,
            dimension=dimension if dimension != 128 else 0,
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")


def embedding_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("text2vec", "sentence-transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def validate_embedding_backend(
    backend: EmbeddingBackend, *, paper_final: bool = False, reject_smoke: bool = False
) -> None:
    actual_embedding_used, smoke_fallback_used = require_exact_embedding_evidence_flags(backend)
    if (paper_final or reject_smoke) and (
        smoke_fallback_used or not actual_embedding_used
    ):
        raise EmbeddingBackendError(
            "A real embedding backend is required when paper_final=true or reject_smoke=true."
        )


def resolve_dimension(cfg: dict[str, Any], default: int = 64) -> int:
    return int(cfg.get("dimension", cfg.get("dim", default)))


def _normalize_backend_name(value: str) -> str:
    return str(value or "").strip().casefold().replace("-", "_")


def embedding_backend_identity(backend: Any) -> dict[str, Any]:
    return {
        "backend": str(getattr(backend, "backend", "") or ""),
        "model_name": str(getattr(backend, "model_name", "") or ""),
        "model_version": str(getattr(backend, "model_version", "") or ""),
        "dimension": int(getattr(backend, "dimension", 0) or 0),
        "actual_embedding_used": getattr(
            backend,
            "actual_embedding_used",
            None,
        ),
        "smoke_fallback_used": getattr(
            backend,
            "smoke_fallback_used",
            None,
        ),
    }


def _manifest_embedding_fields(index_manifest: dict[str, Any]) -> dict[str, Any]:
    model_name_raw = index_manifest.get("model_name")
    if model_name_raw is None:
        model_name_raw = index_manifest.get("embedding_model")
    return {
        "backend_raw": index_manifest.get("backend"),
        "model_name_raw": model_name_raw,
        "model_version_raw": index_manifest.get("model_version"),
        "dimension_raw": index_manifest.get("dimension"),
        "actual_embedding_used_raw": index_manifest.get("actual_embedding_used"),
        "smoke_fallback_used_raw": index_manifest.get("smoke_fallback_used"),
        "normalize_embeddings_raw": index_manifest.get("normalize_embeddings"),
        "backend": "",
        "model_name": "",
        "model_version": "",
        "dimension": 0,
        "actual_embedding_used": None,
        "smoke_fallback_used": None,
        "normalize_embeddings": None,
    }


def _validate_manifest_field_types(index_manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, bool]]:
    fields = _manifest_embedding_fields(index_manifest)
    errors: list[str] = []
    checks: dict[str, bool] = {
        "manifest_backend_present": False,
        "manifest_backend_type_valid": False,
        "manifest_model_name_present": False,
        "manifest_model_name_type_valid": False,
        "manifest_model_version_present": False,
        "manifest_model_version_type_valid": False,
        "manifest_dimension_type_valid": False,
        "manifest_dimension_valid": False,
        "manifest_actual_embedding_used_type_valid": False,
        "manifest_actual_embedding_used_true": False,
        "manifest_smoke_fallback_used_type_valid": False,
        "manifest_smoke_fallback_used_false": False,
        "manifest_normalize_embeddings_type_valid": False,
    }

    backend_raw = fields["backend_raw"]
    if backend_raw is None:
        errors.append("runtime_embedding_manifest_missing: backend")
    elif not isinstance(backend_raw, str):
        errors.append("runtime_embedding_manifest_invalid_type: backend")
    elif not backend_raw.strip():
        errors.append("runtime_embedding_manifest_missing: backend")
    else:
        checks["manifest_backend_present"] = True
        checks["manifest_backend_type_valid"] = True
        fields["backend"] = backend_raw.strip()

    model_name_raw = fields["model_name_raw"]
    if model_name_raw is None:
        errors.append("runtime_embedding_manifest_missing: model_name")
    elif not isinstance(model_name_raw, str):
        errors.append("runtime_embedding_manifest_invalid_type: model_name")
    elif not model_name_raw.strip():
        errors.append("runtime_embedding_manifest_missing: model_name")
    else:
        checks["manifest_model_name_present"] = True
        checks["manifest_model_name_type_valid"] = True
        fields["model_name"] = model_name_raw.strip()

    model_version_raw = fields["model_version_raw"]
    if model_version_raw is None:
        errors.append("runtime_embedding_manifest_missing: model_version")
    elif not isinstance(model_version_raw, str):
        errors.append("runtime_embedding_manifest_invalid_type: model_version")
    elif not model_version_raw.strip():
        errors.append("runtime_embedding_manifest_missing: model_version")
    else:
        checks["manifest_model_version_present"] = True
        checks["manifest_model_version_type_valid"] = True
        fields["model_version"] = model_version_raw.strip()

    dimension_raw = fields["dimension_raw"]
    if dimension_raw is None:
        errors.append("runtime_embedding_manifest_missing: dimension")
    elif type(dimension_raw) is not int:
        errors.append("runtime_embedding_manifest_invalid_type: dimension")
    elif dimension_raw <= 0:
        errors.append("runtime_embedding_manifest_invalid: dimension must be a positive integer")
    else:
        checks["manifest_dimension_type_valid"] = True
        checks["manifest_dimension_valid"] = True
        fields["dimension"] = dimension_raw

    actual_raw = fields["actual_embedding_used_raw"]
    if actual_raw is None:
        errors.append("runtime_embedding_manifest_missing: actual_embedding_used")
    elif type(actual_raw) is not bool:
        errors.append("runtime_embedding_manifest_invalid_type: actual_embedding_used")
    elif actual_raw is not True:
        errors.append("runtime_embedding_manifest_invalid: actual_embedding_used must be true")
    else:
        checks["manifest_actual_embedding_used_type_valid"] = True
        checks["manifest_actual_embedding_used_true"] = True
        fields["actual_embedding_used"] = actual_raw

    smoke_raw = fields["smoke_fallback_used_raw"]
    if smoke_raw is None:
        errors.append("runtime_embedding_manifest_missing: smoke_fallback_used")
    elif type(smoke_raw) is not bool:
        errors.append("runtime_embedding_manifest_invalid_type: smoke_fallback_used")
    elif smoke_raw is not False:
        errors.append("runtime_embedding_manifest_invalid: smoke_fallback_used must be false")
    else:
        checks["manifest_smoke_fallback_used_type_valid"] = True
        checks["manifest_smoke_fallback_used_false"] = True
        fields["smoke_fallback_used"] = smoke_raw

    normalize_raw = fields["normalize_embeddings_raw"]
    if normalize_raw is None:
        errors.append("runtime_embedding_manifest_missing: normalize_embeddings")
    elif type(normalize_raw) is not bool:
        errors.append("runtime_embedding_manifest_invalid_type: normalize_embeddings")
    else:
        checks["manifest_normalize_embeddings_type_valid"] = True
        fields["normalize_embeddings"] = normalize_raw

    return fields, errors, checks


def validate_runtime_embedding_identity(
    *,
    actual_backend: Any,
    configured_backend: str,
    configured_model_name: str,
    configured_model_version: str,
    configured_dimension: int,
    configured_normalize_embeddings: bool | None = None,
    index_manifest: dict[str, Any],
    formal: bool,
) -> dict[str, Any]:
    actual = embedding_backend_identity(actual_backend)
    configured = {
        "backend": str(configured_backend or ""),
        "model_name": str(configured_model_name or ""),
        "model_version": str(configured_model_version or ""),
        "dimension": int(configured_dimension or 0),
        "normalize_embeddings": configured_normalize_embeddings,
    }
    manifest, manifest_errors, manifest_checks = _validate_manifest_field_types(index_manifest)
    checks: dict[str, bool] = dict(manifest_checks)
    checks.update({
        "backend_match": _normalize_backend_name(actual["backend"])
        == _normalize_backend_name(configured["backend"])
        == _normalize_backend_name(manifest["backend"]),
        "model_name_match": actual["model_name"].strip() == configured["model_name"].strip()
        == manifest["model_name"].strip(),
        "model_version_match": actual["model_version"].strip()
        == configured["model_version"].strip()
        == manifest["model_version"].strip(),
        "dimension_match": actual["dimension"] == configured["dimension"] == manifest["dimension"],
        "normalize_embeddings_match": (
            configured_normalize_embeddings is None
            or (
                checks.get("manifest_normalize_embeddings_type_valid") is True
                and type(configured_normalize_embeddings) is bool
                and configured_normalize_embeddings is manifest["normalize_embeddings"]
            )
        ),
        "actual_embedding_used": actual["actual_embedding_used"] is True,
        "smoke_fallback_forbidden": actual["smoke_fallback_used"] is False,
    })
    errors: list[str] = list(manifest_errors)
    if formal:
        if checks["manifest_backend_type_valid"] and checks["manifest_backend_present"] and not checks["backend_match"]:
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"backend configured={configured['backend']!r} actual={actual['backend']!r} "
                f"index={manifest['backend']!r}"
            )
        if checks["manifest_model_name_type_valid"] and actual["model_name"].strip() != configured["model_name"].strip():
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"model_name configured={configured['model_name']!r} actual={actual['model_name']!r}"
            )
        if (
            checks["manifest_model_name_type_valid"]
            and checks["manifest_model_name_present"]
            and actual["model_name"].strip() != manifest["model_name"].strip()
        ):
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"model_name actual={actual['model_name']!r} index={manifest['model_name']!r}"
            )
        if (
            checks["manifest_model_version_type_valid"]
            and actual["model_version"].strip() != configured["model_version"].strip()
        ):
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"model_version configured={configured['model_version']!r} "
                f"actual={actual['model_version']!r}"
            )
        if (
            checks["manifest_model_version_type_valid"]
            and checks["manifest_model_version_present"]
            and actual["model_version"].strip() != manifest["model_version"].strip()
        ):
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"model_version actual={actual['model_version']!r} index={manifest['model_version']!r}"
            )
        if actual["dimension"] != configured["dimension"]:
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"dimension configured={configured['dimension']} actual={actual['dimension']}"
            )
        if (
            checks["manifest_dimension_type_valid"]
            and checks["manifest_dimension_valid"]
            and actual["dimension"] != manifest["dimension"]
        ):
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"dimension actual={actual['dimension']} index={manifest['dimension']}"
            )
        if actual["actual_embedding_used"] is not True:
            errors.append("runtime_embedding_identity_mismatch: actual_embedding_used must be true")
        if actual["smoke_fallback_used"] is not False:
            errors.append("runtime_embedding_identity_mismatch: smoke_fallback_used must be false")
        if configured_normalize_embeddings is not None and type(configured_normalize_embeddings) is not bool:
            errors.append("runtime_embedding_identity_mismatch: normalize_embeddings must be an exact boolean")
        elif (
            configured_normalize_embeddings is not None
            and checks.get("manifest_normalize_embeddings_type_valid")
            and configured_normalize_embeddings is not manifest["normalize_embeddings"]
        ):
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"normalize_embeddings configured={configured_normalize_embeddings!r} "
                f"index={manifest['normalize_embeddings']!r}"
            )
    ok = (not errors and all(checks.values())) if formal else True
    return {
        "ok": ok,
        "actual": actual,
        "configured": configured,
        "index_manifest": manifest,
        "checks": checks,
        "errors": errors,
    }
