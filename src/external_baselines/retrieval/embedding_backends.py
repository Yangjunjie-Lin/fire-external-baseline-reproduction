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
    return vectors


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
    if (paper_final or reject_smoke) and (
        backend.smoke_fallback_used or not backend.actual_embedding_used
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
    actual_used = index_manifest.get("actual_embedding_used")
    smoke_used = index_manifest.get("smoke_fallback_used")
    return {
        "backend": str(index_manifest.get("backend") or ""),
        "model_name": str(
            index_manifest.get("model_name") or index_manifest.get("embedding_model") or ""
        ),
        "model_version": str(index_manifest.get("model_version") or ""),
        "dimension": int(index_manifest.get("dimension") or 0),
        "actual_embedding_used": actual_used if actual_used is None else bool(actual_used),
        "smoke_fallback_used": smoke_used if smoke_used is None else bool(smoke_used),
    }


def validate_runtime_embedding_identity(
    *,
    actual_backend: Any,
    configured_backend: str,
    configured_model_name: str,
    configured_model_version: str,
    configured_dimension: int,
    index_manifest: dict[str, Any],
    formal: bool,
) -> dict[str, Any]:
    actual = embedding_backend_identity(actual_backend)
    configured = {
        "backend": str(configured_backend or ""),
        "model_name": str(configured_model_name or ""),
        "model_version": str(configured_model_version or ""),
        "dimension": int(configured_dimension or 0),
    }
    manifest = _manifest_embedding_fields(index_manifest)
    checks: dict[str, bool] = {
        "manifest_backend_present": bool(manifest["backend"].strip()),
        "manifest_model_name_present": bool(manifest["model_name"].strip()),
        "manifest_model_version_present": bool(manifest["model_version"].strip()),
        "manifest_dimension_valid": manifest["dimension"] > 0,
        "manifest_actual_embedding_used_true": manifest["actual_embedding_used"] is True,
        "manifest_smoke_fallback_used_false": manifest["smoke_fallback_used"] is False,
        "backend_match": _normalize_backend_name(actual["backend"])
        == _normalize_backend_name(configured["backend"])
        == _normalize_backend_name(manifest["backend"]),
        "model_name_match": actual["model_name"].strip() == configured["model_name"].strip()
        == manifest["model_name"].strip(),
        "model_version_match": actual["model_version"].strip()
        == configured["model_version"].strip()
        == manifest["model_version"].strip(),
        "dimension_match": actual["dimension"] == configured["dimension"] == manifest["dimension"],
        "actual_embedding_used": actual["actual_embedding_used"] is True,
        "smoke_fallback_forbidden": actual["smoke_fallback_used"] is False,
    }
    errors: list[str] = []
    if formal:
        if not checks["manifest_backend_present"]:
            errors.append("runtime_embedding_manifest_missing: backend")
        if not checks["manifest_model_name_present"]:
            errors.append("runtime_embedding_manifest_missing: model_name")
        if not checks["manifest_model_version_present"]:
            errors.append("runtime_embedding_manifest_missing: model_version")
        if not checks["manifest_dimension_valid"]:
            errors.append("runtime_embedding_manifest_invalid: dimension")
        if "actual_embedding_used" not in index_manifest:
            errors.append("runtime_embedding_manifest_missing: actual_embedding_used")
        elif manifest["actual_embedding_used"] is not True:
            errors.append("runtime_embedding_manifest_invalid: actual_embedding_used must be true")
        if "smoke_fallback_used" not in index_manifest:
            errors.append("runtime_embedding_manifest_missing: smoke_fallback_used")
        elif manifest["smoke_fallback_used"] is not False:
            errors.append("runtime_embedding_manifest_invalid: smoke_fallback_used must be false")
        if not checks["backend_match"]:
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"backend configured={configured['backend']!r} actual={actual['backend']!r} "
                f"index={manifest['backend']!r}"
            )
        if actual["model_name"].strip() != configured["model_name"].strip():
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"model_name configured={configured['model_name']!r} actual={actual['model_name']!r}"
            )
        if not manifest["model_name"].strip() or actual["model_name"].strip() != manifest["model_name"].strip():
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"model_name actual={actual['model_name']!r} index={manifest['model_name']!r}"
            )
        if actual["model_version"].strip() != configured["model_version"].strip():
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"model_version configured={configured['model_version']!r} "
                f"actual={actual['model_version']!r}"
            )
        if not manifest["model_version"].strip() or actual["model_version"].strip() != manifest["model_version"].strip():
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"model_version actual={actual['model_version']!r} index={manifest['model_version']!r}"
            )
        if actual["dimension"] != configured["dimension"]:
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"dimension configured={configured['dimension']} actual={actual['dimension']}"
            )
        if manifest["dimension"] <= 0 or actual["dimension"] != manifest["dimension"]:
            errors.append(
                "runtime_embedding_identity_mismatch: "
                f"dimension actual={actual['dimension']} index={manifest['dimension']}"
            )
        if actual["actual_embedding_used"] is not True:
            errors.append("runtime_embedding_identity_mismatch: actual_embedding_used must be true")
        if actual["smoke_fallback_used"] is not False:
            errors.append("runtime_embedding_identity_mismatch: smoke_fallback_used must be false")
    ok = (not errors and all(checks.values())) if formal else True
    return {
        "ok": ok,
        "actual": actual,
        "configured": configured,
        "index_manifest": manifest,
        "checks": checks,
        "errors": errors,
    }
