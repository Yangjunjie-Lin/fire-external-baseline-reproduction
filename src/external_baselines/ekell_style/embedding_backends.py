from __future__ import annotations

"""Embedding backends used by the E-KELL-native vector retrieval path."""

import hashlib
import importlib.metadata
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


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
    """Minimal text2vec-compatible embedding contract.

    Implementations expose ``encode`` with the same batch-oriented shape used by
    text2vec/SentenceModel while also reporting auditable backend metadata.
    """

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
    """Controlled wrapper around a text2vec-compatible model.

    ``model`` may be injected in tests or controlled environments. Otherwise
    text2vec's ``SentenceModel`` is imported lazily and initialized using only
    the configured ``model_name``.
    """

    model_name: str
    model_version: str = "unspecified"
    model: Any | None = None
    dimension: int = 0
    backend: str = "text2vec"
    actual_embedding_used: bool = True
    smoke_fallback_used: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be explicitly configured.")
        if self.model is None:
            try:
                from text2vec import SentenceModel  # type: ignore
            except ImportError as exc:
                raise EmbeddingBackendError(
                    "text2vec is required for the paper-fidelity embedding backend."
                ) from exc
            self.model = SentenceModel(self.model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        batch = [str(text) for text in texts]
        if not batch:
            return []
        encoder = getattr(self.model, "encode", None)
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


# Explicit alias: callers should never mistake this backend for a real model.
SmokeHashEmbeddingBackend = HashEmbeddingBackend


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
    """Create a controlled backend and enforce final-run smoke guards."""

    normalized = backend.strip().casefold().replace("-", "_")
    smoke_names = {"smoke", "hash", "hash_smoke", "deterministic_hash", "deterministic_hash_smoke"}
    if normalized in smoke_names:
        if paper_final or reject_smoke:
            raise EmbeddingBackendError(
                "Smoke hash embeddings are forbidden when paper_final=true or reject_smoke=true."
            )
        return HashEmbeddingBackend(
            dimension=dimension,
            model_name=model_name or "deterministic-hash-smoke",
            model_version=model_version if model_version != "unspecified" else "sha256-v1",
        )
    if normalized in {"text2vec", "sentence_model", "text2vec_compatible"}:
        return Text2VecEmbeddingBackend(
            model_name=model_name or "",
            model_version=model_version,
            model=model,
            dimension=dimension if dimension != 128 else 0,
        )
    raise ValueError(f"Unsupported E-KELL embedding backend: {backend}")


def embedding_package_versions() -> dict[str, str]:
    """Capture relevant package versions without importing optional packages."""

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
