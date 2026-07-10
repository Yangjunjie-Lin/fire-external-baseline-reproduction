"""Shared retrieval package (embedding backends + dense evidence indexes)."""

from external_baselines.retrieval.embedding_backends import (
    EmbeddingBackend,
    EmbeddingBackendError,
    HashEmbeddingBackend,
    SmokeHashEmbeddingBackend,
    Text2VecEmbeddingBackend,
    create_embedding_backend,
    embedding_package_versions,
    resolve_dimension,
    validate_embedding_backend,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBackendError",
    "HashEmbeddingBackend",
    "SmokeHashEmbeddingBackend",
    "Text2VecEmbeddingBackend",
    "create_embedding_backend",
    "embedding_package_versions",
    "resolve_dimension",
    "validate_embedding_backend",
]
