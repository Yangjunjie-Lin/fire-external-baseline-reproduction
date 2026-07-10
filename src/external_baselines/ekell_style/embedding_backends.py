"""Compatibility re-export — shared backends live in external_baselines.retrieval."""

from external_baselines.retrieval.embedding_backends import (  # noqa: F401
    EmbeddingBackend,
    EmbeddingBackendError,
    HashEmbeddingBackend,
    SmokeHashEmbeddingBackend,
    Text2VecEmbeddingBackend,
    create_embedding_backend,
    embedding_package_versions,
    validate_embedding_backend,
)
