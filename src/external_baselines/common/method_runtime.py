"""Method-level runtime preparation: load embedding/index once, reuse across cases."""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from external_baselines.common.checksums import sha256_file
from external_baselines.common.decision_suite_guard import sanitize_error_message
from external_baselines.interop.schema import canonicalize_method_id
from external_baselines.retrieval.embedding_backends import (
    EmbeddingBackendError,
    create_embedding_backend,
    embedding_backend_identity,
    resolve_dimension,
    validate_runtime_embedding_identity,
)


@dataclass
class RuntimeAudit:
    embedding_model_load_count: int = 0
    index_load_count: int = 0
    case_count: int = 0
    reused_across_cases: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model_load_count": self.embedding_model_load_count,
            "index_load_count": self.index_load_count,
            "case_count": self.case_count,
            "reused_across_cases": self.reused_across_cases and self.case_count > 1,
        }


@dataclass
class DenseRuntime:
    embedding_backend: Any
    dense_index: Any
    retriever: Any
    index_manifest: dict[str, Any]
    audit: RuntimeAudit = field(default_factory=RuntimeAudit)
    index_built_during_run: bool = False
    embedding_identity_report: dict[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, repr=False, compare=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True


@dataclass
class HybridRuntime:
    lexical_retriever: Any
    dense_runtime: DenseRuntime
    audit: RuntimeAudit = field(default_factory=RuntimeAudit)
    _closed: bool = field(default=False, repr=False, compare=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closer = getattr(self.lexical_retriever, "close", None)
        if callable(closer):
            closer()


class RuntimeCleanupError(RuntimeError):
    """Raised when runtime cache cleanup fails without a suite body exception."""


logger = logging.getLogger(__name__)


@dataclass
class EKELLRuntime:
    kg: Any
    embedding_backend: Any
    vector_index: Any
    vector_retriever: Any
    index_manifest: dict[str, Any]
    audit: RuntimeAudit = field(default_factory=RuntimeAudit)
    index_built_during_run: bool = False
    embedding_identity_report: dict[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, repr=False, compare=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True


# Direct-call cache for tests/tools outside a suite scope.
_DIRECT_RUNTIME_CACHE: dict[tuple[str, ...], Any] = {}
_ACTIVE_RUNTIME_CACHE: ContextVar[dict[tuple[str, ...], Any] | None] = ContextVar(
    "external_baseline_runtime_cache",
    default=None,
)


def _current_runtime_cache() -> dict[tuple[str, ...], Any]:
    cache = _ACTIVE_RUNTIME_CACHE.get()
    if cache is None:
        return _DIRECT_RUNTIME_CACHE
    return cache


def clear_runtime_cache() -> None:
    _current_runtime_cache().clear()


def _index_manifest_checksum(index_path: str | Path) -> str:
    manifest_path = Path(index_path) / "index_manifest.json"
    if manifest_path.is_file():
        return sha256_file(manifest_path)
    return ""


def _runtime_cache_key(
    kind: str,
    *,
    index_path: str,
    model_name: str,
    model_version: str,
    dimension: int,
    corpus_checksum: str | None,
    index_manifest_checksum: str,
) -> tuple[str, ...]:
    return (
        kind,
        str(index_path),
        str(model_name),
        str(model_version),
        str(int(dimension or 0)),
        str(corpus_checksum or ""),
        str(index_manifest_checksum or ""),
    )


def _cache_get(key: tuple[str, ...]) -> Any | None:
    return _current_runtime_cache().get(key)


def _cache_set(key: tuple[str, ...], value: Any) -> None:
    _current_runtime_cache()[key] = value


def assert_cached_runtime_compatible(
    cached_runtime: Any,
    requested_backend: Any | None,
    requested_config: dict[str, Any],
    *,
    formal: bool,
    config_section: str = "dense_rag",
) -> None:
    section = requested_config.get(config_section) or {}
    configured_backend = str(section.get("backend") or "")
    configured_model_name = str(section.get("model_name") or "")
    configured_model_version = str(section.get("model_version") or "unspecified")
    configured_dimension = resolve_dimension(section, 64)
    manifest = dict(getattr(cached_runtime, "index_manifest", None) or {})
    cached_backend = getattr(cached_runtime, "embedding_backend", None)
    if requested_backend is not None and cached_backend is not None:
        if embedding_backend_identity(cached_backend) != embedding_backend_identity(requested_backend):
            raise EmbeddingBackendError(
                "runtime_embedding_identity_mismatch: injected backend differs from cached runtime backend"
            )
    if cached_backend is None:
        return
    report = validate_runtime_embedding_identity(
        actual_backend=cached_backend,
        configured_backend=configured_backend,
        configured_model_name=configured_model_name,
        configured_model_version=configured_model_version,
        configured_dimension=configured_dimension,
        index_manifest=manifest,
        formal=formal,
    )
    if formal and not report.get("ok"):
        raise EmbeddingBackendError("; ".join(report.get("errors") or ["runtime_embedding_identity_mismatch"]))


def prepare_dense_runtime(
    config: dict[str, Any],
    *,
    embedding_backend: Any | None = None,
) -> DenseRuntime:
    from external_baselines.dense_rag.pipeline import DenseIndex, DenseRetriever, build_dense_index
    from external_baselines.retrieval.dense_index import DenseIndexError, load_dense_index

    dense_cfg = config.get("dense_rag") or {}
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    evidence_path = corpus_dir / "evidence_chunks.jsonl"
    backend = str(dense_cfg.get("backend", "smoke_hash_embedding"))
    model_name = str(dense_cfg.get("model_name", "smoke-hash-embedding"))
    model_version = str(dense_cfg.get("model_version", "v0-smoke"))
    dim = resolve_dimension(dense_cfg, 64)
    reject_smoke = bool(dense_cfg.get("reject_smoke", False) or config.get("paper_final", False))
    paper_final = bool(config.get("paper_final", False))
    allow_rebuild = bool(dense_cfg.get("allow_index_rebuild", False))
    if paper_final or reject_smoke:
        allow_rebuild = False
    cache_path = dense_cfg.get("index_path") or str(
        Path(config.get("paths", {}).get("output_dir", "outputs")) / "dense_index_smoke.json"
    )
    corpus_checksum = config.get("corpus_checksum") or (config.get("paths") or {}).get("corpus_checksum")

    cache_key_path = str(cache_path)
    manifest_checksum = _index_manifest_checksum(cache_key_path)
    cache_key = _runtime_cache_key(
        "dense",
        index_path=cache_key_path,
        model_name=model_name,
        model_version=model_version,
        dimension=dim,
        corpus_checksum=str(corpus_checksum or ""),
        index_manifest_checksum=manifest_checksum,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        assert_cached_runtime_compatible(
            cached,
            embedding_backend,
            config,
            formal=paper_final or reject_smoke,
        )
        cached.audit.embedding_model_load_count = getattr(
            cached.embedding_backend, "_load_count", cached.audit.embedding_model_load_count
        )
        return cached

    if embedding_backend is not None:
        emb = embedding_backend
    else:
        emb = create_embedding_backend(
            backend,
            model_name=model_name,
            model_version=model_version,
            dimension=dim,
            paper_final=paper_final,
            reject_smoke=reject_smoke,
            model=dense_cfg.get("injected_model"),
        )

    index_dir = Path(cache_path)
    index_built_during_run = False
    if index_dir.is_dir() and (index_dir / "index_manifest.json").is_file():
        payload = load_dense_index(
            index_dir,
            expected_model_name=model_name,
            expected_model_version=model_version,
            expected_corpus_checksum=corpus_checksum,
            expected_backend=backend,
            expected_dimension=dim if dim > 0 else None,
        )
        index = DenseIndex(
            documents=payload["documents"],
            embeddings=payload["embeddings"],
            model_name=str(payload["model_name"]),
            model_version=payload.get("model_version"),
            backend=str(payload["backend"]),
            dim=int(payload["dimension"]),
            checksum=payload.get("checksum"),
            build_manifest=dict(payload["manifest"]),
            index_dir=Path(payload["index_dir"]),
            directory_payload=payload,
        )
        index_load_count = 1
        manifest = dict(payload["manifest"])
    elif paper_final or reject_smoke or not allow_rebuild:
        if index_dir.is_file() and index_dir.suffix.lower() == ".json":
            if paper_final or reject_smoke:
                raise DenseIndexError("legacy_dense_json_forbidden_in_formal")
        raise DenseIndexError(
            f"Dense index missing or invalid at {cache_path}; "
            "formal/reject_smoke mode forbids pipeline rebuild. "
            "Use scripts/build_comparison_indexes.py."
        )
    else:
        index = build_dense_index(
            evidence_path,
            model_name=model_name,
            model_version=model_version,
            backend=backend,
            dim=dim,
            cache_path=cache_path,
            embedding_model=dense_cfg.get("injected_model"),
            batch_size=int(dense_cfg.get("batch_size", 16)),
            normalize_embeddings=bool(dense_cfg.get("normalize_embeddings", True)),
            paper_final=paper_final,
            reject_smoke=reject_smoke,
            corpus_checksum=corpus_checksum,
        )
        index_built_during_run = True
        index_load_count = 1
        manifest = dict(index.build_manifest)

    if emb.dimension in (0, None) and index.dim:
        emb.dimension = int(index.dim)
    identity_report = validate_runtime_embedding_identity(
        actual_backend=emb,
        configured_backend=backend,
        configured_model_name=model_name,
        configured_model_version=model_version,
        configured_dimension=dim,
        index_manifest=manifest,
        formal=paper_final or reject_smoke,
    )
    if (paper_final or reject_smoke) and not identity_report.get("ok"):
        raise EmbeddingBackendError("; ".join(identity_report.get("errors") or ["runtime_embedding_identity_mismatch"]))
    retriever = DenseRetriever(index, embedding_backend=emb)
    runtime = DenseRuntime(
        embedding_backend=emb,
        dense_index=index,
        retriever=retriever,
        index_manifest=manifest,
        audit=RuntimeAudit(
            embedding_model_load_count=getattr(emb, "_load_count", 0),
            index_load_count=index_load_count,
        ),
        index_built_during_run=index_built_during_run,
        embedding_identity_report=identity_report,
    )
    _cache_set(cache_key, runtime)
    return runtime


def prepare_hybrid_runtime(
    config: dict[str, Any],
    *,
    embedding_backend: Any | None = None,
) -> HybridRuntime:
    from external_baselines.vanilla_rag.retriever import LexicalRetriever

    hybrid_cfg = config.get("hybrid_rag") or {}
    dense_cfg = dict(config.get("dense_rag") or {})
    if hybrid_cfg.get("dense_model_name") and not dense_cfg.get("model_name"):
        dense_cfg["model_name"] = hybrid_cfg["dense_model_name"]
    if hybrid_cfg.get("dense_model_version") and not dense_cfg.get("model_version"):
        dense_cfg["model_version"] = hybrid_cfg["dense_model_version"]
    if hybrid_cfg.get("dense_method") and not dense_cfg.get("backend"):
        dense_cfg["backend"] = hybrid_cfg["dense_method"]
    if hybrid_cfg.get("dimension") and not dense_cfg.get("dimension"):
        dense_cfg["dimension"] = hybrid_cfg["dimension"]
    if hybrid_cfg.get("reject_smoke") is not None and "reject_smoke" not in dense_cfg:
        dense_cfg["reject_smoke"] = hybrid_cfg["reject_smoke"]
    if hybrid_cfg.get("dense_index_path") and not dense_cfg.get("index_path"):
        dense_cfg["index_path"] = hybrid_cfg["dense_index_path"]
    dense_cfg.setdefault("allow_index_rebuild", False)
    merged = dict(config)
    merged["dense_rag"] = dense_cfg

    dense_runtime = prepare_dense_runtime(merged, embedding_backend=embedding_backend)
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    lexical = LexicalRetriever.from_jsonl(str(corpus_dir / "evidence_chunks.jsonl"))
    runtime = HybridRuntime(
        lexical_retriever=lexical,
        dense_runtime=dense_runtime,
        audit=RuntimeAudit(
            embedding_model_load_count=dense_runtime.audit.embedding_model_load_count,
            index_load_count=dense_runtime.audit.index_load_count,
        ),
    )
    return runtime


def prepare_ekell_runtime(
    config: dict[str, Any],
    *,
    embedding_backend: Any | None = None,
) -> EKELLRuntime:
    from external_baselines.ekell_style.embedding_backends import create_embedding_backend
    from external_baselines.ekell_style.kg_loader import load_kg
    from external_baselines.ekell_style.vector_index import VectorIndexError
    from external_baselines.ekell_style.vector_retriever import VectorRetriever

    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    ekell_cfg = config.get("ekell_style") or {}
    vector_cfg = config.get("ekell_vector") or ekell_cfg.get("vector") or {}
    paper_final = bool(config.get("paper_final", False))
    reject_smoke = bool(vector_cfg.get("reject_smoke", paper_final))
    index_path = vector_cfg.get("index_path")
    backend_name = str(vector_cfg.get("backend", "smoke"))
    model_name = str(
        vector_cfg.get("model_name")
        or ("deterministic-hash-smoke" if "smoke" in backend_name or "hash" in backend_name else "")
    )
    model_version = str(vector_cfg.get("model_version") or "unspecified")
    dimension = int(vector_cfg.get("dimension", vector_cfg.get("dim", 64)) or 64)

    if index_path:
        manifest_checksum = _index_manifest_checksum(str(index_path))
        cache_key = _runtime_cache_key(
            "ekell",
            index_path=str(index_path),
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            corpus_checksum=str(config.get("corpus_checksum") or (config.get("paths") or {}).get("corpus_checksum") or ""),
            index_manifest_checksum=manifest_checksum,
        )
        cached = _cache_get(cache_key)
        if cached is not None:
            assert_cached_runtime_compatible(
                cached,
                embedding_backend,
                config,
                formal=paper_final or reject_smoke,
                config_section="ekell_vector",
            )
            return cached

    kg = load_kg(corpus_dir)
    if embedding_backend is not None:
        emb = embedding_backend
    else:
        emb = create_embedding_backend(
            backend_name,
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            paper_final=paper_final,
            reject_smoke=reject_smoke,
            model=vector_cfg.get("injected_model"),
        )

    index_dir = Path(str(index_path)) if index_path else None
    index_built_during_run = False
    if index_dir and index_dir.is_dir() and (index_dir / "index_manifest.json").is_file():
        retriever = VectorRetriever.from_index_directory(
            index_dir,
            emb,
            paper_final=paper_final,
            reject_smoke=reject_smoke,
            max_context_chars=int((config.get("retrieval") or {}).get("max_chunk_chars", 1200)),
            expected_dimension=dimension if dimension > 0 else None,
        )
        index = retriever.index
        index_load_count = 1
        manifest = dict(index.metadata)
    elif paper_final or reject_smoke:
        if index_dir and index_dir.is_file() and index_dir.suffix.lower() == ".json":
            raise VectorIndexError("legacy_ekell_json_forbidden_in_formal")
        raise VectorIndexError(
            f"E-KELL formal/reject_smoke requires a persisted index at index_path={index_path!r}; "
            "refusing to rebuild per case. Use scripts/build_comparison_indexes.py."
        )
    else:
        # Smoke / non-formal: allow temporary from_kg build.
        retriever = VectorRetriever.from_kg(
            kg,
            emb,
            paper_final=paper_final,
            reject_smoke=reject_smoke,
            max_context_chars=int((config.get("retrieval") or {}).get("max_chunk_chars", 1200)),
        )
        index = retriever.index
        index_load_count = 1
        manifest = dict(index.metadata)
        index_built_during_run = True
        if index_dir is not None:
            manifest = index.save_directory(index_dir)

    if emb.dimension in (0, None) and manifest.get("dimension"):
        emb.dimension = int(manifest["dimension"])
    identity_report = validate_runtime_embedding_identity(
        actual_backend=emb,
        configured_backend=backend_name,
        configured_model_name=model_name,
        configured_model_version=model_version,
        configured_dimension=dimension,
        index_manifest=manifest,
        formal=paper_final or reject_smoke,
    )
    if (paper_final or reject_smoke) and not identity_report.get("ok"):
        raise EmbeddingBackendError("; ".join(identity_report.get("errors") or ["runtime_embedding_identity_mismatch"]))

    runtime = EKELLRuntime(
        kg=kg,
        embedding_backend=emb,
        vector_index=index,
        vector_retriever=retriever,
        index_manifest=manifest,
        audit=RuntimeAudit(
            embedding_model_load_count=getattr(emb, "_load_count", 0),
            index_load_count=index_load_count,
        ),
        index_built_during_run=index_built_during_run,
        embedding_identity_report=identity_report,
    )
    if index_path:
        _cache_set(cache_key, runtime)
    return runtime


def prepare_method_runtime(
    method_id: str,
    config: dict[str, Any],
    *,
    embedding_backend: Any | None = None,
) -> Any | None:
    mid = canonicalize_method_id(method_id)
    if mid == "dense_rag":
        return prepare_dense_runtime(config, embedding_backend=embedding_backend)
    if mid == "hybrid_rag":
        return prepare_hybrid_runtime(config, embedding_backend=embedding_backend)
    if mid in {"ekell_style_controlled_shared_llm", "ekell_style_paper_fidelity"}:
        return prepare_ekell_runtime(config, embedding_backend=embedding_backend)
    return None


def runtime_is_cached(runtime: Any | None) -> bool:
    if runtime is None:
        return False
    cache = _current_runtime_cache()
    return any(cached is runtime for cached in cache.values())


def close_method_runtime(runtime: Any | None) -> None:
    if runtime is None:
        return
    closer = getattr(runtime, "close", None)
    if callable(closer):
        closer()


def _close_cache_runtimes_safely(cache: dict[tuple[str, ...], Any]) -> list[str]:
    errors: list[str] = []
    seen: set[int] = set()
    for runtime in cache.values():
        runtime_id = id(runtime)
        if runtime_id in seen:
            continue
        seen.add(runtime_id)
        try:
            close_method_runtime(runtime)
        except Exception as exc:  # noqa: BLE001
            errors.append(sanitize_error_message(str(exc)))
    return errors


@contextmanager
def runtime_cache_scope() -> Iterator[dict[tuple[str, ...], Any]]:
    cache: dict[tuple[str, ...], Any] = {}
    token = _ACTIVE_RUNTIME_CACHE.set(cache)
    body_exception: BaseException | None = None
    try:
        yield cache
    except BaseException as exc:
        body_exception = exc
        raise
    finally:
        close_errors = _close_cache_runtimes_safely(cache)
        cache.clear()
        _ACTIVE_RUNTIME_CACHE.reset(token)
        if close_errors:
            for message in close_errors:
                logger.warning("runtime_cache_close_failed: %s", message)
            if body_exception is None:
                raise RuntimeCleanupError(
                    "; ".join(close_errors) or "runtime_cache_close_failed"
                ) from None


def pipeline_accepts_runtime(pipeline: Callable[..., Any]) -> bool:
    try:
        return "runtime" in inspect.signature(pipeline).parameters
    except (TypeError, ValueError):
        return False


def runtime_index_checksum(runtime: Any | None) -> str | None:
    if runtime is None:
        return None
    if isinstance(runtime, HybridRuntime):
        return runtime_index_checksum(runtime.dense_runtime)
    manifest = getattr(runtime, "index_manifest", None) or {}
    if isinstance(manifest, dict):
        return manifest.get("index_checksum") or manifest.get("checksum")
    index = getattr(runtime, "dense_index", None)
    if index is not None:
        return getattr(index, "checksum", None)
    return None
