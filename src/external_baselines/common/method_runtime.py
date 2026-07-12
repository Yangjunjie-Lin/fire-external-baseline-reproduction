"""Method-level runtime preparation: load embedding/index once, reuse across cases."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from external_baselines.interop.schema import canonicalize_method_id
from external_baselines.retrieval.embedding_backends import (
    create_embedding_backend,
    resolve_dimension,
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

    def close(self) -> None:
        return None


@dataclass
class HybridRuntime:
    lexical_retriever: Any
    dense_runtime: DenseRuntime
    audit: RuntimeAudit = field(default_factory=RuntimeAudit)

    def close(self) -> None:
        if self.dense_runtime is not None:
            self.dense_runtime.close()


@dataclass
class EKELLRuntime:
    kg: Any
    embedding_backend: Any
    vector_index: Any
    vector_retriever: Any
    index_manifest: dict[str, Any]
    audit: RuntimeAudit = field(default_factory=RuntimeAudit)
    index_built_during_run: bool = False

    def close(self) -> None:
        return None


# Shared cache keyed by (kind, path, model_version) within one process/run.
_RUNTIME_CACHE: dict[tuple[str, str, str], Any] = {}


def clear_runtime_cache() -> None:
    _RUNTIME_CACHE.clear()


def _cache_get(kind: str, path: str, model_version: str) -> Any | None:
    return _RUNTIME_CACHE.get((kind, str(path), str(model_version)))


def _cache_set(kind: str, path: str, model_version: str, value: Any) -> None:
    _RUNTIME_CACHE[(kind, str(path), str(model_version))] = value


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
    cached = _cache_get("dense", cache_key_path, model_version)
    if cached is not None:
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
    )
    _cache_set("dense", cache_key_path, model_version, runtime)
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
        cached = _cache_get("ekell", str(index_path), model_version)
        if cached is not None:
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
    )
    if index_path:
        _cache_set("ekell", str(index_path), model_version, runtime)
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


def close_method_runtime(runtime: Any | None) -> None:
    if runtime is None:
        return
    closer = getattr(runtime, "close", None)
    if callable(closer):
        closer()


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
