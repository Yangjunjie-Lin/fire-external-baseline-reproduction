"""Dense embedding RAG with real text2vec indexes and smoke fixtures."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file
from external_baselines.common.io import ensure_dir, read_json, write_json
from external_baselines.common.schema import RetrievedContext, normalize_response_payload, retrieved_context_to_dict
from external_baselines.common.text_utils import compact_text, extract_json_object
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields
from external_baselines.retrieval.dense_index import (
    DenseIndexError,
    load_dense_index,
    query_dense_index,
)
from external_baselines.retrieval.dense_index import (
    build_dense_index as build_directory_index,
)
from external_baselines.retrieval.embedding_backends import (
    HashEmbeddingBackend,
    create_embedding_backend,
    validate_embedding_backend,
)

METHOD = "dense_rag"


def _hash_embed(text: str, dim: int = 64) -> list[float]:
    """Smoke-only hash embedding retained for enhanced/legacy callers."""
    return HashEmbeddingBackend(dimension=dim)._embed(str(text))


def cosine(a: list[float], b: list[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _is_smoke_backend(backend: str) -> bool:
    return backend.strip().casefold().replace("-", "_") in {
        "smoke",
        "hash",
        "smoke_hash",
        "smoke_hash_embedding",
        "deterministic_hash_smoke",
    }


def _legacy_json_index_path(path: Path) -> bool:
    return path.suffix.lower() == ".json"


class DenseIndex:
    """Compatibility wrapper used by hybrid_rag and older callers."""

    def __init__(
        self,
        *,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        model_name: str,
        model_version: str | None,
        backend: str,
        dim: int,
        checksum: str | None,
        build_manifest: dict[str, Any],
        index_dir: Path | None = None,
        directory_payload: dict[str, Any] | None = None,
    ) -> None:
        self.documents = documents
        self.embeddings = embeddings
        self.model_name = model_name
        self.model_version = model_version
        self.backend = backend
        self.dim = dim
        self.checksum = checksum
        self.build_manifest = build_manifest
        self.index_dir = index_dir
        self.directory_payload = directory_payload

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if _legacy_json_index_path(path) or _is_smoke_backend(self.backend):
            ensure_dir(path.parent)
            write_json(
                path,
                {
                    "documents": self.documents,
                    "embeddings": self.embeddings,
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "backend": self.backend,
                    "dim": self.dim,
                    "checksum": self.checksum,
                    "build_manifest": self.build_manifest,
                },
            )
            return
        raise DenseIndexError("Directory indexes are persisted via build_dense_index().")

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        path = Path(path)
        if path.is_dir() or (path / "index_manifest.json").is_file():
            payload = load_dense_index(path if path.is_dir() else path.parent)
            manifest = payload["manifest"]
            return cls(
                documents=payload["documents"],
                embeddings=payload["embeddings"],
                model_name=str(payload["model_name"]),
                model_version=payload.get("model_version"),
                backend=str(payload["backend"]),
                dim=int(payload["dimension"]),
                checksum=payload.get("checksum"),
                build_manifest=dict(manifest),
                index_dir=Path(payload["index_dir"]),
                directory_payload=payload,
            )
        data = read_json(path)
        return cls(
            documents=list(data.get("documents") or []),
            embeddings=list(data.get("embeddings") or []),
            model_name=str(data.get("model_name") or "unknown"),
            model_version=data.get("model_version"),
            backend=str(data.get("backend") or "unknown"),
            dim=int(data.get("dim") or data.get("dimension") or 0),
            checksum=data.get("checksum"),
            build_manifest=dict(data.get("build_manifest") or {}),
        )


def build_dense_index(
    evidence_path: str | Path,
    *,
    model_name: str = "smoke-hash-embedding",
    model_version: str | None = "v0-smoke",
    backend: str = "smoke_hash_embedding",
    dim: int = 64,
    cache_path: str | Path | None = None,
    embedding_model: Any | None = None,
    batch_size: int = 16,
    normalize_embeddings: bool = True,
    paper_final: bool = False,
    reject_smoke: bool = False,
    corpus_checksum: str | None = None,
) -> DenseIndex:
    """Build dense index. Smoke uses legacy JSON; real backends use directory format."""
    evidence_path = Path(evidence_path)
    emb = create_embedding_backend(
        backend,
        model_name=model_name,
        model_version=str(model_version or "unspecified"),
        dimension=dim,
        paper_final=paper_final,
        reject_smoke=reject_smoke,
        model=embedding_model,
    )
    validate_embedding_backend(emb, paper_final=paper_final, reject_smoke=reject_smoke)

    if _is_smoke_backend(backend):
        # Keep deterministic smoke JSON path for CI fixtures.
        from external_baselines.common.io import read_jsonl

        docs_raw = read_jsonl(evidence_path)
        documents: list[dict[str, Any]] = []
        texts: list[str] = []
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
            texts.append(text)
        embeddings = emb.encode(texts)
        from external_baselines.common.checksums import sha256_json

        index_checksum = sha256_json(
            {"docs": [d["chunk_id"] for d in documents], "model": model_name, "backend": backend, "dim": dim}
        )
        manifest = {
            "evidence_path": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
            "model_name": model_name,
            "model_version": model_version,
            "backend": backend,
            "dim": dim,
            "doc_count": len(documents),
            "index_checksum": index_checksum,
            "real_embedding_model_used": False,
            "actual_embedding_used": False,
            "smoke_fallback_used": True,
        }
        index = DenseIndex(
            documents=documents,
            embeddings=embeddings,
            model_name=model_name,
            model_version=model_version,
            backend=backend,
            dim=dim,
            checksum=index_checksum,
            build_manifest=manifest,
        )
        if cache_path:
            index.save(cache_path)
            write_json(Path(cache_path).with_suffix(".manifest.json"), manifest)
        return index

    if not cache_path:
        raise DenseIndexError("Real dense indexes require an index_dir (cache_path).")
    index_dir = Path(cache_path)
    if index_dir.suffix.lower() == ".json":
        raise DenseIndexError(
            "Real dense indexes must use a directory path (documents.jsonl + embeddings.npy), not a .json file."
        )
    if index_dir.is_dir() and (index_dir / "index_manifest.json").is_file():
        # Reuse if valid.
        try:
            payload = load_dense_index(
                index_dir,
                expected_model_name=model_name,
                expected_model_version=str(model_version or ""),
                expected_corpus_checksum=corpus_checksum,
                expected_backend=backend,
                expected_dimension=dim if dim > 0 else None,
            )
            return DenseIndex(
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
        except DenseIndexError:
            if paper_final or reject_smoke:
                raise
            # Non-formal smoke/dev may rebuild below.
            pass

    manifest = build_directory_index(
        evidence_path,
        emb,
        index_dir,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        corpus_checksum=corpus_checksum or sha256_file(evidence_path),
        paper_final=paper_final,
        reject_smoke=reject_smoke,
    )
    payload = load_dense_index(index_dir)
    return DenseIndex(
        documents=payload["documents"],
        embeddings=payload["embeddings"],
        model_name=str(payload["model_name"]),
        model_version=payload.get("model_version"),
        backend=str(payload["backend"]),
        dim=int(payload["dimension"]),
        checksum=payload.get("checksum"),
        build_manifest=dict(manifest),
        index_dir=index_dir,
        directory_payload=payload,
    )


class DenseRetriever:
    def __init__(self, index: DenseIndex, embedding_backend: Any | None = None) -> None:
        self.index = index
        self.embedding_backend = embedding_backend

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedContext]:
        if self.index.directory_payload is not None:
            if self.embedding_backend is None:
                raise DenseIndexError("Directory dense index query requires embedding_backend.")
            return query_dense_index(
                self.index.directory_payload,
                query,
                self.embedding_backend,
                top_k=top_k,
                normalize_embeddings=bool(self.index.build_manifest.get("normalize_embeddings", True)),
            )
        # Smoke / legacy JSON path
        if not self.index.documents:
            return []
        if self.embedding_backend is None:
            self.embedding_backend = create_embedding_backend(
                self.index.backend,
                model_name=self.index.model_name,
                model_version=str(self.index.model_version or "unspecified"),
                dimension=self.index.dim or 64,
                reject_smoke=False,
            )
        q = self.embedding_backend.embed_query(query)
        scored = []
        for i, emb in enumerate(self.index.embeddings):
            scored.append((sum(float(a) * float(b) for a, b in zip(q, emb)), i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        contexts: list[RetrievedContext] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            doc = self.index.documents[idx]
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
                        "embedding_backend": self.index.backend,
                        "embedding_model": self.index.model_name,
                        "embedding_model_version": self.index.model_version,
                        "dense_rank": rank,
                        "dense_score": round(float(score), 6),
                        "index_checksum": self.index.checksum,
                    },
                )
            )
        return contexts


def run_scenario(
    scenario: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    llm=None,
    runtime=None,
) -> dict[str, Any]:
    from external_baselines.common.llm_client import build_llm_client, llm_config_summary, llm_runtime_snapshot
    from external_baselines.common.method_runtime import prepare_dense_runtime

    config = config or {}
    llm = llm or build_llm_client(config)
    dense_cfg = config.get("dense_rag", {})
    top_k = int(config.get("retrieval", {}).get("top_k", dense_cfg.get("top_k", 5)))
    paper_final = bool(config.get("paper_final", False))
    reject_smoke = bool(dense_cfg.get("reject_smoke", False) or paper_final)
    start = time.perf_counter()

    if runtime is None:
        if paper_final or reject_smoke:
            # Still allow prepare once for standalone calls; runner should pass runtime.
            runtime = prepare_dense_runtime(config)
        else:
            runtime = prepare_dense_runtime(config)
    if getattr(runtime, "audit", None) is not None:
        runtime.audit.case_count += 1

    index = runtime.dense_index
    retriever = runtime.retriever
    contexts = [retrieved_context_to_dict(c) for c in retriever.retrieve(scenario["scenario_text"], top_k=top_k)]

    system = (
        "You are reproducing a dense-embedding RAG emergency decision-support baseline. "
        "Use only retrieved contexts and the scenario. Do not use SAFE modules. Return valid JSON."
    )
    ctx_text = "\n\n".join(
        f"[context_id={c.get('context_id')} source_id={c.get('source_id')} citation={c.get('citation')} score={c.get('score')}]\n{c.get('text')}"
        for c in contexts
    ) or "(none)"
    user = f"""Scenario:
{scenario['scenario_text']}

Retrieved contexts:
{ctx_text}

Return JSON with:
- situation_summary
- key_risks
- recommended_actions
- blocked_or_unsafe_actions
- missing_confirmations
- supporting_evidence
- citations
- final_decision_gate
""".strip()
    raw_text = llm.complete(
        system=system,
        user=user,
        temperature=float(config.get("llm", {}).get("temperature", 0.0)),
        max_tokens=int(config.get("llm", {}).get("max_tokens", 1200)),
        top_p=config.get("llm", {}).get("top_p"),
        seed=config.get("llm", {}).get("seed"),
    )
    payload = extract_json_object(raw_text) or {"situation_summary": raw_text}
    parsing_failure = not bool(extract_json_object(raw_text))
    output = normalize_response_payload(payload, scenario_id=scenario["scenario_id"], method=METHOD)
    output.retrieved_contexts = contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {"text": raw_text, "parsed": payload}
    real_dense = bool(index.build_manifest.get("actual_embedding_used") or index.build_manifest.get("real_embedding_model_used"))
    output.method_specific = {
        "baseline_name": "Dense embedding RAG baseline",
        "reproduction_class": "controlled_supplemental" if real_dense else "smoke_fixture",
        "llm_config_summary": llm_config_summary(config, llm),
        "retrieval_used": True,
        "retrieval_backend": "dense",
        "embedding_backend": index.backend,
        "embedding_model": index.model_name,
        "embedding_model_version": index.model_version,
        "index_checksum": index.checksum,
        "corpus_checksum": index.build_manifest.get("corpus_checksum") or index.build_manifest.get("evidence_sha256"),
        "index_build_manifest": index.build_manifest,
        "runtime_reuse": runtime.audit.to_dict() if getattr(runtime, "audit", None) else None,
        "runtime": llm_runtime_snapshot(llm),
        "parsing_failure": parsing_failure,
        "parsing_status": "failed" if parsing_failure else "ok",
        "structured_safety_fields": "baseline_generated_only",
    }
    return maybe_infer_structured_safety_fields(output.to_dict(), config)
