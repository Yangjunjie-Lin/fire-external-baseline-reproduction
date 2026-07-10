from __future__ import annotations

"""Dense embedding retrieval with optional smoke fixture.

Real dense experiments require an embedding model. Without one, this module
exposes the interface and a deterministic smoke fixture, and must not claim a
completed dense paper run.
"""

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file, sha256_json
from external_baselines.common.io import ensure_dir, read_json, read_jsonl, write_json
from external_baselines.common.schema import RetrievedContext
from external_baselines.common.text_utils import compact_text, tokenize


def _hash_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-token hashing vector for smoke fixtures only."""
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] -= 0.25
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class DenseIndex:
    documents: list[dict[str, Any]]
    embeddings: list[list[float]]
    model_name: str
    model_version: str | None = None
    backend: str = "smoke_hash_embedding"
    dim: int = 64
    checksum: str | None = None
    build_manifest: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        payload = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "backend": self.backend,
            "dim": self.dim,
            "checksum": self.checksum,
            "build_manifest": self.build_manifest,
        }
        write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        payload = read_json(path)
        return cls(
            documents=list(payload.get("documents") or []),
            embeddings=list(payload.get("embeddings") or []),
            model_name=str(payload.get("model_name") or "unknown"),
            model_version=payload.get("model_version"),
            backend=str(payload.get("backend") or "unknown"),
            dim=int(payload.get("dim") or 0),
            checksum=payload.get("checksum"),
            build_manifest=dict(payload.get("build_manifest") or {}),
        )


def build_dense_index(
    evidence_path: str | Path,
    *,
    model_name: str = "smoke-hash-embedding",
    model_version: str | None = "v0-smoke",
    backend: str = "smoke_hash_embedding",
    dim: int = 64,
    cache_path: str | Path | None = None,
) -> DenseIndex:
    """Build a dense index. Smoke backend is deterministic and not a real embedding model."""
    docs_raw = read_jsonl(evidence_path)
    documents: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    for i, doc in enumerate(docs_raw):
        text = str(doc.get("text") or doc.get("content") or doc.get("chunk") or doc.get("body") or "").strip()
        if not text:
            continue
        cid = str(doc.get("chunk_id") or doc.get("id") or doc.get("source_id") or f"chunk_{i}")
        row = {
            "chunk_id": cid,
            "text": text,
            "source_id": doc.get("source_id") or doc.get("source") or doc.get("document_id"),
            "citation": doc.get("citation") or doc.get("url"),
            "metadata": {k: v for k, v in doc.items() if k not in {"text", "content", "chunk", "body"}},
        }
        documents.append(row)
        if backend == "smoke_hash_embedding":
            embeddings.append(_hash_embed(text, dim=dim))
        else:
            raise RuntimeError(
                f"Dense embedding backend '{backend}' is not available in this environment. "
                "Install/configure a real embedding model, or use smoke_hash_embedding for fixtures only."
            )

    checksum = sha256_json({"docs": [d["chunk_id"] for d in documents], "model": model_name, "backend": backend, "dim": dim})
    manifest = {
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "model_name": model_name,
        "model_version": model_version,
        "backend": backend,
        "dim": dim,
        "doc_count": len(documents),
        "index_checksum": checksum,
        "real_embedding_model_used": backend != "smoke_hash_embedding",
    }
    index = DenseIndex(
        documents=documents,
        embeddings=embeddings,
        model_name=model_name,
        model_version=model_version,
        backend=backend,
        dim=dim,
        checksum=checksum,
        build_manifest=manifest,
    )
    if cache_path:
        index.save(cache_path)
        # Also write sidecar manifest for audit.
        write_json(Path(cache_path).with_suffix(".manifest.json"), manifest)
    return index


@dataclass
class DenseRetriever:
    index: DenseIndex

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedContext]:
        if not self.index.documents:
            return []
        if self.index.backend == "smoke_hash_embedding":
            q = _hash_embed(query, dim=self.index.dim or 64)
        else:
            raise RuntimeError(f"Unsupported dense backend at query time: {self.index.backend}")
        scored: list[tuple[float, int]] = []
        for i, emb in enumerate(self.index.embeddings):
            scored.append((cosine(q, emb), i))
        scored.sort(key=lambda x: x[0], reverse=True)
        contexts: list[RetrievedContext] = []
        for score, idx in scored[:top_k]:
            if score <= 0 and contexts:
                continue
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
                        "similarity": round(float(score), 6),
                        "embedding_model": self.index.model_name,
                        "embedding_model_version": self.index.model_version,
                        "index_checksum": self.index.checksum,
                    },
                )
            )
        return contexts


METHOD = "dense_rag"


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm=None) -> dict[str, Any]:
    import time

    from external_baselines.common.llm_client import (
        LLMClient,
        build_llm_client,
        llm_config_summary,
        llm_runtime_snapshot,
    )
    from external_baselines.common.schema import normalize_response_payload, retrieved_context_to_dict
    from external_baselines.common.text_utils import extract_json_object
    from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields

    config = config or {}
    llm = llm or build_llm_client(config)
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    dense_cfg = config.get("dense_rag", {})
    top_k = int(config.get("retrieval", {}).get("top_k", dense_cfg.get("top_k", 5)))
    backend = str(dense_cfg.get("backend", "smoke_hash_embedding"))
    model_name = str(dense_cfg.get("model_name", "smoke-hash-embedding"))
    model_version = dense_cfg.get("model_version", "v0-smoke")
    cache_path = dense_cfg.get("index_path") or str(Path(config.get("paths", {}).get("output_dir", "outputs")) / "dense_index_smoke.json")
    start = time.perf_counter()

    evidence_path = corpus_dir / "evidence_chunks.jsonl"
    if Path(cache_path).exists() and dense_cfg.get("reuse_index", True):
        index = DenseIndex.load(cache_path)
    else:
        index = build_dense_index(
            evidence_path,
            model_name=model_name,
            model_version=str(model_version) if model_version else None,
            backend=backend,
            dim=int(dense_cfg.get("dimension", dense_cfg.get("dim", 64))),
            cache_path=cache_path,
        )
    retriever = DenseRetriever(index)
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
    real_dense = bool(index.build_manifest.get("real_embedding_model_used"))
    output.method_specific = {
        "baseline_name": "Dense embedding RAG baseline",
        "reproduction_class": "enhanced" if real_dense else "smoke_fixture",
        "llm_config_summary": llm_config_summary(config, llm),
        "retrieval_used": True,
        "retrieval_backend": "dense",
        "embedding_backend": index.backend,
        "embedding_model": index.model_name,
        "embedding_model_version": index.model_version,
        "index_checksum": index.checksum,
        "index_build_manifest": index.build_manifest,
        "dense_index_built": True,
        "method_status": "ready" if real_dense else "smoke_fixture_only",
        "no_result": len(contexts) == 0,
        "runtime": llm_runtime_snapshot(llm),
        "parsing_failure": parsing_failure,
        "parsing_status": "failed" if parsing_failure else "ok",
        "structured_safety_fields": "baseline_generated_only",
    }
    result = output.to_dict()
    return maybe_infer_structured_safety_fields(result, config)
