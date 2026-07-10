from __future__ import annotations

"""Hybrid BM25 + dense retrieval with transparent RRF fusion."""

import time
from pathlib import Path
from typing import Any

from external_baselines.common.llm_client import build_llm_client, llm_config_summary, llm_runtime_snapshot
from external_baselines.common.schema import RetrievedContext, normalize_response_payload, retrieved_context_to_dict
from external_baselines.common.text_utils import extract_json_object, rrf_fuse
from external_baselines.dense_rag.pipeline import DenseIndex, DenseRetriever, build_dense_index
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields
from external_baselines.vanilla_rag.retriever import LexicalRetriever

METHOD = "hybrid_rag"


def _contexts_by_id(contexts: list[RetrievedContext]) -> dict[str, RetrievedContext]:
    return {c.context_id: c for c in contexts}


def hybrid_retrieve(
    query: str,
    *,
    lexical: LexicalRetriever,
    dense: DenseRetriever,
    top_k: int = 5,
    rrf_k: int = 60,
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
    candidate_pool: int = 20,
) -> list[RetrievedContext]:
    lex = lexical.retrieve(query, top_k=candidate_pool)
    den = dense.retrieve(query, top_k=candidate_pool)
    lex_ranked = [(c.context_id, float(c.score or 0.0)) for c in lex]
    den_ranked = [(c.context_id, float(c.score or 0.0)) for c in den]
    fused = rrf_fuse([lex_ranked, den_ranked], k=rrf_k, weights=[lexical_weight, dense_weight])
    by_id = {**_contexts_by_id(den), **_contexts_by_id(lex)}
    out: list[RetrievedContext] = []
    for doc_id, fused_score, components in fused[:top_k]:
        base = by_id.get(doc_id)
        if base is None:
            continue
        meta = dict(base.metadata or {})
        meta.update({
            "retrieval_backend": "hybrid_rrf",
            "fused_score": round(float(fused_score), 6),
            "component_scores": components,
            "bm25_score": components.get("list_0"),
            "dense_score": components.get("list_1"),
        })
        out.append(
            RetrievedContext(
                context_id=base.context_id,
                text=base.text,
                source_id=base.source_id,
                citation=base.citation,
                score=round(float(fused_score), 6),
                metadata=meta,
            )
        )
    return out


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm=None) -> dict[str, Any]:
    config = config or {}
    llm = llm or build_llm_client(config)
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    hybrid_cfg = config.get("hybrid_rag", {})
    dense_cfg = config.get("dense_rag", {})
    top_k = int(config.get("retrieval", {}).get("top_k", hybrid_cfg.get("top_k", 5)))
    start = time.perf_counter()

    evidence_path = corpus_dir / "evidence_chunks.jsonl"
    lexical = LexicalRetriever.from_jsonl(str(evidence_path))
    cache_path = dense_cfg.get("index_path") or str(Path(config.get("paths", {}).get("output_dir", "outputs")) / "dense_index_smoke.json")
    backend = str(dense_cfg.get("backend", "smoke_hash_embedding"))
    if Path(cache_path).exists() and dense_cfg.get("reuse_index", True):
        index = DenseIndex.load(cache_path)
    else:
        index = build_dense_index(
            evidence_path,
            model_name=str(dense_cfg.get("model_name", "smoke-hash-embedding")),
            model_version=str(dense_cfg.get("model_version", "v0-smoke")),
            backend=backend,
            dim=int(dense_cfg.get("dim", 64)),
            cache_path=cache_path,
        )
    dense = DenseRetriever(index)
    contexts_raw = hybrid_retrieve(
        scenario["scenario_text"],
        lexical=lexical,
        dense=dense,
        top_k=top_k,
        rrf_k=int(hybrid_cfg.get("rrf_k", 60)),
        lexical_weight=float(hybrid_cfg.get("lexical_weight", 1.0)),
        dense_weight=float(hybrid_cfg.get("dense_weight", 1.0)),
        candidate_pool=int(hybrid_cfg.get("candidate_pool", 20)),
    )
    contexts = [retrieved_context_to_dict(c) for c in contexts_raw]

    system = (
        "You are reproducing a hybrid BM25+dense RAG emergency decision-support baseline. "
        "Use only retrieved contexts and the scenario. Do not use SAFE modules. Return valid JSON."
    )
    ctx_text = "\n\n".join(
        f"[context_id={c.get('context_id')} source_id={c.get('source_id')} citation={c.get('citation')} "
        f"score={c.get('score')} components={c.get('metadata', {}).get('component_scores')}]\n{c.get('text')}"
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
    parsed = extract_json_object(raw_text)
    payload = parsed or {"situation_summary": raw_text}
    output = normalize_response_payload(payload, scenario_id=scenario["scenario_id"], method=METHOD)
    output.retrieved_contexts = contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {"text": raw_text, "parsed": payload}
    real_dense = bool(index.build_manifest.get("real_embedding_model_used"))
    output.method_specific = {
        "baseline_name": "Hybrid BM25 + dense RRF RAG baseline",
        "reproduction_class": "enhanced" if real_dense else "smoke_fixture",
        "llm_config_summary": llm_config_summary(config, llm),
        "retrieval_used": True,
        "retrieval_backend": "hybrid_rrf",
        "fusion": "rrf",
        "rrf_k": int(hybrid_cfg.get("rrf_k", 60)),
        "lexical_weight": float(hybrid_cfg.get("lexical_weight", 1.0)),
        "dense_weight": float(hybrid_cfg.get("dense_weight", 1.0)),
        "embedding_backend": index.backend,
        "index_checksum": index.checksum,
        "dense_index_built": True,
        "method_status": "ready" if real_dense else "smoke_fixture_only",
        "component_scores_recorded": True,
        "no_result": len(contexts) == 0,
        "runtime": llm_runtime_snapshot(llm),
        "parsing_failure": not bool(parsed),
        "parsing_status": "failed" if not parsed else "ok",
        "structured_safety_fields": "baseline_generated_only",
        "tuning_note": "Fusion weights must be selected on dev only; test config is frozen.",
    }
    result = output.to_dict()
    return maybe_infer_structured_safety_fields(result, config)
