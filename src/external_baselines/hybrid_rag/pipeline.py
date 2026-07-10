"""Hybrid BM25 + dense retrieval with transparent RRF fusion."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from external_baselines.common.llm_client import build_llm_client, llm_config_summary, llm_runtime_snapshot
from external_baselines.common.schema import RetrievedContext, normalize_response_payload, retrieved_context_to_dict
from external_baselines.common.text_utils import extract_json_object
from external_baselines.dense_rag.pipeline import DenseRetriever, build_dense_index
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields
from external_baselines.retrieval.dense_index import DenseIndexError
from external_baselines.retrieval.embedding_backends import create_embedding_backend, resolve_dimension
from external_baselines.vanilla_rag.retriever import LexicalRetriever

METHOD = "hybrid_rag"


def rrf_fuse_deterministic(
    lexical_ranked: list[tuple[str, float]],
    dense_ranked: list[tuple[str, float]],
    *,
    rrf_k: int = 60,
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Deterministic RRF with explicit ranks and tie-breaking."""
    lex_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(lexical_ranked, start=1)}
    den_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_ranked, start=1)}
    lex_score = {doc_id: score for doc_id, score in lexical_ranked}
    den_score = {doc_id: score for doc_id, score in dense_ranked}
    all_ids = set(lex_rank) | set(den_rank)
    fused: list[tuple[str, float, dict[str, Any]]] = []
    for doc_id in all_ids:
        score = 0.0
        meta: dict[str, Any] = {}
        if doc_id in lex_rank:
            contrib = float(lexical_weight) / (rrf_k + lex_rank[doc_id])
            score += contrib
            meta["lexical_rank"] = lex_rank[doc_id]
            meta["lexical_score"] = float(lex_score[doc_id])
            meta["rrf_lexical"] = contrib
        if doc_id in den_rank:
            contrib = float(dense_weight) / (rrf_k + den_rank[doc_id])
            score += contrib
            meta["dense_rank"] = den_rank[doc_id]
            meta["dense_score"] = float(den_score[doc_id])
            meta["rrf_dense"] = contrib
        meta["rrf_score"] = score
        meta["rrf_k"] = rrf_k
        meta["lexical_weight"] = lexical_weight
        meta["dense_weight"] = dense_weight
        best_rank = min(
            [r for r in (meta.get("lexical_rank"), meta.get("dense_rank")) if r is not None],
            default=10**9,
        )
        fused.append((doc_id, score, {**meta, "_best_rank": best_rank}))
    fused.sort(key=lambda item: (-item[1], item[2]["_best_rank"], item[0]))
    for _, _, meta in fused:
        meta.pop("_best_rank", None)
    return fused


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
    dense_index_checksum: str | None = None,
    corpus_checksum: str | None = None,
) -> list[RetrievedContext]:
    lex = lexical.retrieve(query, top_k=candidate_pool)
    den = dense.retrieve(query, top_k=candidate_pool)
    if not den and dense.index.backend and not str(dense.index.backend).startswith("smoke"):
        raise DenseIndexError("Hybrid dense leg returned no candidates for a real dense index.")
    lex_ranked = [(c.context_id, float(c.score or 0.0)) for c in lex]
    den_ranked = [(c.context_id, float(c.score or 0.0)) for c in den]
    fused = rrf_fuse_deterministic(
        lex_ranked,
        den_ranked,
        rrf_k=rrf_k,
        lexical_weight=lexical_weight,
        dense_weight=dense_weight,
    )
    by_id = {c.context_id: c for c in den}
    by_id.update({c.context_id: c for c in lex})
    out: list[RetrievedContext] = []
    for doc_id, fused_score, components in fused[:top_k]:
        base = by_id.get(doc_id)
        if base is None:
            continue
        meta = dict(base.metadata or {})
        meta.update(components)
        meta.update(
            {
                "retrieval_backend": "hybrid_rrf",
                "rrf_score": round(float(fused_score), 6),
                "dense_index_checksum": dense_index_checksum,
                "corpus_checksum": corpus_checksum,
            }
        )
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
    dense_cfg = dict(config.get("dense_rag") or {})
    # Allow hybrid_rag dense_* fields to populate dense_cfg when nested block absent.
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

    top_k = int(config.get("retrieval", {}).get("top_k", hybrid_cfg.get("top_k", 5)))
    start = time.perf_counter()
    evidence_path = corpus_dir / "evidence_chunks.jsonl"
    lexical = LexicalRetriever.from_jsonl(str(evidence_path))

    backend = str(dense_cfg.get("backend", hybrid_cfg.get("dense_method", "smoke_hash_embedding")))
    model_name = str(dense_cfg.get("model_name", "smoke-hash-embedding"))
    model_version = str(dense_cfg.get("model_version", "v0-smoke"))
    dim = resolve_dimension(dense_cfg, resolve_dimension(hybrid_cfg, 64))
    reject_smoke = bool(dense_cfg.get("reject_smoke", False) or config.get("paper_final", False) or hybrid_cfg.get("reject_smoke", False))
    paper_final = bool(config.get("paper_final", False))
    cache_path = dense_cfg.get("index_path") or hybrid_cfg.get("dense_index_path")
    if not cache_path:
        cache_path = str(Path(config.get("paths", {}).get("output_dir", "outputs")) / "dense_index_smoke.json")

    if reject_smoke and not Path(cache_path).exists() and not Path(cache_path).is_dir():
        # For directory indexes, parent may not exist yet — still hard-fail in real mode if missing.
        if not (Path(cache_path).is_dir() and (Path(cache_path) / "index_manifest.json").is_file()):
            raise DenseIndexError(
                "Hybrid requires an existing Dense index in real/reject_smoke mode; "
                "refusing to silently fall back to BM25-only."
            )

    emb = create_embedding_backend(
        backend,
        model_name=model_name,
        model_version=model_version,
        dimension=dim,
        paper_final=paper_final,
        reject_smoke=reject_smoke,
        model=dense_cfg.get("injected_model"),
    )
    try:
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
            corpus_checksum=config.get("corpus_checksum") or (config.get("paths") or {}).get("corpus_checksum"),
        )
    except Exception as exc:
        if reject_smoke or paper_final:
            raise DenseIndexError(
                f"Hybrid cannot load/build Dense index and will not fall back to BM25-only: {exc}"
            ) from exc
        raise

    dense = DenseRetriever(index, embedding_backend=emb)
    contexts_raw = hybrid_retrieve(
        scenario["scenario_text"],
        lexical=lexical,
        dense=dense,
        top_k=top_k,
        rrf_k=int(hybrid_cfg.get("rrf_k", 60)),
        lexical_weight=float(hybrid_cfg.get("lexical_weight", 1.0)),
        dense_weight=float(hybrid_cfg.get("dense_weight", 1.0)),
        candidate_pool=int(hybrid_cfg.get("candidate_pool", 20)),
        dense_index_checksum=index.checksum,
        corpus_checksum=index.build_manifest.get("corpus_checksum") or index.build_manifest.get("evidence_sha256"),
    )
    contexts = [retrieved_context_to_dict(c) for c in contexts_raw]

    system = (
        "You are reproducing a hybrid BM25+dense RAG emergency decision-support baseline. "
        "Use only retrieved contexts and the scenario. Do not use SAFE modules. Return valid JSON."
    )
    ctx_text = "\n\n".join(
        f"[context_id={c.get('context_id')} source_id={c.get('source_id')} citation={c.get('citation')} "
        f"score={c.get('score')}]\n{c.get('text')}"
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
    real_dense = bool(index.build_manifest.get("actual_embedding_used") or index.build_manifest.get("real_embedding_model_used"))
    output.method_specific = {
        "baseline_name": "Hybrid BM25 + dense RRF RAG baseline",
        "reproduction_class": "controlled_supplemental" if real_dense else "smoke_fixture",
        "llm_config_summary": llm_config_summary(config, llm),
        "retrieval_used": True,
        "retrieval_backend": "hybrid_rrf",
        "fusion": "rrf",
        "rrf_k": int(hybrid_cfg.get("rrf_k", 60)),
        "lexical_weight": float(hybrid_cfg.get("lexical_weight", 1.0)),
        "dense_weight": float(hybrid_cfg.get("dense_weight", 1.0)),
        "embedding_backend": index.backend,
        "embedding_model": index.model_name,
        "embedding_model_version": index.model_version,
        "index_checksum": index.checksum,
        "dense_index_checksum": index.checksum,
        "corpus_checksum": index.build_manifest.get("corpus_checksum") or index.build_manifest.get("evidence_sha256"),
        "dense_index_built": True,
        "actual_embedding_used": real_dense,
        "smoke_fallback_used": not real_dense,
        "method_status": "ready" if real_dense else "smoke_fixture_only",
        "component_scores_recorded": True,
        "no_result": len(contexts) == 0,
        "runtime": llm_runtime_snapshot(llm),
        "parsing_failure": not bool(parsed),
        "parsing_status": "failed" if not parsed else "ok",
        "structured_safety_fields": "baseline_generated_only",
        "tuning_note": "Fusion weights must be selected on dev only; test config is frozen.",
    }
    return maybe_infer_structured_safety_fields(output.to_dict(), config)
