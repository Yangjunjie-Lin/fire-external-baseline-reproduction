from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import prompt_hash
from external_baselines.common.decision_finalize import (
    append_decision_schema,
    finalize_llm_decision,
    use_unified_decision_output,
)
from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary, llm_runtime_snapshot
from external_baselines.common.schema import normalize_response_payload, retrieved_context_to_dict
from external_baselines.common.text_utils import extract_json_object
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields
from external_baselines.vanilla_rag.retriever import LexicalRetriever


METHOD = "bm25_rag"
# Backward-compatible alias retained in runner as vanilla_rag → bm25_rag.


def _format_snapshots(scenario: dict[str, Any]) -> str:
    snaps = scenario.get("dynamic_snapshots") or []
    if not snaps:
        return "(none)"
    return "\n".join(str(s) for s in snaps)


def build_prompt(
    scenario_text: str,
    contexts: list[dict[str, Any]],
    *,
    dynamic_snapshots: str = "(none)",
    unified: bool = True,
) -> tuple[str, str]:
    system = (
        "You are reproducing a vanilla BM25/lexical RAG emergency decision-support baseline. "
        "Use only the retrieved text contexts and the scenario. Do not use KG triples, SAFE modules, "
        "safety checkers, or HITL gates. Preserve evidence IDs in citations when used. Return valid JSON. "
        "Do not invent evidence IDs. Do not assume unstated field conditions. "
        "Do not claim you can execute real operations."
    )
    if contexts:
        ctx_text = "\n\n".join(
            f"[context_id={c.get('context_id')} source_id={c.get('source_id')} citation={c.get('citation')} score={c.get('score')}]\n{c.get('text')}"
            for c in contexts
        )
    else:
        ctx_text = "(none — no lexical matches above threshold; state evidence insufficiency explicitly)"
    if unified:
        user = f"""
Scenario:
{scenario_text}

Dynamic snapshots:
{dynamic_snapshots}

Retrieved contexts (BM25/lexical):
{ctx_text}

Retrieval provenance: deterministic_lexical_bm25. Cite only context_id / citation values listed above.
Recommended actions should reference retrieved evidence IDs when used; do not forge citations.
""".strip()
        user = append_decision_schema(user)
    else:
        user = f"""
Scenario:
{scenario_text}

Retrieved contexts:
{ctx_text}

Return JSON with:
- situation_summary
- key_risks
- recommended_actions
- blocked_or_unsafe_actions
- missing_confirmations
- supporting_evidence (prefer context_id values)
- citations (prefer context_id / source_id / citation values from retrieved contexts)
- final_decision_gate
""".strip()
    return system, user


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    config = config or {}
    llm = llm or build_llm_client(config)
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    top_k = int(config.get("retrieval", {}).get("top_k", 5))
    max_chunk_chars = int(config.get("retrieval", {}).get("max_chunk_chars", 1000))
    start = time.perf_counter()
    unified = use_unified_decision_output(config)

    retriever = LexicalRetriever.from_jsonl(str(corpus_dir / "evidence_chunks.jsonl"), max_chunk_chars=max_chunk_chars)
    contexts = [retrieved_context_to_dict(c) for c in retriever.retrieve(scenario["scenario_text"], top_k=top_k)]
    system, user = build_prompt(
        scenario["scenario_text"],
        contexts,
        dynamic_snapshots=_format_snapshots(scenario),
        unified=unified,
    )
    raw_text = llm.complete(
        system=system,
        user=user,
        temperature=float(config.get("llm", {}).get("temperature", 0.0)),
        max_tokens=int(config.get("llm", {}).get("max_tokens", 1200)),
        top_p=config.get("llm", {}).get("top_p"),
        seed=config.get("llm", {}).get("seed"),
    )
    case_id = str(scenario.get("case_id") or scenario.get("scenario_id"))
    if unified:
        result = finalize_llm_decision(
            case_id=case_id,
            method_id=METHOD,
            raw_text=raw_text,
            config=config,
            llm=llm,
            start=start,
            retrieved_contexts=contexts,
            method_metadata={
                "baseline_name": "Vanilla lexical BM25 RAG baseline",
                "reproduction_class": "baseline",
                "retrieval_used": True,
                "retrieval_backend": "deterministic_lexical_bm25",
                "kg_used": False,
                "top_k": top_k,
                "duplicate_suppression": True,
                "multilingual_tokenization": True,
                "no_result": len(contexts) == 0,
                "prompt_hash": prompt_hash(system, user),
                "structured_safety_fields": "baseline_generated_only",
                "normalizer_policy_injection": False,
            },
            provenance={"retrieval_backend": "deterministic_lexical_bm25", "top_k": top_k},
        )
        assert isinstance(result, dict)
        return result

    parsed = extract_json_object(raw_text)
    payload = parsed or {"situation_summary": raw_text}
    output = normalize_response_payload(payload, scenario_id=scenario["scenario_id"], method=METHOD)
    output.retrieved_contexts = contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {"text": raw_text, "parsed": payload}
    result = output.to_dict()
    result["method_specific"] = {
        "baseline_name": "Vanilla lexical BM25 RAG baseline",
        "reproduction_class": "baseline",
        "llm_config_summary": llm_config_summary(config, llm),
        "retrieval_used": True,
        "retrieval_backend": "deterministic_lexical_bm25",
        "kg_used": False,
        "top_k": top_k,
        "duplicate_suppression": True,
        "multilingual_tokenization": True,
        "no_result": len(contexts) == 0,
        "prompt_hash": prompt_hash(system, user),
        "runtime": llm_runtime_snapshot(llm),
        "parsing_failure": not bool(parsed),
        "parsing_status": "failed" if not parsed else "ok",
        "structured_safety_fields": "baseline_generated_only",
        "normalizer_policy_injection": False,
    }
    return maybe_infer_structured_safety_fields(result, config)


# Keep module path vanilla_rag.pipeline.run_scenario for backward compatibility.
