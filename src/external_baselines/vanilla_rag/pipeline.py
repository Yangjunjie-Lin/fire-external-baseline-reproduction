from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary
from external_baselines.common.schema import normalize_response_payload, retrieved_context_to_dict
from external_baselines.common.text_utils import extract_json_object
from external_baselines.evaluation.normalizer import infer_structured_safety_fields
from external_baselines.vanilla_rag.retriever import LexicalRetriever

METHOD = "vanilla_rag"


def build_prompt(scenario_text: str, contexts: list[dict[str, Any]]) -> tuple[str, str]:
    system = "You are reproducing a vanilla text-RAG emergency decision-support baseline. Use only retrieved text contexts and the scenario. Do not use KG triples, target-project modules, or gates. Return valid JSON."
    ctx_text = "\n\n".join(f"[context_id={c.get('context_id')} source_id={c.get('source_id')} citation={c.get('citation')} score={c.get('score')}]\n{c.get('text')}" for c in contexts)
    user = f"""
Scenario:
{scenario_text}

Retrieved contexts:
{ctx_text or '(none)'}

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
    return system, user


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    config = config or {}
    llm = llm or build_llm_client(config)
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    top_k = int(config.get("retrieval", {}).get("top_k", 5))
    start = time.perf_counter()
    retriever = LexicalRetriever.from_jsonl(str(corpus_dir / "evidence_chunks.jsonl"))
    contexts = [retrieved_context_to_dict(c) for c in retriever.retrieve(scenario["scenario_text"], top_k=top_k)]
    system, user = build_prompt(scenario["scenario_text"], contexts)
    raw_text = llm.complete(system=system, user=user, temperature=float(config.get("llm", {}).get("temperature", 0.0)), max_tokens=int(config.get("llm", {}).get("max_tokens", 1200)))
    payload = extract_json_object(raw_text) or {"situation_summary": raw_text}
    output = normalize_response_payload(payload, scenario_id=scenario["scenario_id"], method=METHOD)
    output.retrieved_contexts = contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {"text": raw_text, "parsed": payload}
    output.method_specific = {"baseline_name": "Vanilla lexical RAG baseline", "llm_config_summary": llm_config_summary(config, llm), "retrieval_used": True, "retrieval_backend": "deterministic_lexical_bm25", "kg_used": False, "structured_safety_fields": "inferred_from_text"}
    result = output.to_dict()
    if config.get("normalization", {}).get("infer_structured_safety_fields", True):
        result = infer_structured_safety_fields(result)
    return result
