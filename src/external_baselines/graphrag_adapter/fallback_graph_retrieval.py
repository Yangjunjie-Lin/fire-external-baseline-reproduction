from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary
from external_baselines.common.schema import normalize_response_payload, retrieved_context_to_dict
from external_baselines.common.text_utils import extract_json_object
from external_baselines.ekell_style.entity_matcher import match_entities
from external_baselines.ekell_style.kg_loader import load_kg
from external_baselines.ekell_style.scenario_parser import deterministic_parse
from external_baselines.ekell_style.subgraph_retriever import retrieve_subgraph
from external_baselines.evaluation.normalizer import infer_structured_safety_fields

METHOD = "fallback_graph_retrieval"


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None, method: str = METHOD) -> dict[str, Any]:
    config = config or {}
    llm = llm or build_llm_client(config)
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    top_k = int(config.get("retrieval", {}).get("top_k", 5))
    start = time.perf_counter()
    kg = load_kg(corpus_dir)
    parsed = deterministic_parse(scenario["scenario_text"])
    matched = match_entities(scenario["scenario_text"], parsed, kg.entities, top_k=top_k)
    subgraph = retrieve_subgraph(scenario["scenario_text"], kg, matched, top_k_triples=top_k * 2, top_k_evidence=top_k)
    contexts = [retrieved_context_to_dict(c) for c in subgraph["contexts"]]
    ctx_text = "\n\n".join(f"[{c.get('context_id')}] {c.get('text')}" for c in contexts)
    system = "GraphRAG-style fallback baseline: use retrieved graph/text contexts only; no target-project modules. Return JSON."
    user = f"Scenario:\n{scenario['scenario_text']}\n\nGraph/text contexts:\n{ctx_text or '(none)'}"
    raw = llm.complete(system=system, user=user, temperature=0.0, max_tokens=1200)
    payload = extract_json_object(raw) or {"situation_summary": raw}
    output = normalize_response_payload(payload, scenario_id=scenario["scenario_id"], method=method)
    output.retrieved_contexts = contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {"text": raw, "parsed": payload}
    output.method_specific = {"backend": "fallback_graph_retrieval", "reason": "Optional external GraphRAG dependency was unavailable or not configured.", "structured_safety_fields": "inferred_from_text", "llm_config_summary": llm_config_summary(config, llm), "actual_external_package_used": False, "fallback_retrieval_used": True, "indexing_performed": False, "external_repository": "not_applicable_fallback"}
    result = output.to_dict()
    if config.get("normalization", {}).get("infer_structured_safety_fields", True):
        result = infer_structured_safety_fields(result)
    return result
