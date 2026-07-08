from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary
from external_baselines.common.schema import normalize_response_payload, retrieved_context_to_dict
from external_baselines.ekell_style.entity_matcher import match_entities
from external_baselines.ekell_style.kg_loader import load_kg
from external_baselines.ekell_style.prompt_chain import run_prompt_chain
from external_baselines.ekell_style.scenario_parser import parse_scenario
from external_baselines.ekell_style.subgraph_retriever import retrieve_subgraph
from external_baselines.evaluation.normalizer import infer_structured_safety_fields

METHOD = "ekell_style"
REPRODUCTION_LABEL = "E-KELL-style paper-faithful pipeline-level reimplementation"


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    config = config or {}
    llm = llm or build_llm_client(config)
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    retrieval_cfg = config.get("retrieval", {})
    ekell_cfg = config.get("ekell_style", {})
    start = time.perf_counter()

    kg = load_kg(corpus_dir)
    parsed = parse_scenario(
        scenario["scenario_text"],
        llm=llm,
        use_llm=bool(config.get("scenario_parser", {}).get("use_llm", False)),
    )
    matched = match_entities(
        scenario["scenario_text"],
        parsed,
        kg.entities,
        top_k=int(retrieval_cfg.get("top_k_entities", 8)),
        min_score=float(retrieval_cfg.get("entity_min_score", 0.08)),
    )
    subgraph = retrieve_subgraph(
        scenario["scenario_text"],
        kg,
        matched,
        top_k_triples=int(retrieval_cfg.get("top_k_triples", 20)),
        top_k_evidence=int(retrieval_cfg.get("top_k_evidence", retrieval_cfg.get("top_k", 6))),
        top_k_relations=int(retrieval_cfg.get("top_k_relations", 10)),
    )
    contexts = [retrieved_context_to_dict(c) for c in subgraph["contexts"]]
    chain = run_prompt_chain(
        scenario_text=scenario["scenario_text"],
        parsed_scenario=parsed,
        contexts=contexts,
        llm=llm,
        temperature=float(config.get("llm", {}).get("temperature", 0.0)),
        max_tokens=int(config.get("llm", {}).get("max_tokens", 1200)),
        max_context_chars=int(retrieval_cfg.get("max_context_chars", 8000)),
        prompt_dir=ekell_cfg.get("prompt_dir", "configs/prompts"),
    )
    final_payload = chain["stage3_final_response"]
    output = normalize_response_payload(final_payload, scenario_id=scenario["scenario_id"], method=METHOD)
    output.retrieved_contexts = contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = chain["raw_outputs"]
    output.method_specific = {
        "baseline_name": "E-KELL-style paper-faithful reimplementation",
        "reproduction_label": REPRODUCTION_LABEL,
        "official_reproduction": False,
        "fidelity_level": "Level 3 data-compatible pipeline-level reproduction when copied KG/evidence inputs are present; not Level 5 official reproduction",
        "llm_config_summary": llm_config_summary(config, llm),
        "kg_asset_counts": kg.counts(),
        "kg_missing_files": kg.missing_files,
        "kg_schema_warnings": kg.schema_warnings[:50],
        "pipeline_trace": [
            "Scenario Input",
            "Scenario Understanding / Parsing",
            "KG Entity Matching",
            "KG Subgraph / Fact Retrieval",
            "Evidence Context Construction",
            "Prompt Chain Reasoning",
            "Final Emergency Decision Support Output",
            "Unified Output Normalization",
        ],
        "scenario_parsing": parsed,
        "parser_fallback_used": bool(parsed.get("parser_fallback_used", False)),
        "matched_entities": matched,
        "retrieved_triples": subgraph["triples"],
        "retrieval_trace": subgraph.get("retrieval_trace", {}),
        "prompt_template_files": chain.get("prompt_template_files", {}),
        "prompt_chain_intermediates": {
            "stage1_situation_understanding": chain["stage1_situation_understanding"],
            "stage2_kg_grounded_decision_reasoning": chain["stage2_kg_grounded_decision_reasoning"],
        },
        "structured_safety_fields": "inferred_from_text",
        "deviations": [
            "Uses copied fire corpus/KG files rather than the original E-KELL emergency KG.",
            "Uses local transparent entity matching and subgraph retrieval because official E-KELL code/data are not integrated.",
            "Uses configurable LLM provider; deterministic heuristic mode is only for smoke tests.",
            "Does not reproduce official expert evaluation or exact paper results.",
        ],
    }
    result = output.to_dict()
    if config.get("normalization", {}).get("infer_structured_safety_fields", True):
        result = infer_structured_safety_fields(result)
    return result
