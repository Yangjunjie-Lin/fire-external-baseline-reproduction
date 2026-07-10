from __future__ import annotations

"""E-KELL-style ENHANCED pipeline (supplemental / extended baseline only).

Must never replace ekell_style_faithful in the main paper table.
"""

import time
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import prompt_hash, sha256_text
from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary, llm_runtime_snapshot
from external_baselines.common.schema import normalize_response_payload, retrieved_context_to_dict
from external_baselines.dense_rag.pipeline import _hash_embed, cosine
from external_baselines.ekell_style.entity_matcher import match_entities
from external_baselines.ekell_style.kg_loader import load_kg
from external_baselines.ekell_style.pipeline import _dedupe_contexts, _graph_paths
from external_baselines.ekell_style.prompt_chain import run_prompt_chain
from external_baselines.ekell_style.scenario_parser import parse_scenario
from external_baselines.ekell_style.subgraph_retriever import retrieve_subgraph
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields

METHOD = "ekell_style_enhanced"
REPRODUCTION_LABEL = "E-KELL-style enhanced baseline (supplemental); not paper-faithful / not official E-KELL."


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    config = config or {}
    llm = llm or build_llm_client(config)
    corpus_dir = Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    retrieval_cfg = config.get("retrieval", {})
    ekell_cfg = config.get("ekell_style", {})
    start = time.perf_counter()

    def embedding_scorer(a: str, b: str) -> float:
        return cosine(_hash_embed(a), _hash_embed(b))

    use_dense_entity = bool(ekell_cfg.get("dense_entity_retrieval", True))
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
        top_k=int(retrieval_cfg.get("top_k_entities", 12)),
        min_score=float(retrieval_cfg.get("entity_min_score", 0.05)),
        embedding_scorer=embedding_scorer if use_dense_entity else None,
    )
    subgraph = retrieve_subgraph(
        scenario["scenario_text"],
        kg,
        matched,
        top_k_triples=int(retrieval_cfg.get("top_k_triples", 30)),
        top_k_evidence=int(retrieval_cfg.get("top_k_evidence", retrieval_cfg.get("top_k", 8))),
        top_k_relations=int(retrieval_cfg.get("top_k_relations", 12)),
    )
    contexts = _dedupe_contexts([retrieved_context_to_dict(c) for c in subgraph["contexts"]])
    if ekell_cfg.get("hybrid_subgraph_ranking", True):
        entity_ids = {str(e.get("entity_id") or e.get("id") or "") for e in matched}
        for ctx in contexts:
            bonus = 0.0
            blob = str(ctx.get("text") or "") + str(ctx.get("metadata") or "")
            for eid in entity_ids:
                if eid and eid in blob:
                    bonus += 0.05
            ctx["score"] = round(float(ctx.get("score") or 0.0) + bonus, 6)
    contexts = sorted(contexts, key=lambda c: (-float(c.get("score") or 0.0), str(c.get("context_id") or "")))

    # Enhanced keeps the legacy 3-stage prompts; do not inherit paper_fidelity stepwise dir.
    prompt_dir = "configs/prompts"
    legacy = ekell_cfg.get("legacy_prompt_dir")
    if isinstance(legacy, str) and legacy.strip():
        candidate = Path(legacy.strip())
        if (candidate / "ekell_stage1_situation_understanding.txt").is_file():
            prompt_dir = str(candidate)
    else:
        inherited = ekell_cfg.get("prompt_dir")
        if isinstance(inherited, str) and inherited.strip():
            candidate = Path(inherited.strip())
            if (candidate / "ekell_stage1_situation_understanding.txt").is_file():
                prompt_dir = str(candidate)
    chain = run_prompt_chain(
        scenario_text=scenario["scenario_text"],
        parsed_scenario=parsed,
        contexts=contexts,
        llm=llm,
        temperature=float(config.get("llm", {}).get("temperature", 0.0)),
        max_tokens=int(config.get("llm", {}).get("max_tokens", 1400)),
        max_context_chars=int(retrieval_cfg.get("max_context_chars", 10000)),
        prompt_dir=prompt_dir,
    )
    final_payload = chain["stage3_final_response"]
    output = normalize_response_payload(final_payload, scenario_id=scenario["scenario_id"], method=METHOD)
    output.retrieved_contexts = contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {
        "stage1": chain["raw_outputs"].get("prompt1"),
        "stage2": chain["raw_outputs"].get("prompt2"),
        "stage3": chain["raw_outputs"].get("prompt3"),
        "parsed_stage3": final_payload,
    }
    result = output.to_dict()
    result["method_specific"] = {
        "baseline_name": "E-KELL-style enhanced baseline",
        "reproduction_label": REPRODUCTION_LABEL,
        "reproduction_class": "enhanced",
        "paper_table_role": "supplemental_extended",
        "official_reproduction": False,
        "llm_config_summary": llm_config_summary(config, llm),
        "matched_entities": matched,
        "retrieved_triples": subgraph["triples"],
        "graph_paths": _graph_paths(subgraph.get("triples") or []),
        "prompt_hashes": {
            "stage1": sha256_text(chain["raw_prompts"]["prompt1"]),
            "stage2": sha256_text(chain["raw_prompts"]["prompt2"]),
            "stage3": sha256_text(chain["raw_prompts"]["prompt3"]),
            "system_user_stage3": prompt_hash("ekell-style-enhanced", chain["raw_prompts"]["prompt3"]),
        },
        "enhanced_features_enabled": {
            "dense_entity_retrieval": use_dense_entity,
            "hybrid_subgraph_ranking": bool(ekell_cfg.get("hybrid_subgraph_ranking", True)),
            "reranker": bool(ekell_cfg.get("reranker", False)),
            "self_consistency": bool(ekell_cfg.get("self_consistency", False)),
            "structured_verification": bool(ekell_cfg.get("structured_verification", False)),
        },
        "runtime": llm_runtime_snapshot(llm),
        "structured_safety_fields": "baseline_generated_only",
        "normalizer_policy_injection": False,
        "deviations": [
            "Supplemental enhanced baseline; must not be reported as paper-faithful E-KELL reproduction.",
            "May use dense entity scoring and hybrid subgraph ranking.",
        ],
    }
    return maybe_infer_structured_safety_fields(result, config)


run_scenario_enhanced = run_scenario
