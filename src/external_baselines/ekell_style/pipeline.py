from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import prompt_hash, sha256_text
from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary, llm_runtime_snapshot
from external_baselines.common.schema import normalize_response_payload, retrieved_context_to_dict
from external_baselines.ekell_style.entity_matcher import match_entities
from external_baselines.ekell_style.kg_loader import load_kg
from external_baselines.ekell_style.prompt_chain import run_prompt_chain
from external_baselines.ekell_style.scenario_parser import parse_scenario
from external_baselines.ekell_style.subgraph_retriever import retrieve_subgraph
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields

METHOD = "ekell_style_faithful"
# Legacy alias "ekell_style" maps here via runner.
REPRODUCTION_LABEL = "E-KELL-style paper-faithful pipeline-level reimplementation, not official E-KELL reproduction."
REPRODUCTION_CLASS = "faithful"


def _dedupe_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for ctx in contexts:
        key = str(ctx.get("context_id") or "") + "|" + sha256_text(str(ctx.get("text") or ""))[:16]
        if key in seen:
            continue
        seen.add(key)
        out.append(ctx)
    return out


def _graph_paths(triples: list[dict[str, Any]], max_paths: int = 20) -> list[dict[str, Any]]:
    """Simple 1–2 hop path serialization from retrieved triples (paper-faithful transparency)."""
    paths: list[dict[str, Any]] = []
    for t in triples[:max_paths]:
        h = t.get("head") or t.get("subject") or t.get("h")
        r = t.get("relation") or t.get("predicate") or t.get("r")
        tail = t.get("tail") or t.get("object") or t.get("t")
        paths.append({
            "path": [h, r, tail],
            "triple_id": t.get("triple_id"),
            "score": t.get("score"),
            "hops": 1,
            "selection_reason": t.get("selection_reason"),
        })
    return paths


def run_scenario(
    scenario: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    method: str = METHOD,
    reproduction_class: str = REPRODUCTION_CLASS,
    enhanced: bool = False,
) -> dict[str, Any]:
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

    embedding_scorer = None
    if enhanced and ekell_cfg.get("dense_entity_retrieval"):
        # Optional enhanced hook: dense alias scoring via smoke/hash or external scorer.
        from external_baselines.dense_rag.pipeline import _hash_embed, cosine

        def embedding_scorer(a: str, b: str) -> float:  # type: ignore[no-redef]
            return cosine(_hash_embed(a), _hash_embed(b))

    matched = match_entities(
        scenario["scenario_text"],
        parsed,
        kg.entities,
        top_k=int(retrieval_cfg.get("top_k_entities", 8)),
        min_score=float(retrieval_cfg.get("entity_min_score", 0.08)),
        embedding_scorer=embedding_scorer,
    )
    subgraph = retrieve_subgraph(
        scenario["scenario_text"],
        kg,
        matched,
        top_k_triples=int(retrieval_cfg.get("top_k_triples", 20)),
        top_k_evidence=int(retrieval_cfg.get("top_k_evidence", retrieval_cfg.get("top_k", 6))),
        top_k_relations=int(retrieval_cfg.get("top_k_relations", 10)),
    )
    contexts = _dedupe_contexts([retrieved_context_to_dict(c) for c in subgraph["contexts"]])
    # Faithful: stable ordering by score then context_id. Enhanced may re-rank later.
    contexts = sorted(contexts, key=lambda c: (-float(c.get("score") or 0.0), str(c.get("context_id") or "")))

    if enhanced and ekell_cfg.get("hybrid_subgraph_ranking"):
        # Mild enhanced re-rank: boost contexts that cite matched entity ids.
        entity_ids = {str(e.get("entity_id") or e.get("id") or "") for e in matched}
        for ctx in contexts:
            bonus = 0.0
            blob = str(ctx.get("text") or "") + str(ctx.get("metadata") or "")
            for eid in entity_ids:
                if eid and eid in blob:
                    bonus += 0.05
            ctx["score"] = round(float(ctx.get("score") or 0.0) + bonus, 6)
        contexts = sorted(contexts, key=lambda c: (-float(c.get("score") or 0.0), str(c.get("context_id") or "")))

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
    output = normalize_response_payload(final_payload, scenario_id=scenario["scenario_id"], method=method)
    output.retrieved_contexts = contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {
        "stage1": chain["raw_outputs"].get("prompt1"),
        "stage2": chain["raw_outputs"].get("prompt2"),
        "stage3": chain["raw_outputs"].get("prompt3"),
        "parsed_stage1": chain["stage1_situation_understanding"],
        "parsed_stage2": chain["stage2_kg_grounded_decision_reasoning"],
        "parsed_stage3": final_payload,
    }

    prompt_hashes = {
        "stage1": sha256_text(chain["raw_prompts"]["prompt1"]),
        "stage2": sha256_text(chain["raw_prompts"]["prompt2"]),
        "stage3": sha256_text(chain["raw_prompts"]["prompt3"]),
        "system_user_stage3": prompt_hash(
            "ekell-style-prompt-chain",
            chain["raw_prompts"]["prompt3"],
        ),
    }

    deviations = [
        "Uses copied fire corpus/KG files rather than the original E-KELL emergency KG.",
        "Uses local transparent entity matching and subgraph retrieval because official E-KELL code/data are not integrated.",
        "Uses configurable LLM provider; deterministic heuristic mode is only for smoke tests.",
        "Does not reproduce official expert evaluation or exact paper results.",
    ]
    if enhanced:
        deviations.append(
            "Enhanced method_id may enable dense entity scoring / hybrid subgraph ranking; "
            "these must not be reported as paper-faithful E-KELL reproduction."
        )

    evidence_ids = [str(c.get("context_id")) for c in contexts if c.get("context_id")]
    result = output.to_dict()
    result["method_specific"] = {
        "baseline_name": "E-KELL-style enhanced baseline" if enhanced else "E-KELL-style paper-faithful reimplementation",
        "reproduction_label": REPRODUCTION_LABEL,
        "reproduction_class": reproduction_class,
        "official_reproduction": False,
        "fidelity_level": (
            "Level 3 data-compatible pipeline-level reproduction when copied KG/evidence inputs are present; "
            "not Level 5 official reproduction"
        ),
        "llm_config_summary": llm_config_summary(config, llm),
        "kg_asset_counts": kg.counts(),
        "kg_missing_files": kg.missing_files,
        "kg_schema_warnings": kg.schema_warnings[:50],
        "pipeline_trace": [
            "Scenario Input",
            "Situation Understanding / Parsing",
            "Entity Matching",
            "KG Subgraph Retrieval",
            "Evidence Context Construction",
            "Prompt Chain Reasoning",
            "Final Response",
            "Output Normalization",
        ],
        "scenario_parsing": parsed,
        "parsed_scenario": parsed,
        "parser_fallback_used": bool(parsed.get("parser_fallback_used", False)),
        "matched_entities": matched,
        "entity_scores": [{"entity_id": e.get("entity_id"), "score": e.get("score"), "reason": e.get("match_reason")} for e in matched],
        "retrieved_triples": subgraph["triples"],
        "graph_paths": _graph_paths(subgraph.get("triples") or []),
        "evidence_chunks": contexts,
        "retrieval_scores": [c.get("score") for c in contexts],
        "retrieval_trace": subgraph.get("retrieval_trace", {}),
        "prompt_template_files": chain.get("prompt_template_files", {}),
        "prompt_hashes": prompt_hashes,
        "raw_prompts_stored": True,
        "raw_prompts": chain.get("raw_prompts"),
        "stage1_raw_output": chain["raw_outputs"].get("prompt1"),
        "stage2_raw_output": chain["raw_outputs"].get("prompt2"),
        "stage3_raw_output": chain["raw_outputs"].get("prompt3"),
        "prompt_chain_intermediates": {
            "stage1_situation_understanding": chain["stage1_situation_understanding"],
            "stage2_kg_grounded_decision_reasoning": chain["stage2_kg_grounded_decision_reasoning"],
        },
        "context_ids": evidence_ids,
        "evidence_ids_preserved": True,
        "no_evidence": len(contexts) == 0,
        "enhanced_features_enabled": {
            "dense_entity_retrieval": bool(enhanced and ekell_cfg.get("dense_entity_retrieval")),
            "hybrid_subgraph_ranking": bool(enhanced and ekell_cfg.get("hybrid_subgraph_ranking")),
            "reranker": bool(enhanced and ekell_cfg.get("reranker")),
            "self_consistency": bool(enhanced and ekell_cfg.get("self_consistency")),
            "structured_verification": bool(enhanced and ekell_cfg.get("structured_verification")),
        } if enhanced else {},
        "runtime": llm_runtime_snapshot(llm),
        "structured_safety_fields": "baseline_generated_only",
        "normalizer_policy_injection": False,
        "deviations": deviations,
    }
    return maybe_infer_structured_safety_fields(result, config)


def run_scenario_faithful(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    return run_scenario(scenario, config=config, llm=llm, method=METHOD, reproduction_class="faithful", enhanced=False)


def run_scenario_enhanced(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    return run_scenario(
        scenario,
        config=config,
        llm=llm,
        method="ekell_style_enhanced",
        reproduction_class="enhanced",
        enhanced=True,
    )
