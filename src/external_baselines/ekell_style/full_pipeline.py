from __future__ import annotations

"""Complete E-KELL-style pipeline for paper-fidelity and controlled comparison.

Pipeline:
  Scenario
  → Query Understanding
  → Logical Query Decomposition
  → AST Validation
  → Vector KG Retrieval
  → Neighborhood Expansion
  → FOL Execution
  → Stepwise Prompt Chain
  → Evidence-grounded Final Response
  → Trace/Provenance Export

Does not import fire_agent_demo, generic dense_rag/hybrid_rag, or enhanced_pipeline.
"""

import time
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_text
from external_baselines.common.decision_finalize import finalize_llm_decision, use_unified_decision_output
from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary, llm_runtime_snapshot
from external_baselines.common.schema import normalize_response_payload, retrieved_context_to_dict
from external_baselines.ekell_style.logical_query import (
    decompose_query,
    execute_query,
    validate_query,
)
from external_baselines.ekell_style.neighborhood_expander import expand_neighborhood
from external_baselines.ekell_style.scenario_parser import parse_scenario
from external_baselines.ekell_style.stepwise_prompt_chain import run_stepwise_prompt_chain
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields

# Controlled reproduction keeps enhanced hooks off.
CONTROLLED_ENHANCED_HOOKS_OFF = {
    "dense_entity_retrieval": False,
    "hybrid_subgraph_ranking": False,
    "reranker": False,
    "self_consistency": False,
    "structured_verification": False,
}
REPRODUCTION_LABEL = (
    "E-KELL-style paper-faithful pipeline-level reimplementation, "
    "not official E-KELL reproduction."
)


def _seed_entities_from_contexts(contexts: list[dict[str, Any]], matched: list[dict[str, Any]]) -> list[str]:
    seeds: list[str] = []
    for m in matched:
        for key in ("entity_id", "name", "label"):
            if m.get(key):
                seeds.append(str(m[key]))
    for ctx in contexts:
        meta = ctx.get("metadata") or {}
        for key in ("entity_id", "head", "tail", "subject", "object"):
            if meta.get(key):
                seeds.append(str(meta[key]))
        str(ctx.get("text") or "")
        # Prefer explicit IDs already collected; avoid inventing entities from free text.
        if meta.get("triple_id"):
            seeds.append(str(meta["triple_id"]))
    return list(dict.fromkeys(s for s in seeds if s))


def _candidate_universe(kg, seeds: list[str], paths: list[dict[str, Any]]) -> list[str]:
    universe: list[str] = list(seeds)
    for path in paths:
        universe.extend(str(n) for n in (path.get("nodes") or []) if n)
    for ent in kg.entities:
        eid = ent.get("entity_id") or ent.get("id") or ent.get("name")
        if eid:
            universe.append(str(eid))
    # Keep deterministic unique order; cap for negation universe practicality.
    return list(dict.fromkeys(universe))[:500]


def _paths_to_contexts(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        pid = str(path.get("path_id") or sha256_text(str(path))[:12])
        nodes = path.get("nodes") or []
        rels = path.get("relations") or []
        text = " -- ".join(
            f"{nodes[i]} -[{rels[i] if i < len(rels) else '?'}]-> {nodes[i+1]}"
            for i in range(max(0, len(nodes) - 1))
        ) or str(nodes)
        out.append({
            "context_id": pid,
            "text": text,
            "source_id": None,
            "citation": pid,
            "score": None,
            "metadata": {
                "kind": "graph_path",
                "path_id": pid,
                "triple_ids": path.get("triple_ids") or [],
                "source_chunk_ids": path.get("source_chunk_ids") or [],
                "hop_count": path.get("hop_count"),
            },
        })
    return out


def _paper_fidelity_payload(final_step: dict[str, Any], fol_result: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Paper-original-style fields only; no SAFE-style forced gates."""
    parsed = final_step.get("parsed_output") if isinstance(final_step.get("parsed_output"), dict) else {}
    response_text = (
        parsed.get("response")
        or parsed.get("final_response")
        or parsed.get("answer")
        or final_step.get("raw_output")
        or ""
    )
    evidence_ids = [c.get("context_id") for c in contexts if c.get("context_id")]
    return {
        "situation_summary": str(parsed.get("summary") or response_text)[:2000],
        "key_risks": list(parsed.get("risks") or []),
        "recommended_actions": list(parsed.get("actions") or parsed.get("recommended_actions") or []),
        "blocked_or_unsafe_actions": list(parsed.get("blocked_or_unsafe_actions") or []),
        "missing_confirmations": list(parsed.get("missing_information") or parsed.get("missing_confirmations") or []),
        "supporting_evidence": list(parsed.get("evidence_statements") or []),
        "citations": [str(x) for x in (parsed.get("evidence_ids") or evidence_ids[:8])],
        "final_decision_gate": str(parsed.get("final_decision_gate") or "not_applicable_paper_fidelity_output"),
        "full_response": str(response_text),
        "paper_fidelity_fields": {
            "decision_support_response": response_text,
            "referenced_kg_facts": fol_result.get("result_entities") or fol_result.get("entities") or [],
            "referenced_standards": [c.get("citation") for c in contexts if c.get("citation")],
            "reasoning_trace_present": True,
        },
    }


def _controlled_payload(final_step: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Shared-outcome structured fields for FireBench controlled comparison."""
    parsed = final_step.get("parsed_output") if isinstance(final_step.get("parsed_output"), dict) else {}
    response_text = (
        parsed.get("response")
        or parsed.get("final_response")
        or parsed.get("answer")
        or final_step.get("raw_output")
        or ""
    )
    actions = parsed.get("recommended_actions") or parsed.get("actions") or []
    # Preserve action-specific evidence if present; do not attach all global refs.
    norm_actions: list[Any] = []
    for i, a in enumerate(actions):
        if isinstance(a, dict):
            norm_actions.append({
                "action_id": a.get("action_id") or f"action_{i+1}",
                "text": a.get("text") or a.get("action") or str(a),
                "evidence_refs": list(a.get("evidence_refs") or []),
            })
        else:
            norm_actions.append({"action_id": f"action_{i+1}", "text": str(a), "evidence_refs": []})
    return {
        "situation_summary": str(parsed.get("situation_summary") or parsed.get("summary") or response_text)[:2000],
        "key_risks": list(parsed.get("key_risks") or parsed.get("risks") or []),
        "recommended_actions": norm_actions,
        "blocked_or_unsafe_actions": list(parsed.get("blocked_or_unsafe_actions") or parsed.get("blocked_actions") or []),
        "missing_confirmations": list(parsed.get("missing_confirmations") or parsed.get("missing_information") or []),
        "supporting_evidence": list(parsed.get("supporting_evidence") or []),
        "citations": [str(x) for x in (parsed.get("citations") or parsed.get("evidence_ids") or [])],
        "final_decision_gate": str(
            parsed.get("final_decision_gate")
            or ("critical_information_missing_or_requires_human_confirmation"
                if (parsed.get("missing_confirmations") or parsed.get("missing_information"))
                else "baseline_response_without_explicit_gate")
        ),
        "full_response": str(response_text),
        "controlled_output_format": True,
        "paper_original_output_format": False,
    }


def run_ekell_full_pipeline(
    scenario: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    runtime: Any | None = None,
    method: str,
    track: str,
) -> dict[str, Any]:
    """Run the complete E-KELL-style pipeline.

    track: "paper_fidelity" | "controlled_shared_llm"
    """
    config = config or {}
    llm = llm or build_llm_client(config)
    Path(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    ekell_cfg = config.get("ekell_style", {})
    retrieval_cfg = config.get("retrieval", {})
    vector_cfg = config.get("ekell_vector", {}) or ekell_cfg.get("vector", {})
    paper_final = bool(config.get("paper_final", False))
    reject_smoke = bool(vector_cfg.get("reject_smoke", paper_final))
    start = time.perf_counter()

    if runtime is None:
        if paper_final or reject_smoke:
            raise RuntimeError(
                "E-KELL formal/reject_smoke requires a prepared EKELLRuntime; "
                "refusing to rebuild the vector index per case."
            )
        from external_baselines.common.method_runtime import prepare_ekell_runtime

        runtime = prepare_ekell_runtime(config)

    kg = runtime.kg
    retriever = runtime.vector_retriever
    if getattr(runtime, "audit", None) is not None:
        runtime.audit.case_count += 1

    kg_status = {
        "substituted_fire_domain_kg": True,
        "official_ekell_kg": False,
        "paper_reported_triple_count": 2264,
        "local_triple_count": len(kg.triples),
        "note": "Local fire-domain KG snapshot is substituted; not the official E-KELL 2264-triple KG.",
        "index_path": vector_cfg.get("index_path"),
        "index_checksum": (runtime.index_manifest or {}).get("index_checksum"),
        "runtime_reused": True,
    }

    parsed = parse_scenario(
        scenario["scenario_text"],
        llm=llm,
        use_llm=bool(config.get("scenario_parser", {}).get("use_llm", True)),
    )

    # 1) Logical query decomposition
    decomp = decompose_query(
        scenario["scenario_text"],
        llm=llm,
        kg=kg,
        temperature=float(config.get("llm", {}).get("temperature", 0.0)),
        max_tokens=int(config.get("llm", {}).get("max_tokens", 1200)),
    )
    plan = decomp.plan
    raw_decomp = decomp.raw_output
    validation = validate_query(plan, kg) if plan is not None else validate_query(
        {"operation": "projection", "entity": "unknown", "relation": "unknown"}, kg
    )
    degraded = bool(decomp.degraded) or bool(getattr(validation, "degraded", False)) or plan is None
    fallback_reason = decomp.fallback_reason or getattr(validation, "fallback_reason", None)
    if degraded and not fallback_reason:
        fallback_reason = "invalid_or_unknown_logical_ast; continuing with vector+neighborhood retrieval only"

    # 2) Vector KG retrieval (reuse prepared index; never rebuild in formal mode)
    top_k = int(vector_cfg.get("top_k", retrieval_cfg.get("top_k", 8)))
    retrieved = retriever.retrieve(scenario["scenario_text"], top_k=top_k)
    contexts = [retrieved_context_to_dict(c) if hasattr(c, "context_id") else dict(c) for c in retrieved]

    # 3) Neighborhood expansion
    seeds = _seed_entities_from_contexts(contexts, [])
    # Also seed from FOL projection entities if available
    if not degraded and hasattr(validation, "plan") and validation.plan is not None:
        def _collect_entities(node) -> list[str]:
            ents = []
            if getattr(node, "entity", None):
                ents.append(str(node.entity))
            for op in getattr(node, "operands", []) or []:
                ents.extend(_collect_entities(op))
            return ents
        seeds = list(dict.fromkeys(seeds + _collect_entities(validation.plan)))

    seed_nodes = seeds or [
        str(e.get("entity_id") or e.get("name"))
        for e in kg.entities[:3]
        if (e.get("entity_id") or e.get("name"))
    ]
    expansion = expand_neighborhood(
        kg,
        seed_nodes,
        k_hop=int(ekell_cfg.get("neighborhood_k_hop", 1)),
        max_nodes=int(ekell_cfg.get("neighborhood_max_nodes", 50)),
        max_triples=int(ekell_cfg.get("neighborhood_max_triples", 80)),
        relation_whitelist=ekell_cfg.get("relation_whitelist"),
    )
    paths = list(expansion.get("paths") or []) if isinstance(expansion, dict) else []
    path_contexts = _paths_to_contexts(paths)
    all_contexts = contexts + path_contexts

    # 4) FOL execution (deterministic)
    fol_result: dict[str, Any] = {"degraded": degraded, "fallback_reason": fallback_reason, "results": []}
    exec_plan = plan if plan is not None else getattr(validation, "plan", None)
    if not degraded and exec_plan is not None:
        universe = _candidate_universe(kg, seeds, paths)
        exec_out = execute_query(exec_plan, kg, candidate_universe=universe)
        fol_result = exec_out.to_dict() if hasattr(exec_out, "to_dict") else dict(exec_out)

    # 5) Stepwise logical prompt chain
    if track == "controlled_shared_llm":
        prompt_dir = ekell_cfg.get("prompt_dir", "configs/prompts/controlled")
    else:
        prompt_dir = ekell_cfg.get("prompt_dir", "configs/prompts/paper_fidelity")
    fallback_ast = {
        "operation": "projection",
        "entity": seed_nodes[0] if seed_nodes else "unknown",
        "relation": "related_to",
    }
    chain_ast = exec_plan if (not degraded and exec_plan is not None) else fallback_ast
    chain = run_stepwise_prompt_chain(
        validated_ast=chain_ast,
        kg_contexts=all_contexts,
        kg_paths=paths,
        query=scenario["scenario_text"],
        candidate_universe=_candidate_universe(kg, seeds, paths),
        llm=llm,
        prompt_dir=prompt_dir,
        fol_execution=fol_result,
    )
    if degraded:
        chain = dict(chain)
        chain["degraded"] = True
        chain["fallback_reason"] = fallback_reason

    final_step = chain.get("final") or {}
    unified = use_unified_decision_output(config) and track == "controlled_shared_llm"
    case_id = str(scenario.get("case_id") or scenario.get("scenario_id"))

    method_specific_base = {
        "baseline_name": f"E-KELL-style {track}",
        "reproduction_label": REPRODUCTION_LABEL,
        "reproduction_class": "paper_fidelity" if track == "paper_fidelity" else "controlled_shared_llm",
        "track": track,
        "official_reproduction": False,
        "paper_original_output_format": track == "paper_fidelity",
        "controlled_output_format": track == "controlled_shared_llm",
        "enhanced_hooks": dict(CONTROLLED_ENHANCED_HOOKS_OFF),
        "pipeline_trace": [
            "Scenario Input",
            "Query Understanding",
            "Logical Query Decomposition",
            "AST Validation",
            "Vector KG Retrieval",
            "Neighborhood Expansion",
            "FOL Execution",
            "Stepwise Prompt Chain",
            "Evidence-grounded Final Response",
            "Trace/Provenance Export",
        ],
        "scenario_parsing": parsed,
        "logical_decomposition": {
            "raw": raw_decomp,
            "plan": plan.to_dict() if hasattr(plan, "to_dict") else plan,
            "validation_valid": bool(getattr(validation, "valid", False)),
            "validation_errors": list(getattr(validation, "errors", []) or []),
            "degraded": degraded,
            "fallback_reason": fallback_reason,
        },
        "vector_retrieval": {
            "backend": (runtime.index_manifest or {}).get("backend")
            or getattr(runtime.embedding_backend, "backend", None),
            "actual_embedding_used": bool(
                (runtime.index_manifest or {}).get("actual_embedding_used")
                or getattr(runtime.embedding_backend, "actual_embedding_used", False)
            ),
            "smoke_fallback_used": bool(
                (runtime.index_manifest or {}).get("smoke_fallback_used")
                or getattr(runtime.embedding_backend, "smoke_fallback_used", True)
            ),
            "index_metadata": getattr(retriever.index, "metadata", {}),
            "index_checksum": (runtime.index_manifest or {}).get("index_checksum"),
            "top_k": top_k,
            "n_retrieved": len(contexts),
            "runtime_reuse": runtime.audit.to_dict() if getattr(runtime, "audit", None) else None,
        },
        "neighborhood_expansion": expansion if isinstance(expansion, dict) else {"paths": paths},
        "fol_execution": fol_result,
        "stepwise_prompt_chain": chain,
        "graph_paths": paths,
        "kg_status": kg_status,
        "structured_safety_fields": "baseline_generated_only",
        "normalizer_policy_injection": False,
        "deviations": [
            "Uses substituted fire-domain KG, not official E-KELL 2264-triple KG.",
            "Prompt templates reconstruct Appendix A structure without claiming official verbatim prompts.",
            "ChatGLM-6B paper-fidelity model run requires user hardware; smoke/heuristic is not paper fidelity.",
            "Vector smoke_hash backend is test-only; formal runs require actual embedding backend.",
        ],
    }

    if unified:
        raw_final = final_step.get("raw_output") or ""
        # Prefer parsed decision+response object when present.
        parsed_final = final_step.get("parsed_output")
        if isinstance(parsed_final, dict) and ("decision" in parsed_final or "response" in parsed_final):
            import json as _json

            raw_for_parse = _json.dumps(parsed_final, ensure_ascii=False)
        else:
            raw_for_parse = str(raw_final)
        result = finalize_llm_decision(
            case_id=case_id,
            method_id=method,
            raw_text=raw_for_parse,
            config=config,
            llm=llm,
            start=start,
            retrieved_contexts=all_contexts,
            method_metadata=method_specific_base,
            provenance={
                "track": track,
                "allowed_evidence_ids": chain.get("allowed_evidence_ids"),
                "logical_ast": chain.get("validated_ast"),
            },
        )
        assert isinstance(result, dict)
        result["raw_output"] = {
            "decomposition_raw": raw_decomp,
            "stepwise_chain": chain,
            "final_stage_text": final_step.get("raw_output"),
            "parsed": parsed_final,
        }
        return result

    if track == "paper_fidelity":
        payload = _paper_fidelity_payload(final_step, fol_result, all_contexts)
        paper_original_output_format = True
        controlled_output_format = False
    else:
        payload = _controlled_payload(final_step, all_contexts)
        paper_original_output_format = False
        controlled_output_format = True

    output = normalize_response_payload(payload, scenario_id=scenario["scenario_id"], method=method)
    output.retrieved_contexts = all_contexts
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {
        "decomposition_raw": raw_decomp,
        "stepwise_chain": chain,
        "final_stage_text": final_step.get("raw_output"),
        "full_response": payload.get("full_response"),
        "parsed": {k: v for k, v in payload.items() if k != "paper_fidelity_fields"},
    }
    result = output.to_dict()
    if payload.get("full_response"):
        result["full_response"] = payload["full_response"]
    result["method_specific"] = {
        **method_specific_base,
        "paper_original_output_format": paper_original_output_format,
        "controlled_output_format": controlled_output_format,
        "llm_config_summary": llm_config_summary(config, llm),
        "runtime": llm_runtime_snapshot(llm),
    }
    # Never invent safety via normalizer for paper fidelity; controlled also defaults off.
    return maybe_infer_structured_safety_fields(result, config)


def run_paper_fidelity(
    scenario: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    return run_ekell_full_pipeline(
        scenario,
        config=config,
        llm=llm,
        runtime=runtime,
        method="ekell_style_paper_fidelity",
        track="paper_fidelity",
    )


def run_controlled_shared_llm(
    scenario: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    return run_ekell_full_pipeline(
        scenario,
        config=config,
        llm=llm,
        runtime=runtime,
        method="ekell_style_controlled_shared_llm",
        track="controlled_shared_llm",
    )
