from __future__ import annotations

import time
from typing import Any

from external_baselines.common.checksums import prompt_hash
from external_baselines.common.decision_finalize import (
    append_decision_schema,
    finalize_llm_decision,
    use_unified_decision_output,
)
from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary, llm_runtime_snapshot
from external_baselines.common.schema import normalize_response_payload
from external_baselines.common.text_utils import extract_json_object
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields

METHOD = "direct_llm"


def _format_snapshots(scenario: dict[str, Any]) -> str:
    snaps = scenario.get("dynamic_snapshots") or []
    if not snaps:
        return "(none)"
    return "\n".join(str(s) for s in snaps)


def build_prompt(scenario_text: str, *, dynamic_snapshots: str = "(none)", unified: bool = True) -> tuple[str, str]:
    system = (
        "You are an emergency decision-support assistant for a Direct LLM baseline "
        "(no retrieval, no knowledge graph, no external safety modules). "
        "Analyze the scenario using only the provided text. "
        "Return ONLY valid JSON with the exact keys requested. "
        "If evidence is insufficient, say so in missing_confirmations and set an appropriate final_decision_gate. "
        "Do not invent citations. Do not authorize real-world execution. "
        "Do not assume unstated field conditions. Do not claim you can execute real operations."
    )
    if unified:
        user = f"""
Scenario:
{scenario_text}

Dynamic snapshots:
{dynamic_snapshots}

This is the Direct LLM baseline: no retrieved evidence is available. Leave citations empty.
Decide only from the scenario and snapshots above.
""".strip()
        user = append_decision_schema(user)
    else:
        user = f"""
Scenario:
{scenario_text}

Return a single JSON object with these keys:
- situation_summary (string)
- key_risks (array of strings) — risk signals you identify
- risk_level (string|null) — optional coarse level if you can justify it from the text; otherwise null
- recommended_actions (array of strings) — concrete recommended actions
- blocked_or_unsafe_actions (array of strings) — actions that should be deferred/blocked based on the scenario text alone
- missing_confirmations (array of strings) — information that must be confirmed before acting
- supporting_evidence (array of strings) — leave empty or note that no retrieval was used
- citations (array of strings) — leave empty for Direct LLM
- human_review_required (boolean)
- final_decision_gate (string) — e.g. baseline_response_without_explicit_gate | critical_information_missing_or_requires_human_confirmation

Because this is the Direct LLM baseline, do not cite retrieved evidence.
""".strip()
    return system, user


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    config = config or {}
    llm = llm or build_llm_client(config)
    start = time.perf_counter()
    unified = use_unified_decision_output(config)
    system, user = build_prompt(
        scenario["scenario_text"],
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
            retrieved_contexts=[],
            method_metadata={
                "baseline_name": "Direct LLM no-retrieval baseline",
                "reproduction_class": "baseline",
                "retrieval_used": False,
                "kg_used": False,
                "corpus_accessed": False,
                "dynamic_state_consumed": bool(scenario.get("dynamic_snapshots")),
                "dynamic_state_support": "passthrough" if scenario.get("dynamic_snapshots") else "unsupported",
                "prompt_hash": prompt_hash(system, user),
                "structured_safety_fields": "baseline_generated_only",
                "normalizer_policy_injection": False,
                "claim_label": "canonical no-retrieval LLM baseline",
            },
            provenance={"prompt_hash": prompt_hash(system, user)},
        )
        assert isinstance(result, dict)
        return result

    parsed = extract_json_object(raw_text)
    payload = parsed or {"situation_summary": raw_text}
    parsing_failure = not bool(parsed)
    output = normalize_response_payload(payload, scenario_id=scenario["scenario_id"], method=METHOD)
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {"text": raw_text, "parsed": payload}
    result = output.to_dict()
    if "risk_level" in payload:
        result["risk_level"] = payload.get("risk_level")
    if "human_review_required" in payload:
        result["human_review_required"] = bool(payload.get("human_review_required"))
    result["method_specific"] = {
        "baseline_name": "Direct LLM no-retrieval baseline",
        "reproduction_class": "baseline",
        "llm_config_summary": llm_config_summary(config, llm),
        "retrieval_used": False,
        "kg_used": False,
        "corpus_accessed": False,
        "dynamic_state_consumed": False,
        "dynamic_state_support": "unsupported",
        "prompt_hash": prompt_hash(system, user),
        "runtime": llm_runtime_snapshot(llm),
        "parsing_failure": parsing_failure,
        "parsing_status": "failed" if parsing_failure else "ok",
        "structured_safety_fields": "baseline_generated_only",
        "normalizer_policy_injection": False,
        "claim_label": "canonical no-retrieval LLM baseline",
    }
    return maybe_infer_structured_safety_fields(result, config)
