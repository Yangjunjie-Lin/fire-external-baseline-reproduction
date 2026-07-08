from __future__ import annotations

import time
from typing import Any

from external_baselines.common.llm_client import LLMClient, build_llm_client, llm_config_summary
from external_baselines.common.schema import normalize_response_payload
from external_baselines.common.text_utils import extract_json_object
from external_baselines.evaluation.normalizer import infer_structured_safety_fields


METHOD = "direct_llm"


def build_prompt(scenario_text: str) -> tuple[str, str]:
    system = (
        "You are reproducing a no-retrieval Direct LLM emergency decision-support baseline. "
        "Do not use external retrieval, knowledge graphs, SAFE modules, safety checkers, or HITL gates. "
        "Return only valid JSON compatible with the requested fields."
    )
    user = f"""
Scenario:
{scenario_text}

Return JSON with these keys:
- situation_summary
- key_risks
- recommended_actions
- blocked_or_unsafe_actions
- missing_confirmations
- supporting_evidence
- citations
- final_decision_gate

Because this is the direct LLM baseline, do not cite retrieved evidence.
""".strip()
    return system, user


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    config = config or {}
    llm = llm or build_llm_client(config)
    start = time.perf_counter()
    system, user = build_prompt(scenario["scenario_text"])
    raw_text = llm.complete(
        system=system,
        user=user,
        temperature=float(config.get("llm", {}).get("temperature", 0.0)),
        max_tokens=int(config.get("llm", {}).get("max_tokens", 1200)),
    )
    payload = extract_json_object(raw_text) or {"situation_summary": raw_text}
    output = normalize_response_payload(payload, scenario_id=scenario["scenario_id"], method=METHOD)
    output.latency_sec = round(time.perf_counter() - start, 4)
    output.raw_output = {"text": raw_text, "parsed": payload}
    output.method_specific = {
        "baseline_name": "Direct LLM no-retrieval baseline",
        "llm_config_summary": llm_config_summary(config, llm),
        "retrieval_used": False,
        "kg_used": False,
        "structured_safety_fields": "inferred_from_text",
    }
    result = output.to_dict()
    if config.get("normalization", {}).get("infer_structured_safety_fields", True):
        result = infer_structured_safety_fields(result)
    return result
