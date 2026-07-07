from __future__ import annotations

import json
from typing import Any

from external_baselines.common.llm_client import LLMClient
from external_baselines.common.schema import retrieved_context_to_dict
from external_baselines.common.text_utils import extract_json_object


def _context_block(contexts: list[dict[str, Any]], max_chars: int = 8000) -> str:
    parts: list[str] = []
    total = 0
    for ctx in contexts:
        item = (
            f"[context_id={ctx.get('context_id')} kind={ctx.get('metadata', {}).get('kind')} "
            f"source_id={ctx.get('source_id')} citation={ctx.get('citation')} score={ctx.get('score')}]\n{ctx.get('text')}"
        )
        if total + len(item) > max_chars:
            break
        parts.append(item)
        total += len(item)
    return "\n\n".join(parts)


def run_prompt_chain(
    *,
    scenario_text: str,
    parsed_scenario: dict[str, Any],
    contexts: list[Any],
    llm: LLMClient,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    max_context_chars: int = 8000,
) -> dict[str, Any]:
    ctx_dicts = [retrieved_context_to_dict(c) for c in contexts]
    ctx_block = _context_block(ctx_dicts, max_context_chars)

    system = (
        "You are reproducing the core E-KELL-style KG + LLM prompt-chain baseline. "
        "Reason over the supplied emergency scenario and retrieved KG facts. "
        "Do not use SAFE-Router, Safety Checker, Dynamic REG, HITL Gate, or hidden target-project modules."
    )

    prompt1 = f"""
Prompt 1: Situation Understanding

Scenario:
{scenario_text}

Parsed scenario candidate:
{json.dumps(parsed_scenario, ensure_ascii=False, indent=2)}

Retrieved KG facts and evidence:
{ctx_block or '(none)'}

Identify emergency type, involved entities, hazards, stage, and missing information. Return JSON.
""".strip()
    raw1 = llm.complete(system=system, user=prompt1, temperature=temperature, max_tokens=max_tokens)
    stage1 = extract_json_object(raw1) or {"analysis_text": raw1}

    prompt2 = f"""
Prompt 2: KG-grounded Decision Reasoning

Scenario:
{scenario_text}

Situation understanding:
{json.dumps(stage1, ensure_ascii=False, indent=2)}

Retrieved KG facts and evidence:
{ctx_block or '(none)'}

Based only on the situation understanding and retrieved KG facts, propose emergency response actions. Do not introduce unsupported facts. Return JSON.
""".strip()
    raw2 = llm.complete(system=system, user=prompt2, temperature=temperature, max_tokens=max_tokens)
    stage2 = extract_json_object(raw2) or {"reasoning_text": raw2}

    prompt3 = f"""
Prompt 3: Final Emergency Response

Scenario:
{scenario_text}

Situation understanding:
{json.dumps(stage1, ensure_ascii=False, indent=2)}

KG-grounded decision reasoning:
{json.dumps(stage2, ensure_ascii=False, indent=2)}

Retrieved KG facts and evidence:
{ctx_block or '(none)'}

Generate a concise emergency decision support response with:
1. situation_summary
2. key_risks
3. recommended_actions
4. blocked_or_unsafe_actions when explicitly supported by the reasoning/evidence
5. missing_confirmations
6. supporting_evidence
7. citations
8. final_decision_gate if inferable from the response text

Return valid JSON only.
""".strip()
    raw3 = llm.complete(system=system, user=prompt3, temperature=temperature, max_tokens=max_tokens)
    final = extract_json_object(raw3) or {"situation_summary": raw3}

    return {
        "stage1_situation_understanding": stage1,
        "stage2_kg_grounded_decision_reasoning": stage2,
        "stage3_final_response": final,
        "raw_prompts": {"prompt1": prompt1, "prompt2": prompt2, "prompt3": prompt3},
        "raw_outputs": {"prompt1": raw1, "prompt2": raw2, "prompt3": raw3},
    }
