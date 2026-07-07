from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from external_baselines.common.llm_client import LLMClient
from external_baselines.common.schema import retrieved_context_to_dict
from external_baselines.common.text_utils import extract_json_object

DEFAULT_PROMPT_DIR = Path("configs/prompts")
PROMPT_FILES = {"stage1": "ekell_stage1_situation_understanding.txt", "stage2": "ekell_stage2_kg_grounded_decision_reasoning.txt", "stage3": "ekell_stage3_final_response.txt"}


def load_prompt_template(name: str, prompt_dir: str | Path = DEFAULT_PROMPT_DIR) -> str:
    path = Path(prompt_dir) / PROMPT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return path.read_text(encoding="utf-8")


def _context_block(contexts: list[dict[str, Any]], max_chars: int = 8000) -> str:
    parts: list[str] = []
    total = 0
    for ctx in contexts:
        item = f"[context_id={ctx.get('context_id')} kind={ctx.get('metadata', {}).get('kind')} source_id={ctx.get('source_id')} citation={ctx.get('citation')} score={ctx.get('score')}]\n{ctx.get('text')}"
        if total + len(item) > max_chars:
            break
        parts.append(item)
        total += len(item)
    return "\n\n".join(parts) if parts else "(none)"


def _ensure_dict_schema(value: Any, schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    result = dict(value)
    for key, default in schema.items():
        if key not in result:
            result[key] = [] if isinstance(default, list) else default
        elif isinstance(default, list) and not isinstance(result[key], list):
            result[key] = [result[key]] if result[key] not in (None, "") else []
        elif isinstance(default, str) and result[key] is None:
            result[key] = ""
    return result


def _safe_json_load(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    parsed = extract_json_object(raw)
    return _ensure_dict_schema(parsed, schema)


def run_prompt_chain(*, scenario_text: str, parsed_scenario: dict[str, Any], contexts: list[Any], llm: LLMClient, temperature: float = 0.0, max_tokens: int = 1200, max_context_chars: int = 8000, prompt_dir: str | Path = DEFAULT_PROMPT_DIR) -> dict[str, Any]:
    ctx_dicts = [retrieved_context_to_dict(c) for c in contexts]
    ctx_block = _context_block(ctx_dicts, max_context_chars)
    system = "You are reproducing an independent E-KELL-style emergency KG + LLM prompt-chain baseline. Reason over supplied KG/evidence contexts. Return valid JSON. Do not import, call, or emulate fire-agent-demo SAFE modules."
    stage1_schema = {"emergency_type": "", "involved_entities": [], "hazards": [], "emergency_stage": "", "missing_information": [], "evidence_used": []}
    stage2_schema = {"reasoning_summary": "", "candidate_actions": [], "deferred_or_unsupported_actions": [], "missing_information": [], "evidence_links": []}
    stage3_schema = {"situation_summary": "", "key_risks": [], "recommended_actions": [], "blocked_or_unsafe_actions": [], "missing_confirmations": [], "supporting_evidence": [], "citations": [], "final_decision_gate": ""}

    prompt1 = load_prompt_template("stage1", prompt_dir).replace("{scenario_text}", scenario_text).replace("{parsed_scenario_json}", json.dumps(parsed_scenario, ensure_ascii=False, indent=2)).replace("{context_block}", ctx_block)
    raw1 = llm.complete(system=system, user=prompt1, temperature=temperature, max_tokens=max_tokens)
    stage1 = _safe_json_load(raw1, stage1_schema)
    prompt2 = load_prompt_template("stage2", prompt_dir).replace("{scenario_text}", scenario_text).replace("{stage1_json}", json.dumps(stage1, ensure_ascii=False, indent=2)).replace("{context_block}", ctx_block)
    raw2 = llm.complete(system=system, user=prompt2, temperature=temperature, max_tokens=max_tokens)
    stage2 = _safe_json_load(raw2, stage2_schema)
    prompt3 = load_prompt_template("stage3", prompt_dir).replace("{scenario_text}", scenario_text).replace("{stage1_json}", json.dumps(stage1, ensure_ascii=False, indent=2)).replace("{stage2_json}", json.dumps(stage2, ensure_ascii=False, indent=2)).replace("{context_block}", ctx_block)
    raw3 = llm.complete(system=system, user=prompt3, temperature=temperature, max_tokens=max_tokens)
    final = _safe_json_load(raw3, stage3_schema)
    return {"stage1_situation_understanding": stage1, "stage2_kg_grounded_decision_reasoning": stage2, "stage3_final_response": final, "raw_prompts": {"prompt1": prompt1, "prompt2": prompt2, "prompt3": prompt3}, "raw_outputs": {"prompt1": raw1, "prompt2": raw2, "prompt3": raw3}, "prompt_template_files": PROMPT_FILES}
