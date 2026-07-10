"""Shared helpers to finalize method runs into DecisionOutput."""

from __future__ import annotations

import time
from typing import Any

from external_baselines.common.decision_output import (
    DecisionOutput,
    DecisionParseError,
    decision_output_to_legacy_row,
    decision_schema_instruction,
    parse_decision_output,
)
from external_baselines.common.llm_client import llm_config_summary, llm_runtime_snapshot
from external_baselines.common.text_utils import extract_json_object


def use_strict_decision_parse(config: dict[str, Any] | None) -> bool:
    config = config or {}
    if "strict_decision_parse" in config:
        return bool(config.get("strict_decision_parse"))
    if str(config.get("execution_stage") or "").lower() == "formal":
        return True
    return bool(config.get("paper_final", False))


def use_unified_decision_output(config: dict[str, Any] | None) -> bool:
    config = config or {}
    if "unified_decision_output" in config:
        return bool(config.get("unified_decision_output"))
    return True


def finalize_llm_decision(
    *,
    case_id: str,
    method_id: str,
    raw_text: str,
    config: dict[str, Any],
    llm: Any,
    start: float,
    retrieved_contexts: list[dict[str, Any]] | None = None,
    method_metadata: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    as_legacy_row: bool = True,
) -> dict[str, Any] | DecisionOutput:
    """Parse LLM text into DecisionOutput and optionally emit a legacy-compatible row."""
    strict = use_strict_decision_parse(config)
    parsed = extract_json_object(raw_text)
    raw_payload: Any = {"text": raw_text, "parsed": parsed}
    try:
        decision = parse_decision_output(
            parsed if isinstance(parsed, dict) else raw_text,
            case_id=case_id,
            method_id=method_id,
            strict=strict,
            retrieved_contexts=retrieved_contexts,
        )
    except DecisionParseError as exc:
        if strict:
            raise
        decision = DecisionOutput(
            case_id=case_id,
            method_id=method_id,
            parsing_failure=True,
            parsing_errors=[str(exc)],
            raw_output=raw_payload,
            retrieved_contexts=list(retrieved_contexts or []),
        )

    decision.raw_output = raw_payload
    decision.runtime = {
        **(llm_runtime_snapshot(llm) if llm is not None else {}),
        "latency_sec": round(time.perf_counter() - start, 4),
        "latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }
    decision.method_metadata = {
        **dict(method_metadata or {}),
        **dict(decision.method_metadata),
        "llm_config_summary": llm_config_summary(config, llm) if llm is not None else {},
        "unified_decision_output": True,
        "strict_decision_parse": strict,
    }
    decision.provenance = dict(provenance or {})
    if as_legacy_row:
        return decision_output_to_legacy_row(decision)
    return decision


def append_decision_schema(user_prompt: str) -> str:
    return f"{user_prompt.rstrip()}\n\n{decision_schema_instruction()}"
