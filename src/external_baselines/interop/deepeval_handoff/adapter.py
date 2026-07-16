"""Read-only firebench-interop-v1 to external prediction adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from external_baselines.interop.deepeval_handoff.constants import DIRECT_METHOD, FORMAL_RAG_METHODS
from external_baselines.interop.deepeval_handoff.context_normalizer import (
    ContextNormalizationError,
    normalize_retrieval_contexts,
)
from external_baselines.method_registry import get_method


class HandoffAdaptationError(ValueError):
    """Raised when a prediction cannot be exported without invention."""


def _comparison_level(method_id: str, contexts_present: bool) -> tuple[str, bool]:
    if method_id == DIRECT_METHOD:
        return "output_only", False
    if method_id in FORMAL_RAG_METHODS:
        return "output_and_rag", True
    try:
        registry = get_method(method_id)
    except KeyError as exc:
        raise HandoffAdaptationError(f"unknown_method:{method_id}") from exc
    retrieval_expected = bool(registry.get("requires_corpus")) and not bool(registry.get("fallback_only"))
    return ("output_and_rag" if contexts_present else "output_only"), retrieval_expected


def adapt_firebench_interop_to_external_prediction(
    record: dict[str, Any],
    *,
    top_k: int = 5,
    source_artifacts: dict[str, Any] | None = None,
    formal: bool = True,
    allow_synthetic_actual_output: bool = False,
) -> dict[str, Any]:
    """Adapt one completed prediction without mutating it or accessing Gold."""
    value = deepcopy(record)
    if value.get("schema_version") != "firebench-interop-v1":
        raise HandoffAdaptationError("input_schema_version_must_be_firebench_interop_v1")
    case_id = value.get("case_id")
    method_id = value.get("method_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise HandoffAdaptationError("case_id_must_be_non_empty_string")
    if not isinstance(method_id, str) or not method_id.strip():
        raise HandoffAdaptationError("method_id_must_be_non_empty_string")
    prediction = value.get("prediction")
    if not isinstance(prediction, dict):
        raise HandoffAdaptationError("prediction_must_be_object")
    final_response = prediction.get("final_response")
    if not isinstance(final_response, dict):
        raise HandoffAdaptationError("prediction.final_response_must_be_object")
    actual_output = final_response.get("text")
    synthetic = False
    if not isinstance(actual_output, str) or not actual_output.strip():
        if formal or not allow_synthetic_actual_output:
            raise HandoffAdaptationError("actual_output_missing; structured fields are not a fallback")
        actual_output = "[development fixture: missing baseline final response]"
        synthetic = True

    method_metadata = value.get("method_metadata")
    method_metadata = method_metadata if isinstance(method_metadata, dict) else {}
    raw_contexts = method_metadata.get("retrieved_contexts")
    contexts_present = isinstance(raw_contexts, list) and bool(raw_contexts)
    comparison_level, retrieval_required = _comparison_level(method_id, contexts_present)
    if method_id == DIRECT_METHOD and contexts_present:
        raise HandoffAdaptationError("direct_llm_must_not_have_retrieval_context")

    context_result = None
    if method_id != DIRECT_METHOD and raw_contexts is not None:
        try:
            context_result = normalize_retrieval_contexts(raw_contexts, top_k=top_k)
        except ContextNormalizationError as exc:
            raise HandoffAdaptationError(str(exc)) from exc
    if formal and retrieval_required and (context_result is None or not context_result.contexts):
        raise HandoffAdaptationError(f"formal_rag_context_missing:{method_id}")

    recommended = prediction.get("recommended_actions")
    recommended = recommended if isinstance(recommended, list) else []
    structured: dict[str, Any] = {
        "risk_signals": list(prediction.get("risk_signals") or []),
        "required_actions": [
            item["action_id"]
            for item in recommended
            if isinstance(item, dict) and isinstance(item.get("action_id"), str)
        ],
        "blocked_actions": list(prediction.get("blocked_actions") or []),
        "missing_confirmations": list(prediction.get("missing_confirmations") or []),
        "hitl_required": prediction.get("human_review_required"),
        "final_status": prediction.get("final_decision_gate", final_response.get("status")),
        "real_world_execution_allowed": final_response.get("real_world_execution_allowed"),
    }
    for field in ("required_tools", "routing_decisions"):
        if field in prediction:
            structured[field] = deepcopy(prediction[field])

    runtime = value.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    artifacts = deepcopy(source_artifacts or {})
    missing_identity = sorted(key for key, item in artifacts.items() if item is None)
    output: dict[str, Any] = {
        "schema_version": "fireagent-external-prediction-v1",
        "case_id": case_id,
        "system_name": method_id,
        "system_type": "external_baseline",
        "actual_output": actual_output,
        "structured_prediction": structured,
        "runtime": {
            "latency_ms": runtime.get("latency_ms"),
            "llm_calls": runtime.get("llm_calls"),
            "token_usage": deepcopy(runtime.get("token_usage") or {}),
            "cost_estimate": runtime.get("cost"),
        },
        "source_artifacts": artifacts,
        "metadata": {
            "comparison_level": comparison_level,
            "retrieval_required": retrieval_required,
            "native_retrieval_context_count": context_result.native_count if context_result else None,
            "submitted_retrieval_context_count": context_result.submitted_count if context_result else None,
            "retrieval_context_truncated_for_handoff": context_result.truncated if context_result else None,
            "handoff_top_k": top_k if context_result else None,
            "context_selection_policy": "original_rank_prefix" if context_result else None,
            "missing_identity": missing_identity,
            "formal_eligible": bool(formal and not synthetic),
            "synthetic_actual_output": synthetic,
            "system_execution_capability": False,
        },
    }
    if context_result is not None:
        output["retrieval_context"] = context_result.contexts
    return output
