from __future__ import annotations

from copy import deepcopy

import pytest

from external_baselines.common.decision_output import (
    DecisionOutput,
    decision_output_to_interop,
    decision_output_to_legacy_row,
    unified_row_to_interop,
)
from external_baselines.interop.deepeval_handoff.adapter import (
    HandoffAdaptationError,
    adapt_firebench_interop_to_external_prediction,
)


def _interop(method_id: str = "bm25_rag") -> dict:
    return {
        "schema_version": "firebench-interop-v1",
        "case_id": "case-1",
        "method_id": method_id,
        "prediction": {
            "risk_signals": ["RISK_A"],
            "recommended_actions": [{"action_id": "ACT_A", "text": "act"}],
            "blocked_actions": ["BLOCK_A"],
            "missing_confirmations": ["CONFIRM_A"],
            "human_review_required": True,
            "final_decision_gate": "await_human_confirmation",
            "final_response": {
                "status": "provided",
                "text": "Exact baseline response.",
                "real_world_execution_allowed": False,
            },
        },
        "runtime": {"latency_ms": 2.5, "llm_calls": 1, "token_usage": {"total": 5}, "cost": None},
        "provenance": {},
        "method_metadata": {"retrieved_contexts": [{"text": "exact context", "rank": 1}]},
    }


def test_adapter_maps_fields_runtime_hashes_and_does_not_mutate() -> None:
    source = _interop()
    before = deepcopy(source)
    hashes = {
        "prediction_path": "predictions/bm25_rag.jsonl",
        "prediction_sha256": "a" * 64,
        "resource_bundle_sha256": None,
        "runtime_config_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
    }
    output = adapt_firebench_interop_to_external_prediction(source, top_k=5, source_artifacts=hashes)
    assert source == before
    assert output["case_id"] == "case-1"
    assert output["system_name"] == "bm25_rag"
    assert output["actual_output"] == "Exact baseline response."
    assert output["structured_prediction"]["required_actions"] == ["ACT_A"]
    assert output["structured_prediction"]["blocked_actions"] == ["BLOCK_A"]
    assert output["structured_prediction"]["missing_confirmations"] == ["CONFIRM_A"]
    assert output["structured_prediction"]["hitl_required"] is True
    assert output["runtime"]["cost_estimate"] is None
    assert output["source_artifacts"] == hashes


def test_direct_context_is_omitted_and_fabricated_context_is_rejected() -> None:
    direct = _interop("direct_llm")
    direct["method_metadata"]["retrieved_contexts"] = []
    output = adapt_firebench_interop_to_external_prediction(direct)
    assert "retrieval_context" not in output
    direct["method_metadata"]["retrieved_contexts"] = [{"text": "not native", "rank": 1}]
    with pytest.raises(HandoffAdaptationError, match="must_not_have"):
        adapt_firebench_interop_to_external_prediction(direct)


def test_empty_formal_actual_output_fails_without_structured_fallback() -> None:
    source = _interop()
    source["prediction"]["final_response"]["text"] = ""
    with pytest.raises(HandoffAdaptationError, match="structured fields are not a fallback"):
        adapt_firebench_interop_to_external_prediction(source)


def test_decision_and_unified_serialization_preserve_contexts() -> None:
    contexts = [
        {"text": "first", "rank": 1, "metadata": {"origin": "native"}},
        {"text": "second", "rank": 2},
    ]
    decision = DecisionOutput(case_id="c", method_id="bm25_rag", retrieved_contexts=contexts)
    direct = decision_output_to_interop(decision)
    unified = unified_row_to_interop(decision_output_to_legacy_row(decision))
    for record in (direct, unified):
        assert record["method_metadata"]["retrieved_contexts"] == contexts
        assert [item["rank"] for item in record["method_metadata"]["retrieved_evidence"]] == [1, 2]
        assert [item["text"] for item in record["method_metadata"]["retrieved_evidence"]] == ["first", "second"]
