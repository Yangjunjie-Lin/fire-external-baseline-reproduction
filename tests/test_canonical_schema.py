"""Canonical firebench-interop-v1 schema and guard tests."""

from __future__ import annotations

import pytest

from external_baselines.common.guards import (
    ConfigGuardError,
    assert_actual_graphrag_allowed,
    assert_paper_final_allowed,
    method_leaderboard_eligibility,
)
from external_baselines.evaluation.normalizer import maybe_infer_structured_safety_fields
from external_baselines.interop.schema import baseline_row_to_interop, validate_interop_record


def _sample_baseline_row():
    return {
        "scenario_id": "case-1",
        "method": "direct_llm",
        "situation_summary": "Smoke reported.",
        "key_risks": ["smoke exposure"],
        "recommended_actions": ["Confirm ventilation status"],
        "blocked_or_unsafe_actions": ["Defer interior entry"],
        "missing_confirmations": ["SCBA readiness"],
        "supporting_evidence": [],
        "citations": [],
        "final_decision_gate": "critical_information_missing_or_requires_human_confirmation",
        "retrieved_contexts": [],
        "latency_sec": 0.12,
        "raw_output": {
            "text": "{\"situation_summary\": \"Smoke reported.\"}",
            "parsed": {"situation_summary": "Smoke reported."},
        },
        "method_specific": {
            "runtime": {
                "llm_calls": 1,
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "llm_calls": 1,
                },
                "cost": None,
            },
            "structured_safety_fields": "baseline_generated_only",
            "normalizer_policy_injection": False,
        },
    }


def test_canonical_schema():
    record = baseline_row_to_interop(_sample_baseline_row(), bundle_checksum="abc")
    errors = validate_interop_record(record)
    assert errors == []
    assert record["case_id"] == "case-1"
    assert record["method_id"] == "direct_llm"
    assert record["prediction"]["raw_output"] is not None
    assert record["prediction"]["final_response"]["real_world_execution_allowed"] is False
    assert record["runtime"]["latency_ms"] == 120.0


def test_normalizer_does_not_inject_missing_safety_fields():
    row = {
        "situation_summary": "Possible electrical fire; water may be unsafe; smoke present needing respiratory protection.",
        "key_risks": ["electrical", "smoke"],
        "recommended_actions": ["Assess scene"],
        "blocked_or_unsafe_actions": [],
        "missing_confirmations": [],
        "final_decision_gate": "not_applicable_or_not_provided",
        "method_specific": {},
    }
    out = maybe_infer_structured_safety_fields(
        row, {"normalization": {"infer_structured_safety_fields": False}}
    )
    assert out["blocked_or_unsafe_actions"] == []
    assert out["missing_confirmations"] == []
    assert out["final_decision_gate"] == "not_applicable_or_not_provided"
    assert out["method_specific"]["normalizer_policy_injection"] is False


def test_normalizer_no_policy_injection_default():
    row = {
        "situation_summary": "power and water mentioned",
        "blocked_or_unsafe_actions": [],
        "missing_confirmations": [],
        "final_decision_gate": "not_applicable_or_not_provided",
        "method_specific": {},
    }
    out = maybe_infer_structured_safety_fields(row, {})
    assert out["blocked_or_unsafe_actions"] == []


def test_heuristic_rejected_for_final():
    with pytest.raises(ConfigGuardError):
        assert_paper_final_allowed(
            {
                "paper_final": True,
                "llm": {"provider": "heuristic", "model": "x", "model_version": "1"},
            }
        )


def test_paper_final_requires_model_version():
    with pytest.raises(ConfigGuardError):
        assert_paper_final_allowed(
            {"paper_final": True, "llm": {"provider": "openai", "model": "gpt-x"}}
        )


def test_fallback_excluded_from_actual_graphrag():
    with pytest.raises(ConfigGuardError):
        assert_actual_graphrag_allowed(
            {"actual_external_package_used": False, "fallback_retrieval_used": True},
            claim_actual=True,
        )
    elig = method_leaderboard_eligibility(
        "microsoft_graphrag",
        {"actual_external_package_used": False, "fallback_retrieval_used": True},
    )
    assert elig["actual_graphrag_leaderboard"] is False
    assert elig["smoke_or_fallback_only"] is True

    elig_fb = method_leaderboard_eligibility(
        "fallback_graph_retrieval", {"fallback_retrieval_used": True}
    )
    assert elig_fb["formal_leaderboard"] is False
