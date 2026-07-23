from __future__ import annotations

from external_baselines.common.runtime_evidence import (
    RuntimeEvidence,
    compute_suite_formal_compliance,
)


def _bm25_evidence() -> RuntimeEvidence:
    return RuntimeEvidence(
        method_id="bm25_rag",
        llm_provider="openai",
        llm_model="gpt-5.6-sol",
        llm_model_version="gpt-5.6-sol",
        llm_temperature=0.0,
        llm_top_p=1.0,
        llm_max_tokens=1200,
        llm_seed=7,
        llm_enable_thinking=False,
        llm_is_smoke=False,
        llm_initialized=True,
    )


def _compliance(method_ids):
    return compute_suite_formal_compliance(
        formal=True,
        experiment_manifest_provided=True,
        limit_used=False,
        preflight_ok=True,
        coverage_ok=True,
        method_evidences={"bm25_rag": _bm25_evidence()},
        method_compliance={"bm25_rag": {"formal_result": True}},
        dev_aliases_enabled=False,
        runner_bundle_integrity_ok=True,
        input_cases_integrity_ok=True,
        prediction_schema_integrity_ok=True,
        corpus_integrity_ok=True,
        shared_generation_identity_match=True,
        method_ids=method_ids,
        phase="pre_publish",
    )


def test_bm25_checkpoint_marks_unaccessed_index_gates_not_applicable():
    result = _compliance(["bm25_rag"])
    assert result["pre_publish_compliance_passed"] is True
    assert result["real_dense_index"] is None
    assert result["real_ekell_index"] is None
    assert result["formal_scope"] == "single_method_checkpoint"


def test_full_suite_cannot_pass_with_only_bm25_evidence():
    result = _compliance(
        [
            "direct_llm",
            "bm25_rag",
            "dense_rag",
            "hybrid_rag",
            "ekell_style_controlled_shared_llm",
        ]
    )
    assert result["pre_publish_compliance_passed"] is False
    assert result["formal_scope"] == "comparison_suite"
