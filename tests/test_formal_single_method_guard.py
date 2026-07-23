from __future__ import annotations

import pytest

from external_baselines.common.decision_suite_guard import (
    FormalSuiteExecutionError,
    validate_decision_suite_execution,
)


def test_formal_guard_allows_one_explicit_checkpoint_method(monkeypatch, tmp_path):
    manifest = tmp_path / "comparison.yaml"
    manifest.write_text("freeze_status: frozen\n", encoding="utf-8")
    monkeypatch.setattr(
        "external_baselines.common.decision_suite_guard.validate_experiment_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "external_baselines.common.firebench_taxonomy.validate_formal_alias_table",
        lambda: None,
    )
    validate_decision_suite_execution(
        execution_stage="formal",
        experiment_manifest=manifest,
        method_ids=["bm25_rag"],
    )


def test_formal_guard_rejects_incomplete_multi_method_subset(tmp_path):
    manifest = tmp_path / "comparison.yaml"
    manifest.write_text("freeze_status: frozen\n", encoding="utf-8")
    with pytest.raises(FormalSuiteExecutionError, match="one explicit atomic method checkpoint"):
        validate_decision_suite_execution(
            execution_stage="formal",
            experiment_manifest=manifest,
            method_ids=["bm25_rag", "direct_llm"],
        )
