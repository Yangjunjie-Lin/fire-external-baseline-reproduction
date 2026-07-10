from __future__ import annotations

from external_baselines.interop.schema import baseline_row_to_interop


def _row(**overrides):
    row = {
        "scenario_id": "case-1",
        "method": "bm25_rag",
        "situation_summary": "Summary",
        "recommended_actions": [{"text": "Act"}],
        "retrieved_contexts": [{"context_id": "ctx-1", "text": "Retrieved"}],
    }
    row.update(overrides)
    return row


def test_supporting_text_is_not_promoted_to_evidence_id():
    prediction = baseline_row_to_interop(
        _row(supporting_evidence=["Free-text rationale"])
    )["prediction"]
    assert prediction["evidence_refs"] == []
    assert prediction["evidence_statements"] == ["Free-text rationale"]


def test_global_evidence_is_not_attached_to_every_action():
    prediction = baseline_row_to_interop(
        _row(citations=["ctx-1"], recommended_actions=[{"text": "Act"}, "Then wait"])
    )["prediction"]
    assert [action["evidence_refs"] for action in prediction["recommended_actions"]] == [[], []]


def test_invalid_claimed_citation_is_preserved():
    prediction = baseline_row_to_interop(_row(citations=["missing-id"]))["prediction"]
    assert prediction["claimed_citations"] == [
        {"evidence_id": "missing-id", "id_exists": False}
    ]
