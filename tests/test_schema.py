from external_baselines.common.schema import BaselineOutput, normalize_response_payload


def test_baseline_output_serializes_correctly():
    out = BaselineOutput(scenario_id="s1", method="direct_llm", key_risks=["smoke"])
    data = out.to_dict()
    assert data["scenario_id"] == "s1"
    assert data["method"] == "direct_llm"
    assert data["key_risks"] == ["smoke"]
    assert "retrieved_contexts" in data


def test_normalize_response_payload_handles_missing_fields():
    out = normalize_response_payload({"summary": "x"}, scenario_id="s1", method="m")
    data = out.to_dict()
    assert data["situation_summary"] == "x"
    assert data["recommended_actions"] == []
    assert data["final_decision_gate"] == "not_applicable_or_not_provided"
