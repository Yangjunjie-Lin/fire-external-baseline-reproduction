import json

from external_baselines.common.io import flatten_scenario, load_scenarios


def test_load_scenarios_handles_scenario_matrix_dict(tmp_path):
    path = tmp_path / "scenario_matrix_v2.json"
    path.write_text(json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "fire", "expected": {"x": 1}}]}), encoding="utf-8")
    rows = load_scenarios(path)
    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "s1"
    assert rows[0]["expected"] == {"x": 1}


def test_flatten_scenario_preserves_id_and_expected():
    row = flatten_scenario({"id": "abc", "description": "Smoke in mall", "expected": {"requires_human_confirmation": True}})
    assert row["scenario_id"] == "abc"
    assert "Smoke" in row["scenario_text"]
    assert row["expected"]["requires_human_confirmation"] is True
