"""Gold isolation: prediction generation must never receive expected/gold/target outputs."""

from __future__ import annotations

import json

from external_baselines.common.io import (
    assert_no_gold_in_prediction_input,
    flatten_scenario,
    to_prediction_input,
)
from external_baselines.runner import generate_predictions, get_pipeline


def test_gold_not_passed_to_pipeline(tmp_path, monkeypatch):
    seen = {}

    def fake_pipeline(scenario, *, config=None, llm=None):
        seen["keys"] = set(scenario.keys())
        seen["scenario"] = scenario
        return {
            "scenario_id": scenario["scenario_id"],
            "method": "direct_llm",
            "situation_summary": "ok",
            "key_risks": [],
            "recommended_actions": [],
            "blocked_or_unsafe_actions": [],
            "missing_confirmations": [],
            "supporting_evidence": [],
            "citations": [],
            "final_decision_gate": "not_provided_by_baseline",
            "retrieved_contexts": [],
            "latency_sec": 0.0,
            "raw_output": {"text": "{}", "parsed": {}},
            "method_specific": {},
        }

    monkeypatch.setattr("external_baselines.runner.get_pipeline", lambda method: fake_pipeline)
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(
        json.dumps({
            "scenarios": [{
                "scenario_id": "s1",
                "scenario_text": "Smoke in electrical room.",
                "expected": {"key_risks": ["electrical"]},
                "gold": {"secret": True},
                "labels": {"x": 1},
                "annotation_notes": "do not leak",
                "target_outputs": {"recommended_actions": ["secret"]},
            }]
        }),
        encoding="utf-8",
    )
    generate_predictions(
        methods=["direct_llm"],
        dataset=dataset,
        output_path=tmp_path / "out.jsonl",
        manifest_path=tmp_path / "manifest.json",
        config={"llm": {"provider": "heuristic"}, "normalization": {"infer_structured_safety_fields": False}},
    )
    assert "expected" not in seen["keys"]
    assert "gold" not in seen["keys"]
    assert "labels" not in seen["keys"]
    assert "source_record" not in seen["keys"]
    assert "target_outputs" not in seen["keys"]
    blob = json.dumps(seen["scenario"])
    assert "secret" not in blob
    assert "do not leak" not in blob


def test_expected_not_passed_to_pipeline():
    record = flatten_scenario({
        "scenario_id": "s1",
        "scenario_text": "Fire",
        "expected": {"a": 1},
    })
    assert "expected" in record
    pred = to_prediction_input(record, config={"paths": {"corpus_dir": "data/corpus"}})
    assert "expected" not in pred
    assert_no_gold_in_prediction_input(pred)


def test_target_outputs_not_read_during_generation():
    record = flatten_scenario({
        "case_id": "c1",
        "scenario_text": "Mall fire",
        "target_outputs": {"final_response": "TARGET"},
        "evaluator_hints": {"boost": True},
    })
    pred = to_prediction_input(record)
    assert "target_outputs" not in pred
    assert "TARGET" not in json.dumps(pred)


def test_prediction_generation_works_without_gold(tmp_path):
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "Small fire with smoke."}]}),
        encoding="utf-8",
    )
    rows = generate_predictions(
        methods=["direct_llm"],
        dataset=dataset,
        output_path=tmp_path / "out.jsonl",
        manifest_path=tmp_path / "manifest.json",
        config={"llm": {"provider": "heuristic"}, "normalization": {"infer_structured_safety_fields": False}},
    )
    assert len(rows) == 1
    assert rows[0]["raw_output"] is not None


def test_raw_output_preserved(tmp_path):
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "Electrical fire."}]}),
        encoding="utf-8",
    )
    rows = generate_predictions(
        methods=["direct_llm"],
        dataset=dataset,
        output_path=tmp_path / "out.jsonl",
        manifest_path=None,
        config={"llm": {"provider": "heuristic"}, "normalization": {"infer_structured_safety_fields": False}},
    )
    assert "raw_output" in rows[0]
    assert rows[0]["raw_output"].get("text")


def test_get_pipeline_aliases():
    assert get_pipeline("vanilla_rag")
    assert get_pipeline("bm25_rag")
    assert get_pipeline("ekell_style")
    assert get_pipeline("ekell_style_faithful")
    assert get_pipeline("ekell_style_enhanced")
    assert get_pipeline("dense_rag")
    assert get_pipeline("hybrid_rag")
