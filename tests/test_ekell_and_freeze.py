"""E-KELL stage trace, evidence IDs, frozen config, prediction count, bundle tests."""

from __future__ import annotations

import json
from pathlib import Path

from external_baselines.common.io import load_config, write_json
from external_baselines.interop.bundle import load_runner_bundle, validate_bundle_checksum
from external_baselines.runner import generate_predictions

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "frozen"


def test_ekell_stage_trace_complete(tmp_path):
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(
        json.dumps({
            "scenarios": [{
                "scenario_id": "s1",
                "scenario_text": "Electrical fire with smoke in a shopping mall.",
            }]
        }),
        encoding="utf-8",
    )
    rows = generate_predictions(
        methods=["ekell_style_controlled_shared_llm"],
        dataset=dataset,
        output_path=tmp_path / "out.jsonl",
        manifest_path=None,
        config=load_config(
            ROOT / "configs" / "default.yaml",
            ROOT / "configs" / "deterministic_heuristic_smoke.yaml",
        ),
    )
    ms = rows[0]["method_specific"]
    for key in [
        "pipeline_trace",
        "logical_decomposition",
        "vector_retrieval",
        "neighborhood_expansion",
        "fol_execution",
        "stepwise_prompt_chain",
        "reproduction_class",
    ]:
        assert key in ms, f"missing {key}"
    assert ms["reproduction_class"] == "controlled_shared_llm"
    assert ms["official_reproduction"] is False
    assert "Vector KG Retrieval" in ms["pipeline_trace"]
    assert "FOL Execution" in ms["pipeline_trace"]
    assert len(ms["pipeline_trace"]) >= 8


def test_evidence_ids_preserved(tmp_path):
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(
        json.dumps({
            "scenarios": [{
                "scenario_id": "s1",
                "scenario_text": "Electrical fire smoke mall.",
            }]
        }),
        encoding="utf-8",
    )
    rows = generate_predictions(
        methods=["bm25_rag"],
        dataset=dataset,
        output_path=tmp_path / "out.jsonl",
        manifest_path=None,
        config=load_config(ROOT / "configs" / "default.yaml"),
    )
    assert "raw_output" in rows[0]
    for ctx in rows[0].get("retrieved_contexts") or []:
        assert ctx.get("context_id")


def test_frozen_config_not_overwritten():
    required = [
        "direct_llm_v1.yaml",
        "bm25_rag_v1.yaml",
        "dense_rag_v1.yaml",
        "hybrid_rag_v1.yaml",
        "ekell_controlled_shared_llm_v1.yaml",
        "ekell_style_faithful_v1.yaml",
        "ekell_style_enhanced_v1.yaml",
        "freeze_manifest.json",
    ]
    for name in required:
        assert (FROZEN / name).exists(), name
    # Guard: freeze files must declare split_policy and provisional status until DEV evidence exists.
    text = (FROZEN / "bm25_rag_v1.yaml").read_text(encoding="utf-8")
    assert "tuned_on_dev_only_test_frozen" in text
    assert "freeze_status: provisional" in text
    freeze_manifest = (FROZEN / "freeze_manifest.json").read_text(encoding="utf-8")
    assert '"freeze_status": "provisional"' in freeze_manifest


def test_prediction_count_matches_cases(tmp_path):
    dataset = tmp_path / "scenarios.json"
    cases = [
        {"scenario_id": "a", "scenario_text": "Fire A"},
        {"scenario_id": "b", "scenario_text": "Fire B"},
    ]
    dataset.write_text(json.dumps({"scenarios": cases}), encoding="utf-8")
    methods = ["direct_llm", "bm25_rag"]
    rows = generate_predictions(
        methods=methods,
        dataset=dataset,
        output_path=tmp_path / "out.jsonl",
        manifest_path=None,
        config={"llm": {"provider": "heuristic"}, "normalization": {"infer_structured_safety_fields": False}, "paths": {"corpus_dir": str(ROOT / "data" / "corpus")}},
    )
    assert len(rows) == len(cases) * len(methods)


def test_bundle_checksum(tmp_path):
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "x"}]}), encoding="utf-8")
    write_json(tmp_path / "manifest.json", {"bundle_checksum": "deadbeef", "name": "test-bundle"})
    write_json(tmp_path / "experiment_config.json", {"paths": {"scenario_file": str(scenarios)}})
    # Point scenario via file copy into bundle layout
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "scenarios" / "scenarios.json").write_text(scenarios.read_text(encoding="utf-8"), encoding="utf-8")
    bundle = load_runner_bundle(tmp_path)
    report = validate_bundle_checksum(bundle, expected="deadbeef")
    assert report["ok"] is False
    assert report["declared"] != report["recomputed"]
    assert bundle["forbidden_keys_stripped"] is False


def test_same_model_config_fields_present():
    cfg = load_config(ROOT / "configs" / "shared_real_model.yaml.example")
    llm = cfg["llm"]
    for key in ("provider", "model", "model_version", "temperature", "top_p", "max_tokens"):
        assert key in llm


def test_ekell_controlled_vs_enhanced_method_ids(tmp_path):
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "Electrical smoke mall fire."}]}),
        encoding="utf-8",
    )
    base = load_config(ROOT / "configs" / "default.yaml")
    controlled = generate_predictions(
        methods=["ekell_style_controlled_shared_llm"],
        dataset=dataset,
        output_path=tmp_path / "f.jsonl",
        manifest_path=None,
        config=base,
    )[0]
    enhanced_cfg = dict(base)
    enhanced_cfg["ekell_style"] = {
        **base.get("ekell_style", {}),
        "dense_entity_retrieval": True,
        "hybrid_subgraph_ranking": True,
    }
    enhanced = generate_predictions(
        methods=["ekell_style_enhanced"],
        dataset=dataset,
        output_path=tmp_path / "e.jsonl",
        manifest_path=None,
        config=enhanced_cfg,
    )[0]
    assert controlled["method"] == "ekell_style_controlled_shared_llm"
    assert enhanced["method"] == "ekell_style_enhanced"
    assert controlled["method_specific"]["reproduction_class"] == "controlled_shared_llm"
    assert enhanced["method_specific"]["reproduction_class"] == "enhanced"
    assert enhanced["method_specific"].get("paper_table_role") == "supplemental_extended"
