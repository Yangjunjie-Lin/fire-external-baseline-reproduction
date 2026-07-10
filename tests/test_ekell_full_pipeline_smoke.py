"""End-to-end smoke for complete E-KELL full pipeline (heuristic only)."""

from __future__ import annotations

import json

from external_baselines.common.llm_client import HeuristicLLMClient
from external_baselines.ekell_style.full_pipeline import run_controlled_shared_llm, run_paper_fidelity


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _tiny_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_jsonl(corpus / "entities.jsonl", [
        {"entity_id": "E_ELECTRICAL_FIRE", "name": "electrical fire", "aliases": ["power fire"]},
        {"entity_id": "E_HIGH_SMOKE", "name": "high smoke", "aliases": ["smoke"]},
        {"entity_id": "E_POWER_ISOLATION", "name": "power isolation"},
        {"entity_id": "E_RESPIRATORY_PROTECTION", "name": "respiratory protection"},
    ])
    write_jsonl(corpus / "triples.jsonl", [
        {"head": "electrical fire", "relation": "requires_confirmation", "tail": "power isolation", "source_id": "kg1"},
        {"head": "high smoke", "relation": "requires", "tail": "respiratory protection", "source_id": "kg1"},
    ])
    write_jsonl(corpus / "relations.jsonl", [
        {"relation": "requires_confirmation"},
        {"relation": "requires"},
    ])
    write_jsonl(corpus / "evidence_chunks.jsonl", [
        {"chunk_id": "c1", "text": "Electrical fires require confirming power isolation.", "source_id": "doc1"},
        {"chunk_id": "c2", "text": "High smoke requires respiratory protection.", "source_id": "doc1"},
    ])
    return corpus


def test_controlled_full_pipeline_smoke(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    scenario = {"scenario_id": "s1", "scenario_text": "Electrical room fire with high smoke and unknown power status."}
    config = {
        "paths": {"corpus_dir": str(corpus)},
        "llm": {"provider": "heuristic", "temperature": 0.0, "max_tokens": 1200},
        "scenario_parser": {"use_llm": False},
        "ekell_style": {"prompt_dir": "configs/prompts/paper_fidelity", "neighborhood_k_hop": 1},
        "ekell_vector": {"backend": "smoke", "dimension": 32, "top_k": 4},
        "normalization": {"infer_structured_safety_fields": False},
        "paper_final": False,
    }
    out = run_controlled_shared_llm(scenario, config=config, llm=HeuristicLLMClient())
    assert out["method"] == "ekell_style_controlled_shared_llm"
    ms = out["method_specific"]
    assert ms["track"] == "controlled_shared_llm"
    assert ms["controlled_output_format"] is True
    assert ms["paper_original_output_format"] is False
    assert "Vector KG Retrieval" in ms["pipeline_trace"]
    assert "FOL Execution" in ms["pipeline_trace"]
    assert "logical_decomposition" in ms
    assert "vector_retrieval" in ms
    assert ms["vector_retrieval"]["smoke_fallback_used"] is True
    assert out["raw_output"]


def test_paper_fidelity_full_pipeline_smoke(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    scenario = {"scenario_id": "s1", "scenario_text": "Electrical fire with smoke."}
    config = {
        "paths": {"corpus_dir": str(corpus)},
        "llm": {"provider": "heuristic"},
        "ekell_style": {"prompt_dir": "configs/prompts/paper_fidelity"},
        "ekell_vector": {"backend": "smoke", "dimension": 32},
        "normalization": {"infer_structured_safety_fields": False},
        "paper_final": False,
    }
    out = run_paper_fidelity(scenario, config=config, llm=HeuristicLLMClient())
    assert out["method"] == "ekell_style_paper_fidelity"
    assert out["method_specific"]["paper_original_output_format"] is True
    assert out["method_specific"]["controlled_output_format"] is False
    assert out["method_specific"]["kg_status"]["official_ekell_kg"] is False
