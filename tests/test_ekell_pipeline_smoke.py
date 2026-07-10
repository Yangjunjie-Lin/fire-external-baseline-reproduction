import json

from external_baselines.common.llm_client import HeuristicLLMClient
from external_baselines.ekell_style.pipeline import run_scenario


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_ekell_style_run_scenario_returns_unified_schema(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_jsonl(corpus / "entities.jsonl", [{"entity_id": "e1", "name": "electrical fire", "aliases": ["power fire"]}])
    write_jsonl(corpus / "triples.jsonl", [{"head": "electrical fire", "relation": "requires", "tail": "power isolation", "source_id": "kg1"}])
    write_jsonl(corpus / "relations.jsonl", [])
    write_jsonl(corpus / "evidence_chunks.jsonl", [{"chunk_id": "c1", "text": "Electrical fires require confirming power isolation.", "source_id": "doc1"}])
    scenario = {"scenario_id": "s1", "scenario_text": "Electrical room fire with high smoke and unknown power status."}
    config = {"paths": {"corpus_dir": str(corpus)}, "llm": {"provider": "heuristic", "temperature": 0.0, "max_tokens": 1200}, "scenario_parser": {"use_llm": False}, "ekell_style": {"prompt_dir": "configs/prompts"}}
    out = run_scenario(scenario, config=config, llm=HeuristicLLMClient())
    assert out["scenario_id"] == "s1"
    assert out["method"] == "ekell_style_faithful"
    assert out["method_specific"]["reproduction_class"] == "faithful"
    assert isinstance(out["recommended_actions"], list)
    assert "llm_config_summary" in out["method_specific"]
    assert out["retrieved_contexts"]
