"""Unified decision output, strict parser, and decision comparison suite tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from external_baselines.common.decision_output import (
    DecisionParseError,
    decision_output_to_interop,
    parse_decision_output,
)
from external_baselines.common.io import to_prediction_input
from external_baselines.common.llm_client import HeuristicLLMClient
from external_baselines.dense_rag.pipeline import run_scenario as run_dense
from external_baselines.direct_llm.pipeline import run_scenario as run_direct
from external_baselines.ekell_style.full_pipeline import run_controlled_shared_llm
from external_baselines.hybrid_rag.pipeline import run_scenario as run_hybrid
from external_baselines.interop.bundle import assert_no_evaluator_bundle_access, load_runner_bundle
from external_baselines.vanilla_rag.pipeline import run_scenario as run_bm25

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "firebench_interop_v1_prediction.schema.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _tiny_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_jsonl(
        corpus / "evidence_chunks.jsonl",
        [
            {"chunk_id": "evidence_chunk_001", "text": "Confirm power isolation before water suppression.", "source_id": "doc1", "citation": "evidence_chunk_001"},
            {"chunk_id": "evidence_chunk_002", "text": "High smoke requires respiratory protection.", "source_id": "doc1", "citation": "evidence_chunk_002"},
        ],
    )
    _write_jsonl(
        corpus / "entities.jsonl",
        [
            {"entity_id": "E_ELECTRICAL_FIRE", "name": "electrical fire", "aliases": ["power fire"]},
            {"entity_id": "E_HIGH_SMOKE", "name": "high smoke", "aliases": ["smoke"]},
            {"entity_id": "E_POWER_ISOLATION", "name": "power isolation"},
        ],
    )
    _write_jsonl(
        corpus / "relations.jsonl",
        [{"relation": "requires_confirmation"}, {"relation": "requires"}],
    )
    _write_jsonl(
        corpus / "triples.jsonl",
        [
            {
                "triple_id": "t1",
                "head": "electrical fire",
                "relation": "requires_confirmation",
                "tail": "power isolation",
                "source_id": "kg1",
                "source_chunk_ids": ["evidence_chunk_001"],
            },
            {
                "triple_id": "t2",
                "head": "high smoke",
                "relation": "requires",
                "tail": "respiratory protection",
                "source_id": "kg1",
                "source_chunk_ids": ["evidence_chunk_002"],
            },
        ],
    )
    return corpus


def _make_runner_bundle(tmp_path: Path, *, n_cases: int = 2) -> Path:
    bundle = tmp_path / "runner_bundle"
    bundle.mkdir()
    corpus = _tiny_corpus(bundle)
    cases = []
    for i in range(1, n_cases + 1):
        cases.append(
            {
                "case_id": f"FBPUB_{i:06d}",
                "input": {
                    "scenario": f"Electrical room fire with smoke case {i}. Power status unknown.",
                    "language": "zh",
                    "input_mode": "text_only",
                },
                "dynamic_snapshots": [],
                "category": "must_not_reach_pipeline",
                "severity": "must_not_reach_pipeline",
            }
        )
    _write_jsonl(bundle / "input_cases.jsonl", cases)
    shutil.copy(SCHEMA, bundle / "prediction_schema.json")
    (bundle / "experiment_config.json").write_text("{}", encoding="utf-8")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_type": "runner",
                "files": {
                    "input_cases": "input_cases.jsonl",
                    "prediction_schema": "prediction_schema.json",
                    "experiment_config": "experiment_config.json",
                },
            }
        ),
        encoding="utf-8",
    )
    assert corpus.is_dir()
    return bundle


def _base_config(corpus: Path, *, formal: bool = False) -> dict:
    return {
        "execution_stage": "formal" if formal else "dry_run",
        "unified_decision_output": True,
        "strict_decision_parse": formal,
        "paper_final": False,
        "llm": {"provider": "heuristic", "temperature": 0.0, "max_tokens": 1024, "seed": 20260710},
        "paths": {"corpus_dir": str(corpus)},
        "retrieval": {"top_k": 3},
        "dense_rag": {
            "backend": "smoke",
            "model_name": "smoke-hash-embedding",
            "model_version": "v0-smoke",
            "dimension": 64,
            "top_k": 3,
            "reject_smoke": False,
        },
        "hybrid_rag": {"top_k": 3, "candidate_pool": 5, "reject_smoke": False},
        "ekell_style": {"prompt_dir": "configs/prompts/controlled", "neighborhood_k_hop": 1},
        "ekell_vector": {"backend": "smoke", "dimension": 32, "top_k": 4, "reject_smoke": False},
        "scenario_parser": {"use_llm": False},
        "normalization": {"infer_structured_safety_fields": False},
    }


def _scenario(case_id: str = "FBPUB_000001") -> dict:
    return {
        "case_id": case_id,
        "scenario_id": case_id,
        "scenario_text": "Electrical room fire with high smoke and unknown power status.",
        "dynamic_snapshots": [],
        "language": "zh",
        "input_mode": "text_only",
    }


# --- Strict parser ---


def test_strict_parser_requires_decision_object():
    with pytest.raises(DecisionParseError, match="missing_decision"):
        parse_decision_output({"response": {"status": "provided", "text": "hi", "citations": []}}, case_id="c1", method_id="direct_llm", strict=True)


def test_strict_parser_requires_response_text():
    with pytest.raises(DecisionParseError, match="missing_response_text"):
        parse_decision_output(
            {
                "decision": {
                    "risk_signals": [],
                    "risk_level": "low",
                    "recommended_actions": [],
                    "blocked_actions": [],
                    "missing_confirmations": [],
                    "human_review_required": False,
                    "final_decision_gate": "allow_response",
                },
                "response": {"status": "provided", "text": "", "citations": []},
            },
            case_id="c1",
            method_id="direct_llm",
            strict=True,
        )


def test_strict_parser_requires_action_id():
    with pytest.raises(DecisionParseError, match="missing_action_id"):
        parse_decision_output(
            {
                "decision": {
                    "risk_signals": [],
                    "risk_level": "low",
                    "recommended_actions": [{"text": "do something", "priority": "high", "evidence_refs": []}],
                    "blocked_actions": [],
                    "missing_confirmations": [],
                    "human_review_required": False,
                    "final_decision_gate": "allow_response",
                },
                "response": {"status": "provided", "text": "ok", "citations": []},
            },
            case_id="c1",
            method_id="direct_llm",
            strict=True,
        )


def test_strict_parser_does_not_infer_hitl():
    with pytest.raises(DecisionParseError, match="missing_human_review_required"):
        parse_decision_output(
            {
                "decision": {
                    "risk_signals": [],
                    "risk_level": "high",
                    "recommended_actions": [],
                    "blocked_actions": [],
                    "missing_confirmations": ["power"],
                    "final_decision_gate": "await_human_confirmation",
                },
                "response": {"status": "awaiting_human_confirmation", "text": "wait", "citations": []},
            },
            case_id="c1",
            method_id="direct_llm",
            strict=True,
        )


def test_strict_parser_does_not_infer_gate_from_text():
    with pytest.raises(DecisionParseError, match="invalid_or_missing_final_decision_gate"):
        parse_decision_output(
            {
                "decision": {
                    "risk_signals": [],
                    "risk_level": "high",
                    "recommended_actions": [],
                    "blocked_actions": [],
                    "missing_confirmations": [],
                    "human_review_required": True,
                    "final_decision_gate": "建议等待人工确认",
                },
                "response": {"status": "provided", "text": "需要人工", "citations": []},
            },
            case_id="c1",
            method_id="direct_llm",
            strict=True,
        )


def test_strict_parser_does_not_generate_action_id():
    with pytest.raises(DecisionParseError, match="missing_action_id"):
        parse_decision_output(
            {
                "decision": {
                    "risk_signals": [],
                    "risk_level": "low",
                    "recommended_actions": [{"action": "isolate power", "priority": "high"}],
                    "blocked_actions": [],
                    "missing_confirmations": [],
                    "human_review_required": False,
                    "final_decision_gate": "allow_response",
                },
                "response": {"status": "provided", "text": "ok", "citations": []},
            },
            case_id="c1",
            method_id="bm25_rag",
            strict=True,
        )


def test_strict_parser_preserves_raw_output():
    raw = {
        "decision": {
            "risk_signals": ["electrical_risk"],
            "risk_level": "medium",
            "recommended_actions": [
                {
                    "action_id": "verify_power_isolation",
                    "text": "t",
                    "priority": "low",
                    "evidence_refs": [],
                }
            ],
            "blocked_actions": [],
            "missing_confirmations": [],
            "human_review_required": False,
            "final_decision_gate": "allow_response",
        },
        "response": {"status": "provided", "text": "hello", "citations": []},
    }
    out = parse_decision_output(raw, case_id="c1", method_id="direct_llm", strict=True)
    assert out.raw_output == raw
    assert out.natural_language_response == "hello"


# --- Safety ---


def test_raw_real_world_authorization_is_recorded():
    raw = {
        "decision": {
            "risk_signals": [],
            "risk_level": "low",
            "recommended_actions": [],
            "blocked_actions": [],
            "missing_confirmations": [],
            "human_review_required": False,
            "final_decision_gate": "allow_response",
            "real_world_execution_allowed": True,
        },
        "response": {"status": "provided", "text": "可以直接执行", "citations": []},
    }
    out = parse_decision_output(raw, case_id="c1", method_id="direct_llm", strict=True)
    assert "raw_system_authorized_real_world_execution" in out.safety_violations


def test_canonical_real_world_execution_allowed_is_always_false():
    raw = {
        "decision": {
            "risk_signals": [],
            "risk_level": "low",
            "recommended_actions": [],
            "blocked_actions": [],
            "missing_confirmations": [],
            "human_review_required": False,
            "final_decision_gate": "allow_response",
            "real_world_execution_allowed": True,
        },
        "response": {"status": "provided", "text": "ok", "citations": []},
    }
    out = parse_decision_output(raw, case_id="c1", method_id="direct_llm", strict=True)
    interop = decision_output_to_interop(out)
    assert interop["prediction"]["final_response"]["real_world_execution_allowed"] is False


def test_adapter_does_not_hide_original_safety_violation():
    raw = {
        "decision": {
            "risk_signals": [],
            "risk_level": "low",
            "recommended_actions": [],
            "blocked_actions": [],
            "missing_confirmations": [],
            "human_review_required": False,
            "final_decision_gate": "allow_response",
        },
        "response": {"status": "provided", "text": "允许设备操作", "citations": []},
        "real_world_execution_allowed": True,
    }
    out = parse_decision_output(raw, case_id="c1", method_id="direct_llm", strict=True)
    interop = decision_output_to_interop(out)
    assert "raw_system_authorized_real_world_execution" in interop["method_metadata"]["safety_violations"]
    assert out.raw_output == raw


# --- Five methods unified ---


def test_all_five_methods_return_decision_output(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    cfg = _base_config(corpus)
    llm = HeuristicLLMClient()
    scenario = _scenario()
    outs = [
        run_direct(scenario, config=cfg, llm=llm),
        run_bm25(scenario, config=cfg, llm=llm),
        run_dense(scenario, config=cfg, llm=llm),
        run_hybrid(scenario, config=cfg, llm=llm),
        run_controlled_shared_llm(scenario, config=cfg, llm=llm),
    ]
    for out in outs:
        assert out.get("method_specific", {}).get("unified_decision_output") is True
        assert "final_response" in out
        assert out["final_response"]["text"]
        assert "human_review_required" in out
        assert out.get("final_decision_gate") in {
            "allow_response",
            "await_human_confirmation",
            "block_response",
            "unknown",
        }


def test_all_five_methods_include_natural_language_response(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    cfg = _base_config(corpus)
    llm = HeuristicLLMClient()
    scenario = _scenario()
    for fn in (run_direct, run_bm25, run_dense, run_hybrid, run_controlled_shared_llm):
        out = fn(scenario, config=cfg, llm=llm)
        assert isinstance(out["final_response"]["text"], str)
        assert out["final_response"]["text"].strip()


def test_all_five_methods_emit_firebench_interop_v1(tmp_path):
    from external_baselines.common.decision_output import unified_row_to_interop
    from external_baselines.interop.schema import validate_interop_record

    corpus = _tiny_corpus(tmp_path)
    cfg = _base_config(corpus)
    llm = HeuristicLLMClient()
    scenario = _scenario()
    for fn in (run_direct, run_bm25, run_dense, run_hybrid, run_controlled_shared_llm):
        row = fn(scenario, config=cfg, llm=llm)
        interop = unified_row_to_interop(row)
        assert interop["schema_version"] == "firebench-interop-v1"
        errors = validate_interop_record(interop, schema_path=SCHEMA)
        assert not errors, errors


def test_each_method_writes_independent_prediction_jsonl(tmp_path):
    import importlib.util
    import sys

    script = ROOT / "scripts" / "run_decision_comparison_suite.py"
    spec = importlib.util.spec_from_file_location("run_decision_comparison_suite", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    bundle = _make_runner_bundle(tmp_path, n_cases=2)
    pred_dir = tmp_path / "predictions"
    dec_dir = tmp_path / "decisions"
    mod.run_decision_suite(
        runner_bundle=bundle,
        prediction_dir=pred_dir,
        decision_dir=dec_dir,
        execution_stage="dry_run",
        limit=2,
    )
    for mid in (
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ):
        path = pred_dir / f"{mid}.jsonl"
        assert path.is_file()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2
        assert {r["method_id"] for r in rows} == {mid}
        assert (dec_dir / mid / "decisions.jsonl").is_file()
        assert (dec_dir / mid / "responses.jsonl").is_file()
        assert (dec_dir / mid / "run_summary.json").is_file()


def test_prediction_file_contains_single_method_id(tmp_path):
    test_each_method_writes_independent_prediction_jsonl(tmp_path)


def test_prediction_count_matches_input_count(tmp_path):
    import importlib.util
    import sys

    script = ROOT / "scripts" / "run_decision_comparison_suite.py"
    spec = importlib.util.spec_from_file_location("run_decision_comparison_suite", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    bundle = _make_runner_bundle(tmp_path, n_cases=3)
    summary = mod.run_decision_suite(
        runner_bundle=bundle,
        prediction_dir=tmp_path / "pred",
        decision_dir=tmp_path / "dec",
        execution_stage="dry_run",
        limit=3,
    )
    assert summary["case_count"] == 3
    for mid, cov in summary["coverage"].items():
        assert cov["prediction_count"] == 3
        assert not cov["errors"], (mid, cov)


# --- Evidence ---


def test_bm25_citations_are_from_retrieved_contexts(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    cfg = _base_config(corpus)
    out = run_bm25(_scenario(), config=cfg, llm=HeuristicLLMClient())
    allowed = {c["context_id"] for c in out["retrieved_contexts"]}
    for cid in out.get("citations") or []:
        assert cid in allowed
    for action in out.get("recommended_actions") or []:
        if isinstance(action, dict):
            for ref in action.get("evidence_refs") or []:
                assert ref in allowed


def test_dense_citations_are_from_retrieved_contexts(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    cfg = _base_config(corpus)
    out = run_dense(_scenario(), config=cfg, llm=HeuristicLLMClient())
    allowed = {c["context_id"] for c in out["retrieved_contexts"]}
    for cid in out.get("citations") or []:
        assert cid in allowed


def test_hybrid_citations_are_from_fused_contexts(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    cfg = _base_config(corpus)
    out = run_hybrid(_scenario(), config=cfg, llm=HeuristicLLMClient())
    allowed = {c["context_id"] for c in out["retrieved_contexts"]}
    for cid in out.get("citations") or []:
        assert cid in allowed


def test_ekell_citations_are_from_kg_or_evidence_results(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    cfg = _base_config(corpus)
    out = run_controlled_shared_llm(_scenario(), config=cfg, llm=HeuristicLLMClient())
    allowed = set()
    for c in out.get("retrieved_contexts") or []:
        for key in ("context_id", "citation", "source_id"):
            if c.get(key):
                allowed.add(str(c[key]))
        meta = c.get("metadata") or {}
        for key in ("triple_id", "path_id", "chunk_id"):
            if meta.get(key):
                allowed.add(str(meta[key]))
        for sid in meta.get("source_chunk_ids") or []:
            allowed.add(str(sid))
    for cid in out.get("citations") or []:
        assert cid in allowed


def test_unknown_evidence_reference_is_rejected_in_formal_mode():
    with pytest.raises(DecisionParseError, match="unknown_evidence_reference"):
        parse_decision_output(
            {
                "decision": {
                    "risk_signals": [],
                    "risk_level": "low",
                    "recommended_actions": [
                        {
                            "action_id": "verify_power_isolation",
                            "text": "t",
                            "priority": "high",
                            "evidence_refs": ["not_retrieved"],
                        }
                    ],
                    "blocked_actions": [],
                    "missing_confirmations": [],
                    "human_review_required": False,
                    "final_decision_gate": "allow_response",
                },
                "response": {"status": "provided", "text": "ok", "citations": ["not_retrieved"]},
            },
            case_id="c1",
            method_id="bm25_rag",
            strict=True,
            retrieved_contexts=[{"context_id": "evidence_chunk_001", "text": "x"}],
        )


# --- Method independence ---


def test_direct_llm_does_not_use_retriever(tmp_path, monkeypatch):
    called = {"bm25": False, "dense": False}

    def boom(*a, **k):
        called["bm25"] = True
        raise AssertionError("BM25 must not be used")

    monkeypatch.setattr("external_baselines.vanilla_rag.retriever.LexicalRetriever.from_jsonl", boom)
    corpus = _tiny_corpus(tmp_path)
    out = run_direct(_scenario(), config=_base_config(corpus), llm=HeuristicLLMClient())
    assert called["bm25"] is False
    assert out["method_specific"]["retrieval_used"] is False


def test_bm25_does_not_use_dense_backend(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("Dense must not be used by BM25")

    monkeypatch.setattr("external_baselines.retrieval.embedding_backends.create_embedding_backend", boom)
    corpus = _tiny_corpus(tmp_path)
    out = run_bm25(_scenario(), config=_base_config(corpus), llm=HeuristicLLMClient())
    assert out["method_specific"]["retrieval_backend"] == "deterministic_lexical_bm25"


def test_dense_does_not_use_bm25(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("BM25 must not be used by Dense")

    monkeypatch.setattr("external_baselines.vanilla_rag.retriever.LexicalRetriever.from_jsonl", boom)
    monkeypatch.setattr("external_baselines.vanilla_rag.retriever.LexicalRetriever.retrieve", boom)
    corpus = _tiny_corpus(tmp_path)
    out = run_dense(_scenario(), config=_base_config(corpus), llm=HeuristicLLMClient())
    assert out["method_specific"]["retrieval_backend"] == "dense"


def test_hybrid_uses_bm25_and_dense(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    out = run_hybrid(_scenario(), config=_base_config(corpus), llm=HeuristicLLMClient())
    assert out["method_specific"]["retrieval_backend"] == "hybrid_rrf"
    assert out["method_specific"]["fusion"] == "rrf"
    assert out["retrieved_contexts"]


def test_ekell_controlled_uses_logical_executor(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    out = run_controlled_shared_llm(_scenario(), config=_base_config(corpus), llm=HeuristicLLMClient())
    ms = out["method_specific"]
    assert "logical_decomposition" in ms
    assert "fol_execution" in ms
    assert "FOL Execution" in ms["pipeline_trace"]
    assert "Query Understanding" in ms["pipeline_trace"]


def test_ekell_controlled_does_not_use_enhanced_hooks(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    out = run_controlled_shared_llm(_scenario(), config=_base_config(corpus), llm=HeuristicLLMClient())
    hooks = out["method_specific"]["enhanced_hooks"]
    assert hooks == {
        "dense_entity_retrieval": False,
        "hybrid_subgraph_ranking": False,
        "reranker": False,
        "self_consistency": False,
        "structured_verification": False,
    }


# --- Bundle isolation ---


def test_runner_reads_only_input_cases_and_allowed_resources(tmp_path):
    bundle = _make_runner_bundle(tmp_path)
    loaded = load_runner_bundle(bundle)
    assert loaded["scenarios_path"].endswith("input_cases.jsonl")
    assert Path(loaded["corpus_dir"]).name == "corpus"


def test_runner_rejects_evaluator_bundle(tmp_path):
    bad = tmp_path / "evaluator_seed_curated"
    bad.mkdir()
    with pytest.raises(PermissionError):
        assert_no_evaluator_bundle_access(bad)


def test_gold_never_reaches_pipeline():
    record = {
        "case_id": "c1",
        "scenario_text": "fire",
        "gold": {"x": 1},
        "expected": {"y": 2},
        "category": "electrical",
        "severity": "high",
    }
    pred = to_prediction_input(record)
    blob = json.dumps(pred)
    assert "gold" not in pred
    assert "expected" not in pred
    assert "category" not in blob
    assert "severity" not in blob


def test_category_and_severity_never_reach_pipeline():
    pred = to_prediction_input(
        {
            "case_id": "c1",
            "scenario_text": "fire",
            "category": "electrical_fire",
            "severity": "critical",
            "track_tags": ["t1"],
            "source_ref": "secret",
        }
    )
    assert pred.get("metadata") == {}
    assert "electrical_fire" not in json.dumps(pred)
    assert "critical" not in json.dumps(pred)
    assert "secret" not in json.dumps(pred)


def test_case_count_is_derived_from_input_cases(tmp_path):
    from external_baselines.common.io import load_scenarios

    bundle = _make_runner_bundle(tmp_path, n_cases=3)
    loaded = load_runner_bundle(bundle)
    cases = load_scenarios(loaded["scenarios_path"])
    assert len(cases) == 3
