"""FireBench taxonomy normalization and validation tests (offline only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baselines.common.decision_output import DecisionParseError, parse_decision_output
from external_baselines.common.firebench_taxonomy import (
    load_aliases,
    load_taxonomy,
    membership_set,
    validate_alias_table,
)
from external_baselines.common.llm_client import HeuristicLLMClient
from external_baselines.common.taxonomy_normalizer import (
    TaxonomyNormalizeReport,
    normalize_action_id,
    normalize_blocked_action_id,
    normalize_confirmation_id,
    normalize_identifier_characters,
    normalize_risk_signal,
    sort_by_taxonomy_order,
)
from external_baselines.dense_rag.pipeline import run_scenario as run_dense
from external_baselines.direct_llm.pipeline import run_scenario as run_direct
from external_baselines.ekell_style.full_pipeline import run_controlled_shared_llm
from external_baselines.hybrid_rag.pipeline import run_scenario as run_hybrid
from external_baselines.vanilla_rag.pipeline import run_scenario as run_bm25

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _tiny_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_jsonl(
        corpus / "evidence_chunks.jsonl",
        [
            {
                "chunk_id": "evidence_chunk_001",
                "text": "Confirm power isolation before water suppression.",
                "source_id": "doc1",
                "citation": "evidence_chunk_001",
            }
        ],
    )
    _write_jsonl(
        corpus / "entities.jsonl",
        [
            {"entity_id": "E_ELECTRICAL_FIRE", "name": "electrical fire"},
            {"entity_id": "E_POWER_ISOLATION", "name": "power isolation"},
        ],
    )
    _write_jsonl(corpus / "relations.jsonl", [{"relation": "requires_confirmation"}])
    _write_jsonl(
        corpus / "triples.jsonl",
        [
            {
                "triple_id": "t1",
                "head": "electrical fire",
                "relation": "requires_confirmation",
                "tail": "power isolation",
                "source_chunk_ids": ["evidence_chunk_001"],
            }
        ],
    )
    return corpus


def _cfg(corpus: Path) -> dict:
    return {
        "execution_stage": "dry_run",
        "unified_decision_output": True,
        "strict_decision_parse": False,
        "llm": {"provider": "heuristic", "temperature": 0.0, "max_tokens": 1024},
        "paths": {"corpus_dir": str(corpus)},
        "retrieval": {"top_k": 3},
        "dense_rag": {
            "backend": "smoke",
            "model_name": "smoke-hash-embedding",
            "model_version": "v0-smoke",
            "dimension": 64,
            "reject_smoke": False,
            "allow_index_rebuild": True,
        },
        "hybrid_rag": {"top_k": 3, "candidate_pool": 5, "reject_smoke": False},
        "ekell_style": {"prompt_dir": "configs/prompts/controlled", "neighborhood_k_hop": 1},
        "ekell_vector": {"backend": "smoke", "dimension": 32, "top_k": 4, "reject_smoke": False},
        "scenario_parser": {"use_llm": False},
        "normalization": {"infer_structured_safety_fields": False},
    }


def _scenario() -> dict:
    return {
        "case_id": "FBPUB_000001",
        "scenario_id": "FBPUB_000001",
        "scenario_text": "Electrical room fire with high smoke and unknown power status.",
        "dynamic_snapshots": [],
    }


def _valid_payload(**overrides) -> dict:
    base = {
        "decision": {
            "risk_signals": ["electrical_risk"],
            "risk_level": "high",
            "recommended_actions": [
                {
                    "action_id": "verify_power_isolation",
                    "text": "确认电源已经切断。",
                    "priority": "high",
                    "evidence_refs": [],
                }
            ],
            "blocked_actions": ["BLOCK_UNVERIFIED_WATER_SUPPRESSION"],
            "missing_confirmations": ["power_cutoff_status"],
            "human_review_required": True,
            "final_decision_gate": "await_human_confirmation",
        },
        "response": {
            "status": "awaiting_human_confirmation",
            "text": "当前存在电气风险，请先确认断电。",
            "citations": [],
        },
    }
    base.update(overrides)
    return base


# --- character normalization ---


def test_nfkc_normalization():
    assert normalize_identifier_characters("Ｅｌｅｃｔｒｉｃａｌ＿Ｒｉｓｋ", case="lower") == "electrical_risk"


def test_trims_whitespace():
    assert normalize_identifier_characters("  electrical_risk  ", case="lower") == "electrical_risk"


def test_spaces_to_underscores():
    assert normalize_identifier_characters("electrical risk", case="lower") == "electrical_risk"


def test_hyphens_to_underscores():
    assert normalize_identifier_characters("power-cutoff-unknown", case="lower") == "power_cutoff_unknown"


def test_collapses_multiple_underscores():
    assert normalize_identifier_characters("electrical__risk", case="lower") == "electrical_risk"


def test_risk_ids_are_lowercase():
    assert normalize_risk_signal(" Electrical-Risk ", strict=True) == "electrical_risk"


def test_blocked_action_ids_are_uppercase():
    assert (
        normalize_blocked_action_id("block execution", strict=True)
        == "BLOCK_REAL_WORLD_EXECUTION"
    )


# --- taxonomy validation ---


def test_valid_risk_signal_passes():
    assert normalize_risk_signal("electrical_risk", strict=True) == "electrical_risk"


def test_invalid_risk_signal_fails_formal():
    with pytest.raises(DecisionParseError, match="invalid_risk_signal"):
        parse_decision_output(
            _valid_payload(
                decision={
                    **_valid_payload()["decision"],
                    "risk_signals": ["not_a_real_signal"],
                }
            ),
            case_id="c1",
            method_id="direct_llm",
            strict=True,
        )


def test_valid_action_id_passes():
    assert normalize_action_id("verify_power_isolation", strict=True) == "verify_power_isolation"


def test_natural_language_action_id_fails():
    with pytest.raises(DecisionParseError, match="invalid_action_id"):
        parse_decision_output(
            _valid_payload(
                decision={
                    **_valid_payload()["decision"],
                    "recommended_actions": [
                        {
                            "action_id": "确认电源已经切断",
                            "text": "确认电源已经切断。",
                            "priority": "high",
                            "evidence_refs": [],
                        }
                    ],
                }
            ),
            case_id="c1",
            method_id="direct_llm",
            strict=True,
        )


def test_valid_blocked_action_passes():
    assert (
        normalize_blocked_action_id("BLOCK_UNVERIFIED_WATER_SUPPRESSION", strict=True)
        == "BLOCK_UNVERIFIED_WATER_SUPPRESSION"
    )


def test_chinese_blocked_action_text_fails():
    with pytest.raises(DecisionParseError, match="invalid_blocked_action_id"):
        parse_decision_output(
            _valid_payload(
                decision={
                    **_valid_payload()["decision"],
                    "blocked_actions": ["不要在没有断电时喷水"],
                }
            ),
            case_id="c1",
            method_id="direct_llm",
            strict=True,
        )


def test_valid_confirmation_passes():
    assert normalize_confirmation_id("power_cutoff_status", strict=True) == "power_cutoff_status"


def test_invalid_confirmation_fails():
    with pytest.raises(DecisionParseError, match="invalid_confirmation_id"):
        parse_decision_output(
            _valid_payload(
                decision={
                    **_valid_payload()["decision"],
                    "missing_confirmations": ["需要领导确认"],
                }
            ),
            case_id="c1",
            method_id="direct_llm",
            strict=True,
        )


# --- aliases ---


def test_exact_alias_maps_to_standard_id():
    assert normalize_risk_signal("active_or_suspected_fire", strict=True) == "fire_detected"
    assert normalize_action_id("ACT_ELECTRICAL_POWER_CHECK", strict=True) == "verify_power_isolation"


def test_alias_application_is_recorded():
    report = TaxonomyNormalizeReport()
    assert (
        normalize_risk_signal("high_smoke_confirmed", strict=True, report=report)
        == "high_smoke_detected"
    )
    assert report.aliases_applied
    assert report.aliases_applied[0]["target"] == "high_smoke_detected"


def test_unknown_alias_is_not_guessed():
    assert normalize_risk_signal("totally_unknown_signal_xyz", strict=False) is None


def test_alias_table_has_no_duplicate_sources():
    from external_baselines.common.firebench_taxonomy import DEV_ALIAS_KEYS, FORMAL_ALIAS_KEYS

    aliases = load_aliases()
    for key in FORMAL_ALIAS_KEYS + DEV_ALIAS_KEYS:
        block = aliases.get(key) or {}
        if not isinstance(block, dict):
            continue
        assert len(block) == len(set(block.keys())), key


def test_alias_table_targets_exist_in_taxonomy():
    report = validate_alias_table()
    assert report["ok"] is True
    assert report["all_targets_valid"] is True


# --- no semantic inference ---


def test_electrical_chinese_phrase_not_inferred_as_electrical_risk():
    assert normalize_risk_signal("现场可能有电", strict=False) is None


def test_do_not_spray_water_text_not_inferred_as_block_id():
    assert normalize_blocked_action_id("先不要喷水", strict=False) is None


def test_wait_for_leader_text_not_inferred_as_human_confirmation():
    assert normalize_confirmation_id("需要领导确认", strict=False) is None


def test_high_smoke_sentence_not_inferred_as_high_smoke_id():
    assert normalize_risk_signal("烟很大", strict=False) is None


# --- five methods ---


def _assert_taxonomy_row(out: dict) -> None:
    risks = out.get("key_risks") or []
    assert risks
    assert set(risks) <= membership_set("risk_signals")
    for action in out.get("recommended_actions") or []:
        assert isinstance(action, dict)
        assert action["action_id"] in membership_set("recommended_action_ids")
        assert isinstance(action.get("text"), str) and action["text"].strip()
    for blocked in out.get("blocked_actions") or []:
        assert blocked in membership_set("blocked_action_ids")
        assert blocked == blocked.upper()
    for conf in out.get("missing_confirmations") or []:
        assert conf in membership_set("confirmation_ids")
    assert out["final_response"]["text"].strip()


def test_direct_outputs_standard_taxonomy(tmp_path):
    out = run_direct(_scenario(), config=_cfg(_tiny_corpus(tmp_path)), llm=HeuristicLLMClient())
    _assert_taxonomy_row(out)


def test_bm25_outputs_standard_taxonomy(tmp_path):
    out = run_bm25(_scenario(), config=_cfg(_tiny_corpus(tmp_path)), llm=HeuristicLLMClient())
    _assert_taxonomy_row(out)


def test_dense_outputs_standard_taxonomy(tmp_path):
    out = run_dense(_scenario(), config=_cfg(_tiny_corpus(tmp_path)), llm=HeuristicLLMClient())
    _assert_taxonomy_row(out)


def test_hybrid_outputs_standard_taxonomy(tmp_path):
    out = run_hybrid(_scenario(), config=_cfg(_tiny_corpus(tmp_path)), llm=HeuristicLLMClient())
    _assert_taxonomy_row(out)


def test_ekell_outputs_standard_taxonomy(tmp_path):
    out = run_controlled_shared_llm(
        _scenario(), config=_cfg(_tiny_corpus(tmp_path)), llm=HeuristicLLMClient()
    )
    _assert_taxonomy_row(out)


def test_character_variants_normalize_identically():
    a = parse_decision_output(
        _valid_payload(
            decision={
                **_valid_payload()["decision"],
                "risk_signals": [" Electrical-Risk ", "high_smoke_confirmed"],
                "blocked_actions": ["rely_on_stale_route"],
            }
        ),
        case_id="c1",
        method_id="direct_llm",
        strict=True,
    )
    assert a.risk_signals == sort_by_taxonomy_order(
        ["electrical_risk", "high_smoke_detected"], "risk_signals"
    )
    assert a.blocked_actions == ["BLOCK_RELY_ON_STALE_ROUTE"]


# --- file output ---


def test_prediction_jsonl_is_utf8_without_bom(tmp_path):
    from external_baselines.common.decision_output import decision_output_to_interop

    out = parse_decision_output(_valid_payload(), case_id="c1", method_id="direct_llm", strict=True)
    path = tmp_path / "direct_llm.jsonl"
    payload = decision_output_to_interop(out)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "electrical_risk" in raw.decode("utf-8")


def test_prediction_jsonl_uses_double_quote_json(tmp_path):
    from external_baselines.common.decision_output import decision_output_to_interop

    out = parse_decision_output(_valid_payload(), case_id="c1", method_id="direct_llm", strict=True)
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps(decision_output_to_interop(out), ensure_ascii=False) + "\n", encoding="utf-8")
    line = path.read_text(encoding="utf-8").strip()
    assert line.startswith("{")
    assert "'" not in line or '"action_id"' in line
    json.loads(line)


def test_prediction_jsonl_has_one_record_per_line(tmp_path):
    from external_baselines.common.decision_output import decision_output_to_interop

    rows = [
        decision_output_to_interop(
            parse_decision_output(_valid_payload(), case_id=f"c{i}", method_id="direct_llm", strict=True)
        )
        for i in range(2)
    ]
    path = tmp_path / "x.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


def test_arrays_are_deduplicated():
    out = parse_decision_output(
        _valid_payload(
            decision={
                **_valid_payload()["decision"],
                "risk_signals": ["electrical_risk", "electrical_risk", "Electrical-Risk"],
            }
        ),
        case_id="c1",
        method_id="direct_llm",
        strict=True,
    )
    assert out.risk_signals.count("electrical_risk") == 1


def test_taxonomy_arrays_have_deterministic_order():
    tax = load_taxonomy()
    out = parse_decision_output(
        _valid_payload(
            decision={
                **_valid_payload()["decision"],
                "risk_signals": ["power_cutoff_unknown", "fire_detected", "electrical_risk"],
            }
        ),
        case_id="c1",
        method_id="direct_llm",
        strict=True,
    )
    order = {v: i for i, v in enumerate(tax["risk_signals"])}
    assert out.risk_signals == sorted(out.risk_signals, key=lambda x: order[x])


def test_action_order_is_preserved():
    out = parse_decision_output(
        _valid_payload(
            decision={
                **_valid_payload()["decision"],
                "recommended_actions": [
                    {
                        "action_id": "prepare_respiratory_protection",
                        "text": "a",
                        "priority": "high",
                        "evidence_refs": [],
                    },
                    {
                        "action_id": "verify_power_isolation",
                        "text": "b",
                        "priority": "high",
                        "evidence_refs": [],
                    },
                ],
            }
        ),
        case_id="c1",
        method_id="direct_llm",
        strict=True,
    )
    assert [a["action_id"] for a in out.recommended_actions] == [
        "prepare_respiratory_protection",
        "verify_power_isolation",
    ]


def test_taxonomy_snapshot_loads():
    tax = load_taxonomy()
    assert "electrical_risk" in tax["risk_signals"]
    assert "BLOCK_REAL_WORLD_EXECUTION" in tax["blocked_action_ids"]
    assert "verify_power_isolation" in membership_set("recommended_action_ids")
