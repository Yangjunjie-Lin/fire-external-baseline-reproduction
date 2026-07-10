"""Fidelity registry and claim-guard tests."""

from __future__ import annotations

import json
from pathlib import Path

from external_baselines.method_registry import (
    METHOD_REGISTRY,
    all_method_ids,
    canonicalize_method_id,
    main_table_methods,
    paper_fidelity_methods,
    resolve_pipeline,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs" / "fidelity" / "method_fidelity_matrix.json"


def test_all_registered_methods_have_method_card():
    assert MATRIX.exists()
    rows = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_id = {r["method_id"]: r for r in rows}
    for mid in all_method_ids():
        assert mid in by_id, mid
        card = by_id[mid]
        assert card.get("method_class")
        assert card.get("implementation_status")
        # Honest: no method is empirically validated in this matrix yet.
        assert "empirically" in str(card.get("empirical_status") or "").lower() or card.get("empirical_status")


def test_no_fallback_marked_actual():
    rows = json.loads(MATRIX.read_text(encoding="utf-8"))
    for row in rows:
        if row["method_id"] in {"lightrag", "microsoft_graphrag", "fallback_graph_retrieval"}:
            status = str(row.get("implementation_status") or "")
            assert "fallback" in status or "interface" in status or row.get("method_class") in {
                "fallback_only",
                "official_system_adapter",
            }
            assert "actual_complete" not in status


def test_enhanced_never_replaces_fidelity():
    assert "ekell_style_enhanced" not in main_table_methods()
    assert "ekell_style_enhanced" not in paper_fidelity_methods()
    assert canonicalize_method_id("ekell_style_faithful") == "ekell_style_controlled_shared_llm"
    assert canonicalize_method_id("ekell_style_legacy_bm25") == "ekell_style_legacy_bm25"


def test_legacy_ekell_not_main_table():
    assert "ekell_style_legacy_bm25" not in main_table_methods()
    entry = METHOD_REGISTRY["ekell_style_legacy_bm25"]
    assert entry["method_class"] == "legacy_diagnostic"
    assert entry["main_table"] is False


def test_pipelines_resolve():
    for mid in main_table_methods():
        fn = resolve_pipeline(mid)
        assert callable(fn)
