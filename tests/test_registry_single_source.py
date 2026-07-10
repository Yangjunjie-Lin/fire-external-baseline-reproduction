"""Registry single-source-of-truth and CLI/manifest consistency tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from external_baselines.method_registry import (  # noqa: E402
    METHOD_REGISTRY,
    canonicalize_method_id,
    fallback_methods,
    legacy_methods,
    main_table_methods,
    method_id_aliases,
    paper_fidelity_methods,
    supplemental_methods,
)
from external_baselines.registry import methods as registry_facade  # noqa: E402


def test_registry_is_single_source_of_truth():
    assert METHOD_REGISTRY is registry_facade.METHOD_REGISTRY
    assert main_table_methods() == ("direct_llm", "bm25_rag", "ekell_style_controlled_shared_llm")
    assert paper_fidelity_methods() == ("ekell_style_paper_fidelity",)
    assert "ekell_style_enhanced" in supplemental_methods()
    assert "lightrag" in fallback_methods()
    assert "ekell_style_legacy_bm25" in legacy_methods()


def test_all_method_ids_unique():
    assert len(METHOD_REGISTRY) == len(set(METHOD_REGISTRY.keys()))


def test_aliases_unique():
    aliases = method_id_aliases()
    assert len(aliases) == len(set(aliases.keys()))
    for alias, canonical in aliases.items():
        assert alias != canonical
        assert canonical in METHOD_REGISTRY


def test_alias_does_not_shadow_canonical_method():
    for mid in METHOD_REGISTRY:
        assert mid not in method_id_aliases()


def test_main_table_exact():
    assert set(main_table_methods()) == {
        "direct_llm",
        "bm25_rag",
        "ekell_style_controlled_shared_llm",
    }


def test_paper_fidelity_separate():
    assert "ekell_style_paper_fidelity" not in main_table_methods()
    assert "ekell_style_paper_fidelity" in paper_fidelity_methods()


def test_legacy_not_main_table():
    assert "ekell_style_legacy_bm25" not in main_table_methods()


def test_fallback_not_main_table():
    for mid in fallback_methods():
        assert mid not in main_table_methods()


def test_all_cli_methods_exist_in_registry():
    scripts = list((ROOT / "scripts").glob("*.py"))
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        for match in re.findall(r"--methods\s+([a-z0-9_,\-]+)", text):
            for token in match.split(","):
                token = token.strip()
                if token and token not in {"help"}:
                    canonicalize_method_id(token)  # raises if unknown after alias


def test_all_manifest_methods_exist_in_registry():
    for path in (ROOT / "configs" / "experiments").glob("*.yaml*"):
        if "deprecated" in path.read_text(encoding="utf-8")[:200]:
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "method_id:" in line:
                mid = line.split(":", 1)[1].strip()
                if mid and not mid.startswith("path/"):
                    canonicalize_method_id(mid)


def test_all_registry_methods_have_fidelity_entries():
    matrix = json.loads((ROOT / "docs/fidelity/method_fidelity_matrix.json").read_text(encoding="utf-8"))
    ids = {entry["method_id"] for entry in matrix}
    for mid in METHOD_REGISTRY:
        assert mid in ids, f"missing fidelity entry for {mid}"


def test_all_method_cards_exist_in_registry():
    cards = (ROOT / "docs/baseline_method_cards.md").read_text(encoding="utf-8")
    for mid in main_table_methods():
        assert mid in cards
