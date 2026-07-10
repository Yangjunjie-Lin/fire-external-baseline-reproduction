"""AST dependency audit for ekell_style_faithful."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "external_baselines"
FAITHFUL = SRC / "ekell_style" / "pipeline.py"

FORBIDDEN_SUBSTRINGS = (
    "dense_rag",
    "hybrid_rag",
    "enhanced_pipeline",
    "fire_agent_demo",
    "safe_router",
    "safety_checker",
    "dynamic_reg",
    "hitl",
)


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def _local_ekell_modules() -> list[Path]:
    d = SRC / "ekell_style"
    return [
        d / "pipeline.py",
        d / "scenario_parser.py",
        d / "entity_matcher.py",
        d / "kg_loader.py",
        d / "subgraph_retriever.py",
        d / "prompt_chain.py",
    ]


def test_faithful_pipeline_has_no_forbidden_imports():
    offenders: list[str] = []
    for path in _local_ekell_modules():
        for name in _imported_names(path):
            lower = name.lower()
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad in lower:
                    offenders.append(f"{path.name}: import {name}")
    assert not offenders, "Forbidden imports in faithful closure:\n" + "\n".join(offenders)


def test_faithful_source_text_has_no_dense_import():
    text = FAITHFUL.read_text(encoding="utf-8")
    assert "from external_baselines.dense_rag" not in text
    assert "import external_baselines.dense_rag" not in text
    assert "from external_baselines.hybrid_rag" not in text
    assert "enhanced_pipeline" not in text
    assert "fire_agent_demo" not in text
    assert "embedding_scorer=None" in text


def test_enhanced_is_separate_module():
    enhanced = SRC / "ekell_style" / "enhanced_pipeline.py"
    assert enhanced.exists()
    text = enhanced.read_text(encoding="utf-8")
    assert "dense_rag" in text
    assert 'paper_table_role": "supplemental_extended"' in text or "supplemental_extended" in text


def test_main_table_methods_constant():
    from external_baselines.common.experiment_manifest import MAIN_TABLE_METHODS, SUPPLEMENTAL_METHODS

    assert MAIN_TABLE_METHODS == ("direct_llm", "bm25_rag", "ekell_style_faithful")
    assert "ekell_style_enhanced" in SUPPLEMENTAL_METHODS
    assert "dense_rag" in SUPPLEMENTAL_METHODS
    assert "hybrid_rag" in SUPPLEMENTAL_METHODS
