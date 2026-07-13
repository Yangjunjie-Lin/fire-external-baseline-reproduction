"""Dependency audit for complete E-KELL pipelines (paper fidelity + controlled).

Faithful/complete E-KELL may use ekell_style.vector_retriever and logical_query.
It must NOT use generic dense_rag/hybrid_rag baselines, enhanced_pipeline, or target modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "external_baselines"
FULL = SRC / "ekell_style" / "full_pipeline.py"

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

ALLOWED_EKELL_NATIVE = (
    "external_baselines.ekell_style.vector_retriever",
    "external_baselines.ekell_style.logical_query",
    "external_baselines.ekell_style.neighborhood_expander",
    "external_baselines.ekell_style.stepwise_prompt_chain",
    "external_baselines.ekell_style.embedding_backends",
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


def _complete_ekell_modules() -> list[Path]:
    d = SRC / "ekell_style"
    paths = [
        d / "full_pipeline.py",
        d / "vector_retriever.py",
        d / "vector_index.py",
        d / "embedding_backends.py",
        d / "neighborhood_expander.py",
        d / "stepwise_prompt_chain.py",
        d / "scenario_parser.py",
        d / "kg_loader.py",
    ]
    paths.extend((d / "logical_query").glob("*.py"))
    return [p for p in paths if p.exists()]


def test_complete_ekell_has_no_forbidden_imports():
    offenders: list[str] = []
    for path in _complete_ekell_modules():
        for name in _imported_names(path):
            lower = name.lower()
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad in lower:
                    offenders.append(f"{path.name}: import {name}")
    assert not offenders, "Forbidden imports:\n" + "\n".join(offenders)


def test_full_pipeline_uses_ekell_native_vector_not_generic_dense():
    text = FULL.read_text(encoding="utf-8")
    imports = _imported_names(FULL)
    assert "from external_baselines.dense_rag" not in text
    assert "from external_baselines.hybrid_rag" not in text
    assert not any("enhanced_pipeline" in name for name in imports)
    assert not any("dense_rag" in name for name in imports)
    assert not any("hybrid_rag" in name for name in imports)
    assert not any("fire_agent_demo" in name for name in imports)
    assert "vector_retriever" in text
    assert "logical_query" in text
    assert "neighborhood_expander" in text
    assert "stepwise_prompt_chain" in text


def test_enhanced_is_separate_module():
    enhanced = SRC / "ekell_style" / "enhanced_pipeline.py"
    assert enhanced.exists()
    text = enhanced.read_text(encoding="utf-8")
    assert "dense_rag" in text
    assert "supplemental_extended" in text


def test_main_table_and_fidelity_tracks():
    from external_baselines.common.experiment_manifest import (
        MAIN_TABLE_METHODS,
        PAPER_FIDELITY_METHODS,
        SUPPLEMENTAL_METHODS,
    )

    assert MAIN_TABLE_METHODS == ("direct_llm", "bm25_rag", "ekell_style_controlled_shared_llm")
    assert PAPER_FIDELITY_METHODS == ("ekell_style_paper_fidelity",)
    assert "ekell_style_enhanced" in SUPPLEMENTAL_METHODS
    assert "ekell_style_paper_fidelity" not in MAIN_TABLE_METHODS


def test_ci_fidelity_audit_accepts_runtime_vector_retriever_wiring():
    from scripts.audit.audit_ekell_fidelity import build_audit

    audit = build_audit()
    vector_check = next(item for item in audit["checks"] if item["id"] == "vector_retriever")
    assert vector_check["wired_in_pipeline"] is True
    assert audit["all_pass"] is True
