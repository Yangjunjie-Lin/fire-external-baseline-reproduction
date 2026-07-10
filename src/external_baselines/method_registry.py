from __future__ import annotations

"""Single source of truth for external baseline method IDs and pipeline wiring.

Do not import fire_agent_demo. Method aliases here supersede ad-hoc maps in
callers; keep runner/manifest consumers aligned with this registry.
"""

import importlib
from typing import Any, Callable

METHOD_REGISTRY: dict[str, dict[str, Any]] = {
    "direct_llm": {
        "method_id": "direct_llm",
        "aliases": [],
        "method_class": "canonical_baseline",
        "pipeline_import": "external_baselines.direct_llm.pipeline",
        "pipeline_attr": "run_scenario",
        "formal_track": "A_main_table",
        "main_table": True,
        "paper_fidelity_track": False,
        "supplemental": False,
        "legacy": False,
        "requires_corpus": False,
        "requires_real_embedding": False,
        "paper_fidelity": False,
        "fallback_only": False,
        "claim_label": "Strong no-retrieval LLM baseline (main table)",
    },
    "bm25_rag": {
        "method_id": "bm25_rag",
        "aliases": ["vanilla_rag"],
        "method_class": "canonical_baseline",
        "pipeline_import": "external_baselines.vanilla_rag.pipeline",
        "pipeline_attr": "run_scenario",
        "formal_track": "A_main_table",
        "main_table": True,
        "paper_fidelity_track": False,
        "supplemental": False,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": False,
        "paper_fidelity": False,
        "fallback_only": False,
        "claim_label": "True BM25 lexical RAG baseline (main table)",
    },
    "dense_rag": {
        "method_id": "dense_rag",
        "aliases": [],
        "method_class": "supplemental_extension",
        "pipeline_import": "external_baselines.dense_rag.pipeline",
        "pipeline_attr": "run_scenario",
        "formal_track": "A_supplemental",
        "main_table": False,
        "paper_fidelity_track": False,
        "supplemental": True,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": True,
        "paper_fidelity": False,
        "fallback_only": False,
        "claim_label": "Dense embedding RAG (supplemental; formal only with real embeddings)",
    },
    "hybrid_rag": {
        "method_id": "hybrid_rag",
        "aliases": [],
        "method_class": "supplemental_extension",
        "pipeline_import": "external_baselines.hybrid_rag.pipeline",
        "pipeline_attr": "run_scenario",
        "formal_track": "A_supplemental",
        "main_table": False,
        "paper_fidelity_track": False,
        "supplemental": True,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": True,
        "paper_fidelity": False,
        "fallback_only": False,
        "claim_label": "Hybrid BM25+dense RRF RAG (supplemental; formal only with real dense)",
    },
    "ekell_style_controlled_shared_llm": {
        "method_id": "ekell_style_controlled_shared_llm",
        "aliases": ["ekell", "ekell_style", "e-kell-style", "ekell_style_faithful"],
        "method_class": "paper_reproduction",
        "pipeline_import": "external_baselines.ekell_style.full_pipeline",
        "pipeline_attr": "run_controlled_shared_llm",
        "formal_track": "A_main_table",
        "main_table": True,
        "paper_fidelity_track": False,
        "supplemental": False,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": True,
        "paper_fidelity": False,
        "fallback_only": False,
        "claim_label": (
            "E-KELL-style paper-faithful pipeline-level reimplementation, "
            "not official reproduction (controlled shared-LLM main table)"
        ),
    },
    "ekell_style_paper_fidelity": {
        "method_id": "ekell_style_paper_fidelity",
        "aliases": [],
        "method_class": "paper_reproduction",
        "pipeline_import": "external_baselines.ekell_style.full_pipeline",
        "pipeline_attr": "run_paper_fidelity",
        "formal_track": "B_paper_fidelity",
        "main_table": False,
        "paper_fidelity_track": True,
        "supplemental": False,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": True,
        "paper_fidelity": True,
        "fallback_only": False,
        "claim_label": (
            "E-KELL-style paper-fidelity track (ChatGLM-6B interface); "
            "separate experiment, not main-table controlled comparison"
        ),
    },
    "ekell_style_enhanced": {
        "method_id": "ekell_style_enhanced",
        "aliases": [],
        "method_class": "supplemental_extension",
        "pipeline_import": "external_baselines.ekell_style.enhanced_pipeline",
        "pipeline_attr": "run_scenario_enhanced",
        "formal_track": "A_supplemental",
        "main_table": False,
        "paper_fidelity_track": False,
        "supplemental": True,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": False,
        "paper_fidelity": False,
        "fallback_only": False,
        "claim_label": "E-KELL-style enhanced hooks (supplemental; never replaces controlled/paper-fidelity)",
    },
    "ekell_style_legacy_bm25": {
        "method_id": "ekell_style_legacy_bm25",
        "aliases": [],
        "method_class": "legacy_diagnostic",
        "pipeline_import": "external_baselines.ekell_style.pipeline",
        "pipeline_attr": "run_legacy_bm25",
        "formal_track": "legacy_diagnostic",
        "main_table": False,
        "paper_fidelity_track": False,
        "supplemental": False,
        "legacy": True,
        "requires_corpus": True,
        "requires_real_embedding": False,
        "paper_fidelity": False,
        "fallback_only": False,
        "claim_label": "Legacy BM25+3-stage E-KELL scaffold (diagnostics only; not aliased to controlled)",
    },
    "lightrag": {
        "method_id": "lightrag",
        "aliases": [],
        "method_class": "official_system_adapter",
        "pipeline_import": "external_baselines.graphrag_adapter.lightrag_adapter",
        "pipeline_attr": "run_scenario",
        "formal_track": "adapter_fallback",
        "main_table": False,
        "paper_fidelity_track": False,
        "supplemental": False,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": False,
        "paper_fidelity": False,
        "fallback_only": True,
        "claim_label": "LightRAG official-system adapter (fallback_only unless actual index+query)",
    },
    "microsoft_graphrag": {
        "method_id": "microsoft_graphrag",
        "aliases": ["graphrag"],
        "method_class": "official_system_adapter",
        "pipeline_import": "external_baselines.graphrag_adapter.microsoft_graphrag_adapter",
        "pipeline_attr": "run_scenario",
        "formal_track": "adapter_fallback",
        "main_table": False,
        "paper_fidelity_track": False,
        "supplemental": False,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": False,
        "paper_fidelity": False,
        "fallback_only": True,
        "claim_label": "Microsoft GraphRAG official-system adapter (fallback_only unless actual index+query)",
    },
    "fallback_graph_retrieval": {
        "method_id": "fallback_graph_retrieval",
        "aliases": [],
        "method_class": "fallback_only",
        "pipeline_import": "external_baselines.graphrag_adapter.fallback_graph_retrieval",
        "pipeline_attr": "run_scenario",
        "formal_track": "adapter_fallback",
        "main_table": False,
        "paper_fidelity_track": False,
        "supplemental": False,
        "legacy": False,
        "requires_corpus": True,
        "requires_real_embedding": False,
        "paper_fidelity": False,
        "fallback_only": True,
        "claim_label": "Local KG subgraph fallback retrieval (never actual GraphRAG leaderboard)",
    },
}


def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for method_id, entry in METHOD_REGISTRY.items():
        index[method_id] = method_id
        for alias in entry.get("aliases") or []:
            index[str(alias).strip().lower()] = method_id
    return index


_ALIAS_TO_CANONICAL = _alias_index()


def method_id_aliases() -> dict[str, str]:
    """Alias → canonical map derived from METHOD_REGISTRY (compatibility export)."""
    return {
        alias: canonical
        for alias, canonical in _ALIAS_TO_CANONICAL.items()
        if alias != canonical
    }


def canonicalize_method_id(method: str) -> str:
    mid = str(method or "").strip().lower()
    return _ALIAS_TO_CANONICAL.get(mid, mid)


def get_method(method_id: str) -> dict[str, Any]:
    canonical = canonicalize_method_id(method_id)
    entry = METHOD_REGISTRY.get(canonical)
    if entry is None:
        raise KeyError(f"Unknown method_id: {method_id!r} (canonical={canonical!r})")
    return dict(entry)


def main_table_methods() -> tuple[str, ...]:
    return tuple(
        mid for mid, entry in METHOD_REGISTRY.items() if entry.get("main_table")
    )


def supplemental_methods() -> tuple[str, ...]:
    return tuple(
        mid
        for mid, entry in METHOD_REGISTRY.items()
        if entry.get("supplemental") or entry.get("method_class") == "supplemental_extension"
    )


def paper_fidelity_methods() -> tuple[str, ...]:
    return tuple(
        mid for mid, entry in METHOD_REGISTRY.items() if entry.get("paper_fidelity_track")
    )


def legacy_methods() -> tuple[str, ...]:
    return tuple(mid for mid, entry in METHOD_REGISTRY.items() if entry.get("legacy"))


def fallback_methods() -> tuple[str, ...]:
    return tuple(mid for mid, entry in METHOD_REGISTRY.items() if entry.get("fallback_only"))


def resolve_pipeline(method_id: str) -> Callable[..., Any]:
    entry = get_method(method_id)
    module = importlib.import_module(str(entry["pipeline_import"]))
    attr = getattr(module, str(entry["pipeline_attr"]))
    if not callable(attr):
        raise TypeError(
            f"Pipeline attribute {entry['pipeline_attr']!r} on "
            f"{entry['pipeline_import']!r} is not callable"
        )
    return attr


def all_method_ids() -> list[str]:
    return list(METHOD_REGISTRY.keys())
