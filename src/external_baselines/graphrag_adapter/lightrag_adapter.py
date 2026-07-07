from __future__ import annotations

from typing import Any

from external_baselines.graphrag_adapter.fallback_graph_retrieval import run_scenario as fallback_run

METHOD = "lightrag"
EXTERNAL_REPOSITORY = "https://github.com/HKUDS/LightRAG"


def is_available() -> bool:
    try:
        import lightrag  # noqa: F401
        return True
    except Exception:
        return False


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm=None) -> dict[str, Any]:
    """LightRAG adapter with explicit fallback status."""
    package_available = is_available()
    result = fallback_run(scenario, config=config, llm=llm, method=METHOD)
    ms = result.setdefault("method_specific", {})
    ms.update({"adapter_status": "actual_package_available_but_official_indexing_not_configured_used_fallback" if package_available else "lightrag_not_installed_used_fallback", "actual_external_package_used": False, "fallback_retrieval_used": True, "indexing_performed": False, "external_repository": EXTERNAL_REPOSITORY, "deviation_from_official_system": "No official LightRAG indexing/query pipeline is executed by this adapter yet."})
    return result
