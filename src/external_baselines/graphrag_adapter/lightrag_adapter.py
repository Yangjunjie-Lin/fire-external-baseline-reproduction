from __future__ import annotations

from typing import Any

from external_baselines.graphrag_adapter.fallback_graph_retrieval import run_scenario as fallback_run

METHOD = "lightrag"
EXTERNAL_REPOSITORY = "https://github.com/HKUDS/LightRAG"
REQUIREMENTS_FOR_REAL_REPRODUCTION = [
    "Install and record an exact LightRAG package/repository version.",
    "Build a LightRAG index from the frozen fire corpus.",
    "Run the official LightRAG query path over the same scenario matrix.",
    "Record index configuration, package commit/version, prompt/config files, and output checksums.",
]


def is_available() -> bool:
    try:
        import lightrag  # noqa: F401
        return True
    except Exception:
        return False


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm=None) -> dict[str, Any]:
    """Transparent LightRAG adapter stub.

    This function does not claim complete LightRAG reproduction. If strict mode
    is disabled, it falls back to local graph/text retrieval and records the
    deviation. If strict mode is enabled and the package is present but not
    configured, it raises a clear error rather than silently overclaiming.
    """
    config = config or {}
    package_available = is_available()
    strict = bool(config.get("graphrag_adapters", {}).get("strict_external_package", False))
    workspace_configured = bool(config.get("graphrag_adapters", {}).get("lightrag_index_path"))
    if strict and package_available and not workspace_configured:
        raise RuntimeError(
            "LightRAG package is installed but no LightRAG index/query integration is configured. "
            "Set graphrag_adapters.strict_external_package=false for fallback mode, or configure lightrag_index_path and actual query integration."
        )

    result = fallback_run(scenario, config=config, llm=llm, method=METHOD)
    ms = result.setdefault("method_specific", {})
    ms.update({
        "external_repository": EXTERNAL_REPOSITORY,
        "actual_external_package_used": False,
        "fallback_retrieval_used": True,
        "indexing_performed": False,
        "query_performed": False,
        "adapter_status": "actual_package_available_but_not_configured_used_fallback" if package_available else "lightrag_not_installed_used_fallback",
        "reproduction_status": "adapter_stub_not_official_reproduction",
        "requirements_for_real_reproduction": REQUIREMENTS_FOR_REAL_REPRODUCTION,
        "deviation_from_official_system": "No official LightRAG indexing/query pipeline is executed by this adapter yet.",
    })
    return result
