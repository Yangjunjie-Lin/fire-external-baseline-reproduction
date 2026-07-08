from __future__ import annotations

from typing import Any

from external_baselines.graphrag_adapter.fallback_graph_retrieval import run_scenario as fallback_run

METHOD = "microsoft_graphrag"
EXTERNAL_REPOSITORY = "https://github.com/microsoft/graphrag"
REQUIREMENTS_FOR_REAL_REPRODUCTION = [
    "Install and record an exact Microsoft GraphRAG package/repository version.",
    "Create an official GraphRAG workspace from the frozen fire corpus.",
    "Run indexing with archived settings and prompts.",
    "Run the official query workflow for the same scenario matrix.",
    "Record workspace configuration, package commit/version, index artifacts, and output checksums.",
]


def is_available() -> bool:
    try:
        import graphrag  # noqa: F401
        return True
    except Exception:
        return False


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm=None) -> dict[str, Any]:
    """Transparent Microsoft GraphRAG adapter stub."""
    config = config or {}
    package_available = is_available()
    strict = bool(config.get("graphrag_adapters", {}).get("strict_external_package", False))
    workspace_configured = bool(config.get("graphrag_adapters", {}).get("microsoft_graphrag_workspace"))
    if strict and package_available and not workspace_configured:
        raise RuntimeError(
            "Microsoft GraphRAG package is installed but no workspace/index/query integration is configured. "
            "Set graphrag_adapters.strict_external_package=false for fallback mode, or configure microsoft_graphrag_workspace and actual query integration."
        )

    result = fallback_run(scenario, config=config, llm=llm, method=METHOD)
    ms = result.setdefault("method_specific", {})
    ms.update({
        "external_repository": EXTERNAL_REPOSITORY,
        "actual_external_package_used": False,
        "fallback_retrieval_used": True,
        "indexing_performed": False,
        "query_performed": False,
        "adapter_status": "actual_package_available_but_workspace_not_configured_used_fallback" if package_available else "microsoft_graphrag_not_installed_used_fallback",
        "reproduction_status": "adapter_stub_not_official_reproduction",
        "requirements_for_real_reproduction": REQUIREMENTS_FOR_REAL_REPRODUCTION,
        "deviation_from_official_system": "No official Microsoft GraphRAG workspace/index/query output is executed by this adapter yet.",
    })
    return result
