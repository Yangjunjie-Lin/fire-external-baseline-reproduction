from __future__ import annotations

from typing import Any

from external_baselines.graphrag_adapter.fallback_graph_retrieval import run_scenario as fallback_run


METHOD = "microsoft_graphrag"


def is_available() -> bool:
    try:
        import graphrag  # noqa: F401
        return True
    except Exception:
        return False


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm=None) -> dict[str, Any]:
    """Microsoft GraphRAG adapter with explicit fallback.

    This project keeps Microsoft GraphRAG as an optional dependency and does not
    copy the repository. Full indexing requires a separate GraphRAG workspace.
    """
    if not is_available():
        result = fallback_run(scenario, config=config, llm=llm, method=METHOD)
        result.setdefault("method_specific", {})["adapter_status"] = "microsoft_graphrag_not_installed_used_fallback"
        result["method_specific"]["external_repository"] = "https://github.com/microsoft/graphrag"
        return result
    raise NotImplementedError(
        "Microsoft GraphRAG is installed, but a GraphRAG workspace/index is not configured. "
        "Use fallback mode or configure the official CLI outputs externally."
    )
