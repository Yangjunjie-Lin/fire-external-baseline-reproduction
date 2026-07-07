from __future__ import annotations

from typing import Any

from external_baselines.graphrag_adapter.fallback_graph_retrieval import run_scenario as fallback_run


METHOD = "lightrag"


def is_available() -> bool:
    try:
        import lightrag  # noqa: F401
        return True
    except Exception:
        return False


def run_scenario(scenario: dict[str, Any], *, config: dict[str, Any] | None = None, llm=None) -> dict[str, Any]:
    """LightRAG adapter.

    This project does not vendor HKUDS/LightRAG. If the external package is not
    installed/configured, this returns a compatible fallback result and records
    the deviation explicitly.
    """
    if not is_available():
        result = fallback_run(scenario, config=config, llm=llm, method=METHOD)
        result.setdefault("method_specific", {})["adapter_status"] = "lightrag_not_installed_used_fallback"
        result["method_specific"]["external_repository"] = "https://github.com/HKUDS/LightRAG"
        return result
    raise NotImplementedError(
        "LightRAG is installed, but project-specific indexing/query wiring has not been configured. "
        "Use fallback mode or extend this adapter without vendoring the external repository."
    )
