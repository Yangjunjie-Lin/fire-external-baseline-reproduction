from __future__ import annotations

"""Paper-final and claim guards for external baselines.

These guards prevent overclaiming: heuristic smoke runs cannot be marked as
paper_final; fallback GraphRAG cannot be marked as actual GraphRAG.
"""

from typing import Any


class ConfigGuardError(ValueError):
    """Raised when a final-paper or actual-package claim is invalid."""


def _llm_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    llm = config.get("llm", config)
    return dict(llm) if isinstance(llm, dict) else {}


def assert_paper_final_allowed(config: dict[str, Any] | None = None) -> None:
    """Reject paper_final=true when provider is heuristic or model/version missing."""
    config = config or {}
    if not bool(config.get("paper_final", False)):
        return
    llm = _llm_cfg(config)
    provider = str(llm.get("provider", "heuristic")).lower()
    model = str(llm.get("model") or "").strip()
    model_version = str(llm.get("model_version") or llm.get("version") or "").strip()
    if provider in {"heuristic", "local", "smoke", ""}:
        raise ConfigGuardError(
            "paper_final=true is rejected for heuristic/smoke LLM providers. "
            "Use a real shared model config for final paper runs."
        )
    if not model:
        raise ConfigGuardError("paper_final=true requires llm.model to be set.")
    if not model_version:
        raise ConfigGuardError(
            "paper_final=true requires llm.model_version (or llm.version) for reproducibility."
        )
    if not config.get("bundle_checksum") and not config.get("paths", {}).get("bundle_checksum"):
        # Bundle checksum may be supplied at interop runtime; warn via hard fail only when
        # require_bundle_checksum is explicitly true.
        if bool(config.get("require_bundle_checksum", False)):
            raise ConfigGuardError("paper_final=true requires bundle_checksum when require_bundle_checksum=true.")


def assert_actual_graphrag_allowed(method_specific: dict[str, Any] | None = None, *, claim_actual: bool = False) -> None:
    """Reject actual_graphrag=true when fallback retrieval was used."""
    if not claim_actual:
        return
    ms = method_specific or {}
    if not bool(ms.get("actual_external_package_used")):
        raise ConfigGuardError(
            "actual_graphrag=true is rejected because actual_external_package_used is false."
        )
    if bool(ms.get("fallback_retrieval_used")):
        raise ConfigGuardError(
            "actual_graphrag=true is rejected because fallback_retrieval_used is true."
        )
    if not bool(ms.get("indexing_performed")) or not bool(ms.get("query_performed")):
        raise ConfigGuardError(
            "actual_graphrag=true requires indexing_performed=true and query_performed=true."
        )


def method_leaderboard_eligibility(method_id: str, method_specific: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify whether a method row may enter formal vs smoke/fallback leaderboards."""
    ms = method_specific or {}
    mid = method_id.lower().strip()
    actual = bool(ms.get("actual_external_package_used"))
    fallback = bool(ms.get("fallback_retrieval_used"))
    reproduction_level = str(ms.get("reproduction_level") or ms.get("fidelity_level") or "unknown")

    if mid in {"lightrag", "microsoft_graphrag", "graphrag"}:
        if actual and not fallback:
            return {
                "formal_leaderboard": True,
                "actual_graphrag_leaderboard": True,
                "smoke_or_fallback_only": False,
                "reason": "actual external package indexing+query completed",
            }
        return {
            "formal_leaderboard": False,
            "actual_graphrag_leaderboard": False,
            "smoke_or_fallback_only": True,
            "reason": "fallback_only or incomplete actual package path",
        }

    if mid == "fallback_graph_retrieval":
        return {
            "formal_leaderboard": False,
            "actual_graphrag_leaderboard": False,
            "smoke_or_fallback_only": True,
            "reason": "explicit fallback method; never enters actual GraphRAG leaderboard",
        }

    if mid in {"dense_rag", "hybrid_rag"} and ms.get("embedding_backend") in {None, "unavailable", "smoke_fixture"}:
        if ms.get("dense_index_built") is False or ms.get("method_status") == "smoke_fixture_only":
            return {
                "formal_leaderboard": False,
                "actual_graphrag_leaderboard": False,
                "smoke_or_fallback_only": True,
                "reason": "dense/hybrid without real embedding index; smoke/fixture only",
                "reproduction_level": reproduction_level,
            }

    return {
        "formal_leaderboard": True,
        "actual_graphrag_leaderboard": False,
        "smoke_or_fallback_only": False,
        "reason": "architecture-allowed baseline eligible for system-level Track A",
        "reproduction_level": reproduction_level,
    }
