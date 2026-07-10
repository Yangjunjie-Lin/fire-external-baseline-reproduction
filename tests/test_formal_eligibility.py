"""Formal eligibility guard tests."""

from __future__ import annotations

from external_baselines.common.guards import method_leaderboard_eligibility


def test_fallback_graphrag_never_formal():
    for mid in ("lightrag", "microsoft_graphrag", "fallback_graph_retrieval"):
        result = method_leaderboard_eligibility(
            mid,
            {
                "actual_external_package_used": False,
                "fallback_retrieval_used": True,
                "indexing_performed": False,
            },
        )
        assert result["formal_leaderboard"] is False
        assert result["smoke_or_fallback_only"] is True


def test_dense_smoke_never_formal():
    result = method_leaderboard_eligibility(
        "dense_rag",
        {"embedding_backend": "smoke_hash_embedding", "method_status": "smoke_fixture_only"},
    )
    assert result["formal_leaderboard"] is False


def test_hybrid_smoke_never_formal():
    result = method_leaderboard_eligibility(
        "hybrid_rag",
        {"embedding_backend": "hash", "actual_embedding_used": False},
    )
    assert result["formal_leaderboard"] is False


def test_controlled_ekell_smoke_vector_never_formal():
    result = method_leaderboard_eligibility(
        "ekell_style_controlled_shared_llm",
        {"smoke_fallback_used": True, "embedding_backend": "deterministic_hash_smoke"},
    )
    assert result["formal_leaderboard"] is False


def test_formal_eligibility_requires_reproducibility_metadata():
    result = method_leaderboard_eligibility(
        "direct_llm",
        {"llm_provider": "heuristic"},
    )
    assert result["formal_leaderboard"] is False

    ok = method_leaderboard_eligibility(
        "direct_llm",
        {"llm_provider": "siliconflow", "model_version": "v1"},
    )
    assert ok["formal_leaderboard"] is True
