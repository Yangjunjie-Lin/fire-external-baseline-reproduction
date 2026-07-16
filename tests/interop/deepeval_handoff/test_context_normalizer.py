from __future__ import annotations

import pytest

from external_baselines.interop.deepeval_handoff.context_normalizer import (
    ContextNormalizationError,
    normalize_retrieval_contexts,
)


def test_aliases_order_rank_and_top_k_prefix_are_preserved() -> None:
    result = normalize_retrieval_contexts(
        [
            {"content": "first", "document_id": "doc-1", "similarity": 0.9},
            {"passage": "second", "context_id": "chunk-2", "similarity": 0.8},
            {"text": "third"},
        ],
        top_k=2,
    )
    assert result.contexts == [
        {"text": "first", "rank": 1, "source_id": "doc-1", "score": 0.9},
        {"text": "second", "rank": 2, "chunk_id": "chunk-2", "score": 0.8},
    ]
    assert result.native_count == 3
    assert result.submitted_count == 2
    assert result.truncated is True


@pytest.mark.parametrize(
    "contexts",
    [
        [{"text": "a", "rank": 0}],
        [{"text": "a", "rank": True}],
        [{"text": "a", "rank": "1"}],
        [{"text": "a", "rank": 2}, {"text": "b", "rank": 1}],
        [{"text": "a", "rank": 1}, {"text": "b", "rank": 1}],
        [{"text": "a", "rank": 1}, {"text": "b"}],
    ],
)
def test_invalid_explicit_ranks_fail_closed(contexts: list[dict]) -> None:
    with pytest.raises(ContextNormalizationError):
        normalize_retrieval_contexts(contexts, top_k=5)
