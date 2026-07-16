"""Deterministic preservation of baseline-submitted retrieval contexts."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class ContextNormalizationError(ValueError):
    """Raised when retrieval context cannot be preserved without repair."""


@dataclass(frozen=True)
class NormalizedContexts:
    contexts: list[dict[str, Any]]
    native_count: int
    submitted_count: int
    truncated: bool
    handoff_top_k: int


def _first(context: dict[str, Any], names: tuple[str, ...]) -> tuple[bool, Any]:
    for name in names:
        if name in context:
            return True, context[name]
    return False, None


def normalize_retrieval_contexts(contexts: Any, *, top_k: int) -> NormalizedContexts:
    if type(top_k) is not int or top_k <= 0:
        raise ContextNormalizationError("handoff_top_k_must_be_positive_integer")
    if not isinstance(contexts, list):
        raise ContextNormalizationError("retrieved_contexts_must_be_array")
    raw = deepcopy(contexts)
    explicit_rank_flags: list[bool] = []
    ranks: list[int] = []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ContextNormalizationError(f"context_{index}_must_be_object")
        has_text, text = _first(item, ("text", "content", "document", "passage"))
        if not has_text or not isinstance(text, str) or not text.strip():
            raise ContextNormalizationError(f"context_{index}_text_must_be_non_empty_string")
        has_rank, rank = _first(item, ("rank", "retrieval_rank"))
        explicit_rank_flags.append(has_rank)
        if has_rank:
            if type(rank) is not int or rank <= 0:
                raise ContextNormalizationError(f"context_{index}_rank_must_be_positive_integer")
            ranks.append(rank)
        else:
            ranks.append(index)

        output: dict[str, Any] = {"text": text, "rank": ranks[-1]}
        for output_key, aliases in (
            ("source_id", ("source_id", "document_id")),
            ("chunk_id", ("chunk_id", "context_id", "evidence_id", "citation")),
            ("score", ("score", "retrieval_score", "similarity")),
        ):
            present, value = _first(item, aliases)
            if not present:
                continue
            if output_key in {"source_id", "chunk_id"} and value is not None and not isinstance(value, str):
                raise ContextNormalizationError(f"context_{index}_{output_key}_must_be_string_or_null")
            if output_key == "score" and value is not None:
                if type(value) not in (int, float) or not math.isfinite(float(value)):
                    raise ContextNormalizationError(f"context_{index}_score_must_be_finite_number_or_null")
            output[output_key] = value
        normalized.append(output)

    if any(explicit_rank_flags) and not all(explicit_rank_flags):
        raise ContextNormalizationError("explicit_rank_must_be_present_for_all_contexts")
    if all(explicit_rank_flags) and ranks:
        if len(ranks) != len(set(ranks)):
            raise ContextNormalizationError("explicit_ranks_must_be_unique")
        if any(current <= previous for previous, current in zip(ranks, ranks[1:])):
            raise ContextNormalizationError("explicit_ranks_must_be_strictly_increasing")

    submitted = normalized[:top_k]
    return NormalizedContexts(
        contexts=submitted,
        native_count=len(normalized),
        submitted_count=len(submitted),
        truncated=len(normalized) > len(submitted),
        handoff_top_k=top_k,
    )
