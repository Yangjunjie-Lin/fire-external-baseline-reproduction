from __future__ import annotations

import json
from typing import Any

from external_baselines.common.llm_client import LLMClient

from .parser import QueryParseError, parse_query
from .schema import DecompositionResult
from .validator import kg_vocabulary, validate_query

SYSTEM_PROMPT = """Convert a natural-language KG question to one constrained JSON AST.
Return only a JSON object. Do not return prose, Markdown, code, or executable expressions.
Allowed operations and aliases are projection/p, intersection/i, union/u, and negation/n.
A projection has a relation and either an entity or one operand.
Intersection and union have an operands array with at least two AST objects.
Negation has exactly one operand and is evaluated against a separately supplied universe.
Use the exact string "unknown" when an entity or relation cannot be grounded."""


def _prompt(question: str, kg: Any) -> str:
    entities, relations = kg_vocabulary(kg)
    schema_example = {
        "operation": "intersection",
        "operands": [
            {"operation": "projection", "entity": "entity_a", "relation": "relation_a"},
            {"operation": "projection", "entity": "entity_b", "relation": "relation_b"},
        ],
    }
    return (
        f"Question:\n{question}\n\n"
        f"Known entities:\n{json.dumps(sorted(entities), ensure_ascii=False)}\n\n"
        f"Known relations:\n{json.dumps(sorted(relations), ensure_ascii=False)}\n\n"
        f"AST shape example:\n{json.dumps(schema_example, ensure_ascii=False)}"
    )


def decompose_query(
    question: str,
    *,
    llm: LLMClient,
    kg: Any,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> DecompositionResult:
    raw = llm.complete(
        system=SYSTEM_PROMPT,
        user=_prompt(question, kg),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        plan = parse_query(raw)
    except QueryParseError as exc:
        reason = str(exc)
        return DecompositionResult(
            plan=None,
            degraded=True,
            errors=(reason,),
            fallback_reason=reason,
            raw_output=raw,
        )

    validation = validate_query(plan, kg)
    return DecompositionResult(
        plan=plan,
        degraded=validation.degraded,
        errors=validation.errors,
        fallback_reason=validation.fallback_reason,
        raw_output=raw,
    )


class LogicalQueryDecomposer:
    def __init__(
        self,
        llm: LLMClient,
        kg: Any,
        *,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> None:
        self.llm = llm
        self.kg = kg
        self.temperature = temperature
        self.max_tokens = max_tokens

    def decompose(self, question: str) -> DecompositionResult:
        return decompose_query(
            question,
            llm=self.llm,
            kg=self.kg,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
