from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from external_baselines.ekell_style.kg_loader import triple_parts

from .schema import ExecutionResult, QueryNode
from .trace import TraceStep
from .validator import validate_query


def _triples_from(kg: Any) -> list[dict[str, Any]]:
    if isinstance(kg, list):
        rows = kg
    elif isinstance(kg, Mapping):
        rows = kg.get("triples", [])
    else:
        rows = getattr(kg, "triples", [])
    if not isinstance(rows, Iterable):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _contains_negation(node: QueryNode) -> bool:
    return node.operation == "negation" or any(_contains_negation(child) for child in node.operands)


class FOLExecutor:
    def __init__(
        self,
        kg: Any,
        *,
        candidate_universe: Iterable[str] | None = None,
    ) -> None:
        self.kg = kg
        self.triples = _triples_from(kg)
        self.candidate_universe = (
            {str(entity) for entity in candidate_universe}
            if candidate_universe is not None
            else None
        )

    def execute(
        self,
        query: Any,
        *,
        candidate_universe: Iterable[str] | None = None,
    ) -> ExecutionResult:
        validation = validate_query(query, self.kg)
        if not validation.valid or validation.plan is None:
            return ExecutionResult(
                degraded=True,
                errors=list(validation.errors),
                fallback_reason=validation.fallback_reason,
            )

        universe = (
            {str(entity) for entity in candidate_universe}
            if candidate_universe is not None
            else self.candidate_universe
        )
        if _contains_negation(validation.plan) and universe is None:
            reason = "negation requires an explicit candidate universe"
            return ExecutionResult(degraded=True, errors=[reason], fallback_reason=reason)

        trace: list[TraceStep] = []

        def evaluate(node: QueryNode) -> tuple[set[str], list[dict[str, Any]]]:
            child_values = [evaluate(child) for child in node.operands]
            child_sets = [value[0] for value in child_values]
            child_support = [triple for value in child_values for triple in value[1]]
            inputs: set[str] = set().union(*child_sets) if child_sets else set()
            supporting: list[dict[str, Any]] = []

            if node.operation == "projection":
                inputs = child_sets[0] if child_sets else ({node.entity} if node.entity else set())
                input_keys = {entity.casefold() for entity in inputs}
                relation_key = (node.relation or "").casefold()
                results: set[str] = set()
                for triple in self.triples:
                    head, relation, tail = triple_parts(triple)
                    if head.casefold() in input_keys and relation.casefold() == relation_key:
                        results.add(tail)
                        supporting.append(dict(triple))
            elif node.operation == "intersection":
                results = set.intersection(*child_sets)
                supporting = [
                    triple
                    for triple in child_support
                    if triple_parts(triple)[2] in results
                ]
            elif node.operation == "union":
                results = set().union(*child_sets)
                supporting = child_support
            else:
                results = set(universe or ()) - child_sets[0]
                supporting = child_support

            trace.append(
                TraceStep(
                    step_id=len(trace) + 1,
                    operation=node.operation,
                    input_entities=set(inputs),
                    relation=node.relation,
                    results=set(results),
                    supporting_triples=supporting,
                )
            )
            return results, supporting

        results, _ = evaluate(validation.plan)
        return ExecutionResult(
            results=results,
            trace=trace,
            degraded=validation.degraded,
            fallback_reason=validation.fallback_reason,
        )


def execute_query(
    query: Any,
    kg: Any,
    *,
    candidate_universe: Iterable[str] | None = None,
) -> ExecutionResult:
    return FOLExecutor(kg, candidate_universe=candidate_universe).execute(query)


execute_fol = execute_query
