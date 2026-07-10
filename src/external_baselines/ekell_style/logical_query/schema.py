from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Operation = Literal["projection", "intersection", "union", "negation"]

OPERATION_ALIASES: dict[str, Operation] = {
    "p": "projection",
    "projection": "projection",
    "i": "intersection",
    "intersection": "intersection",
    "u": "union",
    "union": "union",
    "n": "negation",
    "negation": "negation",
}

UNKNOWN_MARKERS = frozenset({"unknown", "__unknown__", "[unknown]"})


def is_unknown(value: str | None) -> bool:
    return value is not None and value.strip().casefold() in UNKNOWN_MARKERS


@dataclass(frozen=True)
class QueryNode:
    operation: Operation
    operands: tuple["QueryNode", ...] = ()
    entity: str | None = None
    relation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"operation": self.operation}
        if self.entity is not None:
            value["entity"] = self.entity
        if self.relation is not None:
            value["relation"] = self.relation
        if self.operands:
            value["operands"] = [operand.to_dict() for operand in self.operands]
        return value


@dataclass(frozen=True)
class ValidationResult:
    plan: QueryNode | None
    valid: bool
    degraded: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    fallback_reason: str | None = None


@dataclass(frozen=True)
class DecompositionResult:
    plan: QueryNode | None
    degraded: bool = False
    errors: tuple[str, ...] = ()
    fallback_reason: str | None = None
    raw_output: str | None = None


@dataclass
class ExecutionResult:
    results: set[str] = field(default_factory=set)
    trace: list[Any] = field(default_factory=list)
    degraded: bool = False
    errors: list[str] = field(default_factory=list)
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": sorted(self.results),
            "trace": [
                step.to_dict() if hasattr(step, "to_dict") else step
                for step in self.trace
            ],
            "degraded": self.degraded,
            "errors": list(self.errors),
            "fallback_reason": self.fallback_reason,
        }


def normalize_operation(value: object) -> Operation:
    if not isinstance(value, str):
        raise ValueError("operation must be a string")
    operation = OPERATION_ALIASES.get(value.strip().casefold())
    if operation is None:
        raise ValueError(f"unsupported operation: {value!r}")
    return operation


def node_from_mapping(value: Mapping[str, Any]) -> QueryNode:
    allowed = {"operation", "operands", "operand", "entity", "relation"}
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unexpected AST fields: {', '.join(sorted(extra))}")

    operation = normalize_operation(value.get("operation"))
    raw_operands = value.get("operands")
    if raw_operands is None and "operand" in value:
        raw_operands = [value["operand"]]
    if raw_operands is None:
        raw_operands = []
    if not isinstance(raw_operands, list):
        raise ValueError("operands must be a list")

    operands: list[QueryNode] = []
    for operand in raw_operands:
        if not isinstance(operand, Mapping):
            raise ValueError("each operand must be an AST object")
        operands.append(node_from_mapping(operand))

    entity = value.get("entity")
    relation = value.get("relation")
    if entity is not None and not isinstance(entity, str):
        raise ValueError("entity must be a string")
    if relation is not None and not isinstance(relation, str):
        raise ValueError("relation must be a string")
    return QueryNode(
        operation=operation,
        operands=tuple(operands),
        entity=entity,
        relation=relation,
    )
