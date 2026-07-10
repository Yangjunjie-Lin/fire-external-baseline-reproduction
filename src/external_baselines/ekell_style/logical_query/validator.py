from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from external_baselines.ekell_style.kg_loader import entity_aliases, triple_parts

from .parser import QueryParseError, parse_query
from .schema import QueryNode, ValidationResult, is_unknown


def _rows(kg: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(kg, Mapping):
        value = kg.get(name, [])
    else:
        value = getattr(kg, name, [])
    return [row for row in value if isinstance(row, dict)] if isinstance(value, Iterable) else []


def _triple_rows(kg: Any) -> list[dict[str, Any]]:
    if isinstance(kg, list):
        return [row for row in kg if isinstance(row, dict)]
    return _rows(kg, "triples")


def kg_vocabulary(kg: Any) -> tuple[set[str], set[str]]:
    triples = _triple_rows(kg)
    entities: set[str] = set()
    relations: set[str] = set()
    for triple in triples:
        head, relation, tail = triple_parts(triple)
        entities.update(value.casefold() for value in (head, tail) if value)
        if relation:
            relations.add(relation.casefold())

    for entity in _rows(kg, "entities"):
        entities.update(alias.casefold() for alias in entity_aliases(entity) if alias)
    for relation in _rows(kg, "relations"):
        for key in ("relation", "predicate", "name", "label", "relation_id", "id"):
            value = relation.get(key)
            if value not in (None, ""):
                relations.add(str(value).casefold())
    return entities, relations


def validate_query(query: Any, kg: Any) -> ValidationResult:
    try:
        plan = parse_query(query)
    except QueryParseError as exc:
        reason = str(exc)
        return ValidationResult(
            plan=None,
            valid=False,
            degraded=True,
            errors=(reason,),
            fallback_reason=reason,
        )

    known_entities, known_relations = kg_vocabulary(kg)
    errors: list[str] = []
    warnings: list[str] = []

    def visit(node: QueryNode, path: str) -> None:
        if node.operation == "projection":
            if not node.relation:
                errors.append(f"{path}: projection requires relation")
            elif is_unknown(node.relation):
                warnings.append(f"{path}: relation is explicitly unknown")
            elif node.relation.casefold() not in known_relations:
                errors.append(f"{path}: relation {node.relation!r} is not in the KG")

            has_entity = node.entity is not None
            has_operand = bool(node.operands)
            if has_entity == has_operand:
                errors.append(f"{path}: projection requires exactly one entity or one operand")
            if len(node.operands) > 1:
                errors.append(f"{path}: projection accepts at most one operand")
            if node.entity is not None:
                if is_unknown(node.entity):
                    warnings.append(f"{path}: entity is explicitly unknown")
                elif node.entity.casefold() not in known_entities:
                    errors.append(f"{path}: entity {node.entity!r} is not in the KG")
        elif node.operation in {"intersection", "union"}:
            if len(node.operands) < 2:
                errors.append(f"{path}: {node.operation} requires at least two operands")
            if node.entity is not None or node.relation is not None:
                errors.append(f"{path}: {node.operation} does not accept entity or relation")
        elif node.operation == "negation":
            if len(node.operands) != 1:
                errors.append(f"{path}: negation requires exactly one operand")
            if node.entity is not None or node.relation is not None:
                errors.append(f"{path}: negation does not accept entity or relation")

        for index, operand in enumerate(node.operands):
            visit(operand, f"{path}.operands[{index}]")

    visit(plan, "$")
    fallback_reason = "; ".join(errors) if errors else ("; ".join(warnings) if warnings else None)
    return ValidationResult(
        plan=plan,
        valid=not errors,
        degraded=bool(errors or warnings),
        errors=tuple(errors),
        warnings=tuple(warnings),
        fallback_reason=fallback_reason,
    )


validate_plan = validate_query
