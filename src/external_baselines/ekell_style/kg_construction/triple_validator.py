from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .schema import ENTITY_TYPES, RELATION_TYPES, KGTriple


@dataclass(frozen=True)
class TripleValidation:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_triple(value: KGTriple | Mapping[str, Any]) -> TripleValidation:
    try:
        triple = value if isinstance(value, KGTriple) else KGTriple.from_mapping(value)
    except (TypeError, ValueError) as exc:
        return TripleValidation(False, (str(exc),))
    errors: list[str] = []
    warnings: list[str] = []
    for name in ("triple_id", "subject", "predicate", "object", "source_id", "chunk_id", "source_text", "extraction_method"):
        if not str(getattr(triple, name)).strip():
            errors.append(f"{name} must be non-empty")
    if triple.predicate not in RELATION_TYPES:
        errors.append(f"predicate {triple.predicate!r} is not in schema")
    if not 0.0 <= triple.confidence <= 1.0:
        errors.append("confidence must be between 0 and 1")
    if triple.review_status not in {"candidate", "approved", "rejected"}:
        errors.append("review_status must be candidate, approved, or rejected")
    for field_name in ("subject_type", "object_type"):
        entity_type = getattr(triple, field_name)
        if entity_type and entity_type not in ENTITY_TYPES:
            errors.append(f"{field_name} {entity_type!r} is not in schema")
    if triple.review_status == "candidate":
        warnings.append("candidate triple has not been human-approved")
    return TripleValidation(not errors, tuple(errors), tuple(warnings))


def validate_triples(values: Iterable[KGTriple | Mapping[str, Any]]) -> list[TripleValidation]:
    return [validate_triple(value) for value in values]
