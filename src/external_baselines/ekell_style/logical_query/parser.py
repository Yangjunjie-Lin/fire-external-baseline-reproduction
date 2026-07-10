from __future__ import annotations

import json
from typing import Any, Mapping

from .schema import QueryNode, node_from_mapping


class QueryParseError(ValueError):
    """Raised when a logical query is not constrained AST JSON."""


def parse_query(value: str | Mapping[str, Any] | QueryNode) -> QueryNode:
    if isinstance(value, QueryNode):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise QueryParseError(f"invalid query JSON: {exc.msg}") from exc
    elif isinstance(value, Mapping):
        decoded = value
    else:
        raise QueryParseError("query must be a JSON object, mapping, or QueryNode")

    if not isinstance(decoded, Mapping):
        raise QueryParseError("query JSON must have an object at its root")
    try:
        return node_from_mapping(decoded)
    except ValueError as exc:
        raise QueryParseError(str(exc)) from exc


parse_logical_query = parse_query
