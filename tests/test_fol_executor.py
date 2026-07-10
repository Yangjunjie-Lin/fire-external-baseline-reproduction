from __future__ import annotations

import pytest

from external_baselines.ekell_style.logical_query import (
    QueryParseError,
    execute_query,
    parse_query,
    validate_query,
)

TRIPLES = [
    {"head": "fire", "relation": "requires", "tail": "evacuate", "id": "t1"},
    {"head": "fire", "relation": "requires", "tail": "alarm", "id": "t2"},
    {"subject": "smoke", "predicate": "requires", "object": "evacuate", "id": "t3"},
    {"head": "evacuate", "relation": "protects", "tail": "people", "id": "t4"},
]
KG = {"triples": TRIPLES}


def projection(entity: str, relation: str = "requires") -> dict[str, str]:
    return {"operation": "p", "entity": entity, "relation": relation}


def test_projection() -> None:
    result = execute_query(projection("fire"), KG)
    assert result.results == {"evacuate", "alarm"}


def test_intersection() -> None:
    query = {"operation": "i", "operands": [projection("fire"), projection("smoke")]}
    assert execute_query(query, KG).results == {"evacuate"}


def test_union() -> None:
    query = {"operation": "union", "operands": [projection("fire"), projection("smoke")]}
    assert execute_query(query, KG).results == {"evacuate", "alarm"}


def test_negation_uses_explicit_universe() -> None:
    query = {"operation": "n", "operands": [projection("smoke")]}
    result = execute_query(query, KG, candidate_universe={"evacuate", "alarm", "wait"})
    assert result.results == {"alarm", "wait"}

    missing_universe = execute_query(query, KG)
    assert missing_universe.degraded
    assert not missing_universe.results


def test_nested_expression() -> None:
    query = {
        "operation": "projection",
        "relation": "protects",
        "operands": [
            {
                "operation": "intersection",
                "operands": [projection("fire"), projection("smoke")],
            }
        ],
    }
    assert execute_query(query, KG).results == {"people"}


def test_invalid_relation_rejected() -> None:
    validation = validate_query(projection("fire", "invented_relation"), KG)
    assert not validation.valid
    assert validation.degraded
    assert "not in the KG" in validation.errors[0]

    execution = execute_query(projection("fire", "invented_relation"), KG)
    assert execution.degraded
    assert execution.results == set()


def test_unknown_entity_is_explicitly_degraded() -> None:
    result = execute_query(projection("unknown"), KG)
    assert result.degraded
    assert result.results == set()
    assert result.fallback_reason


def test_logical_trace_supports_results() -> None:
    result = execute_query(projection("fire"), KG)
    assert result.trace
    step = result.trace[-1]
    assert step.results == result.results
    supported_tails = {triple.get("tail") or triple.get("object") for triple in step.supporting_triples}
    assert result.results <= supported_tails
    assert step.input_entities == {"fire"}
    assert step.relation == "requires"


def test_parser_never_executes_arbitrary_code(tmp_path) -> None:
    marker = tmp_path / "executed"
    payload = f'__import__("pathlib").Path({str(marker)!r}).write_text("bad")'
    with pytest.raises(QueryParseError):
        parse_query(payload)
    assert not marker.exists()
