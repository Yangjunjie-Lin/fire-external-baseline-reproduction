from __future__ import annotations

from external_baselines.ekell_style.neighborhood_expander import NeighborhoodExpander


TRIPLES = [
    {
        "triple_id": "t1",
        "head": "A",
        "relation": "causes",
        "tail": "B",
        "source_chunk_ids": ["c1"],
    },
    {
        "triple_id": "t2",
        "head": "B",
        "relation": "requires",
        "tail": "C",
        "source_chunk_ids": ["c2"],
    },
    {
        "triple_id": "t3",
        "head": "C",
        "relation": "returns",
        "tail": "A",
        "source_chunk_ids": ["c3"],
    },
    {
        "triple_id": "t4",
        "head": "B",
        "relation": "unrelated",
        "tail": "D",
        "source_chunk_ids": ["c4"],
    },
]


def test_one_hop_and_provenance():
    result = NeighborhoodExpander(TRIPLES, k_hop=1).expand("A")
    assert set(result["triple_ids"]) == {"t1", "t3"}
    assert all(path["hop_count"] == 1 for path in result["paths"])
    for path in result["paths"]:
        assert path["path_id"]
        assert path["source_chunk_ids"]


def test_multi_hop_prevents_cycles():
    result = NeighborhoodExpander(TRIPLES, k_hop=4).expand("A")
    assert any(path["hop_count"] >= 2 for path in result["paths"])
    assert all(len(path["nodes"]) == len(set(path["nodes"])) for path in result["paths"])
    assert all(path["hop_count"] <= 3 for path in result["paths"])


def test_node_and_triple_budgets():
    result = NeighborhoodExpander(
        TRIPLES, k_hop=3, max_nodes=2, max_triples=1
    ).expand("A")
    assert len(result["nodes"]) <= 2
    assert len(result["triples"]) <= 1


def test_relation_filter():
    result = NeighborhoodExpander(
        TRIPLES, k_hop=3, relation_whitelist={"causes", "requires"}
    ).expand("A")
    assert result["triple_ids"] == ["t1", "t2"]
    assert all(
        relation in {"causes", "requires"}
        for path in result["paths"]
        for relation in path["relations"]
    )


def test_serialized_path_provenance_accumulates():
    result = NeighborhoodExpander(TRIPLES, k_hop=2).expand("A")
    path = next(path for path in result["paths"] if path["triple_ids"] == ["t1", "t2"])
    assert path == {
        "path_id": path["path_id"],
        "nodes": ["A", "B", "C"],
        "relations": ["causes", "requires"],
        "triple_ids": ["t1", "t2"],
        "source_chunk_ids": ["c1", "c2"],
        "hop_count": 2,
    }
