from __future__ import annotations

"""Budgeted, provenance-preserving KG neighborhood expansion."""

from collections import defaultdict, deque
from typing import Any, Iterable, Sequence

from external_baselines.common.checksums import sha256_json
from external_baselines.ekell_style.kg_loader import FireKG, triple_id, triple_parts


def _as_ids(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _source_chunk_ids(row: dict[str, Any]) -> list[str]:
    for key in (
        "source_chunk_ids",
        "evidence_chunk_ids",
        "chunk_ids",
        "source_chunk_id",
        "evidence_chunk_id",
        "chunk_id",
    ):
        if key in row:
            return _as_ids(row[key])
    return []


class NeighborhoodExpander:
    def __init__(
        self,
        triples: FireKG | Iterable[dict[str, Any]] | None = None,
        *,
        k_hop: int = 1,
        relation_whitelist: Iterable[str] | None = None,
        max_nodes: int = 100,
        max_triples: int = 100,
        max_node_budget: int | None = None,
        max_triple_budget: int | None = None,
    ) -> None:
        if k_hop < 0:
            raise ValueError("k_hop must be non-negative.")
        self.k_hop = k_hop
        self.relation_whitelist = (
            {str(relation) for relation in relation_whitelist}
            if relation_whitelist is not None
            else None
        )
        self.max_nodes = max_node_budget if max_node_budget is not None else max_nodes
        self.max_triples = max_triple_budget if max_triple_budget is not None else max_triples
        if self.max_nodes < 0 or self.max_triples < 0:
            raise ValueError("Neighborhood budgets must be non-negative.")
        self.triples = self._coerce_triples(triples)

    @staticmethod
    def _coerce_triples(
        triples: FireKG | Iterable[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if triples is None:
            return []
        if isinstance(triples, FireKG):
            triples = triples.triples
        return [dict(row) for row in triples]

    def expand(
        self,
        seed_nodes: str | Sequence[str],
        *,
        triples: FireKG | Iterable[dict[str, Any]] | None = None,
        k_hop: int | None = None,
        relation_whitelist: Iterable[str] | None = None,
        max_nodes: int | None = None,
        max_triples: int | None = None,
    ) -> dict[str, Any]:
        hops = self.k_hop if k_hop is None else k_hop
        if hops < 0:
            raise ValueError("k_hop must be non-negative.")
        node_budget = self.max_nodes if max_nodes is None else max_nodes
        triple_budget = self.max_triples if max_triples is None else max_triples
        if node_budget < 0 or triple_budget < 0:
            raise ValueError("Neighborhood budgets must be non-negative.")
        whitelist = (
            self.relation_whitelist
            if relation_whitelist is None
            else {str(relation) for relation in relation_whitelist}
        )
        rows = self.triples if triples is None else self._coerce_triples(triples)
        seeds = [str(seed_nodes)] if isinstance(seed_nodes, str) else [str(node) for node in seed_nodes]
        seeds = list(dict.fromkeys(node for node in seeds if node))
        selected_nodes: list[str] = seeds[:node_budget]
        selected_node_set = set(selected_nodes)

        adjacency: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = defaultdict(list)
        normalized_rows: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(rows):
            head, relation, tail = triple_parts(raw)
            if not head or not tail or (whitelist is not None and relation not in whitelist):
                continue
            tid = triple_id(raw, index)
            row = {
                **raw,
                "triple_id": tid,
                "head": head,
                "relation": relation,
                "tail": tail,
                "source_chunk_ids": _source_chunk_ids(raw),
            }
            normalized_rows[tid] = row
            # Expansion is neighborhood-based, so an edge can be traversed from
            # either endpoint while retaining its original triple orientation.
            adjacency[head].append((tail, relation, tid, row))
            adjacency[tail].append((head, relation, tid, row))
        for edges in adjacency.values():
            edges.sort(key=lambda edge: (edge[1], edge[2], edge[0]))

        # Queue entries hold the complete simple path, preventing cycles locally.
        queue: deque[tuple[str, list[str], list[str], list[str], list[str]]] = deque(
            (seed, [seed], [], [], []) for seed in selected_nodes
        )
        selected_triple_ids: list[str] = []
        selected_triple_set: set[str] = set()
        paths: list[dict[str, Any]] = []
        seen_path_signatures: set[tuple[str, ...]] = set()

        while queue:
            current, path_nodes, relations, triple_ids, source_ids = queue.popleft()
            if len(triple_ids) >= hops:
                continue
            for neighbor, relation, tid, row in adjacency.get(current, []):
                if neighbor in path_nodes:
                    continue
                is_new_triple = tid not in selected_triple_set
                if is_new_triple and len(selected_triple_ids) >= triple_budget:
                    continue
                is_new_node = neighbor not in selected_node_set
                if is_new_node and len(selected_nodes) >= node_budget:
                    continue

                if is_new_triple:
                    selected_triple_set.add(tid)
                    selected_triple_ids.append(tid)
                if is_new_node:
                    selected_node_set.add(neighbor)
                    selected_nodes.append(neighbor)

                next_nodes = path_nodes + [neighbor]
                next_relations = relations + [relation]
                next_triples = triple_ids + [tid]
                next_sources = list(
                    dict.fromkeys(source_ids + _source_chunk_ids(row))
                )
                signature = tuple(next_triples)
                if signature not in seen_path_signatures:
                    seen_path_signatures.add(signature)
                    path_payload = {
                        "nodes": next_nodes,
                        "relations": next_relations,
                        "triple_ids": next_triples,
                        "source_chunk_ids": next_sources,
                        "hop_count": len(next_triples),
                    }
                    path_payload["path_id"] = "path_" + sha256_json(path_payload)[:16]
                    paths.append(path_payload)
                queue.append(
                    (neighbor, next_nodes, next_relations, next_triples, next_sources)
                )

        selected_triples = [normalized_rows[tid] for tid in selected_triple_ids]
        return {
            "seed_nodes": seeds,
            "nodes": selected_nodes,
            "triples": selected_triples,
            "triple_ids": selected_triple_ids,
            "paths": paths,
            "k_hop": hops,
            "relation_whitelist": sorted(whitelist) if whitelist is not None else None,
            "budgets": {"max_nodes": node_budget, "max_triples": triple_budget},
            "budget_exhausted": {
                "nodes": len(selected_nodes) >= node_budget,
                "triples": len(selected_triple_ids) >= triple_budget,
            },
        }


def expand_neighborhood(
    triples: FireKG | Iterable[dict[str, Any]],
    seed_nodes: str | Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Functional convenience API around :class:`NeighborhoodExpander`."""

    return NeighborhoodExpander(triples, **kwargs).expand(seed_nodes)
