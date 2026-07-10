from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    step_id: int
    operation: str
    input_entities: set[str] = field(default_factory=set)
    relation: str | None = None
    results: set[str] = field(default_factory=set)
    supporting_triples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def intermediate_results(self) -> set[str]:
        return self.results

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "input_entities": sorted(self.input_entities),
            "relation": self.relation,
            "intermediate_results": sorted(self.results),
            "supporting_triples": list(self.supporting_triples),
        }
