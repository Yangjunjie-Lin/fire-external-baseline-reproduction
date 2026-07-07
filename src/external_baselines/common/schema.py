from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .text_utils import as_list


@dataclass
class RetrievedContext:
    context_id: str
    text: str
    source_id: str | None = None
    citation: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineOutput:
    scenario_id: str
    method: str
    situation_summary: str = ""
    key_risks: list[Any] = field(default_factory=list)
    recommended_actions: list[Any] = field(default_factory=list)
    blocked_or_unsafe_actions: list[Any] = field(default_factory=list)
    missing_confirmations: list[Any] = field(default_factory=list)
    supporting_evidence: list[Any] = field(default_factory=list)
    citations: list[Any] = field(default_factory=list)
    final_decision_gate: str = "not_applicable_or_not_provided"
    retrieved_contexts: list[Any] = field(default_factory=list)
    latency_sec: float = 0.0
    raw_output: dict[str, Any] = field(default_factory=dict)
    method_specific: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_response_payload(payload: dict[str, Any], *, scenario_id: str, method: str) -> BaselineOutput:
    """Map flexible model responses into the unified output schema."""
    return BaselineOutput(
        scenario_id=scenario_id,
        method=method,
        situation_summary=str(payload.get("situation_summary") or payload.get("summary") or ""),
        key_risks=[str(x) for x in as_list(payload.get("key_risks") or payload.get("risks"))],
        recommended_actions=[str(x) for x in as_list(payload.get("recommended_actions") or payload.get("actions"))],
        blocked_or_unsafe_actions=[
            str(x) for x in as_list(payload.get("blocked_or_unsafe_actions") or payload.get("unsafe_actions"))
        ],
        missing_confirmations=[
            str(x) for x in as_list(payload.get("missing_confirmations") or payload.get("information_gaps"))
        ],
        supporting_evidence=[str(x) for x in as_list(payload.get("supporting_evidence") or payload.get("evidence"))],
        citations=[str(x) for x in as_list(payload.get("citations"))],
        final_decision_gate=str(payload.get("final_decision_gate") or payload.get("decision_gate") or "not_applicable_or_not_provided"),
        raw_output=payload,
    )


def retrieved_context_to_dict(ctx: RetrievedContext | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ctx, RetrievedContext):
        return asdict(ctx)
    return dict(ctx)
