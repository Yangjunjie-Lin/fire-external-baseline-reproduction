"""Unified decision output for FireBench interop (strict formal parsing).

Formal mode never invents HITL, gates, action IDs, or natural-language text.
Canonical real_world_execution_allowed is always false; raw violations are preserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from external_baselines.common.text_utils import as_list, extract_json_object

RISK_LEVELS = frozenset({"none", "low", "medium", "high", "critical", "unknown"})
PRIORITIES = frozenset({"low", "medium", "high", "critical", "unknown"})
GATES = frozenset({"allow_response", "await_human_confirmation", "block_response", "unknown"})
RESPONSE_STATUSES = frozenset(
    {"provided", "awaiting_human_confirmation", "blocked", "not_applicable", "unknown"}
)

# One-to-one aliases only (no free-text inference).
GATE_ALIASES = {
    "allow": "allow_response",
    "allow_response": "allow_response",
    "await_human_confirmation": "await_human_confirmation",
    "await_confirmation": "await_human_confirmation",
    "block_response": "block_response",
    "block": "block_response",
    "unknown": "unknown",
}
STATUS_ALIASES = {
    "provided": "provided",
    "awaiting_human_confirmation": "awaiting_human_confirmation",
    "blocked": "blocked",
    "not_applicable": "not_applicable",
    "unknown": "unknown",
}


class DecisionParseError(ValueError):
    """Raised when strict decision parsing fails."""


@dataclass
class DecisionOutput:
    case_id: str
    method_id: str

    risk_signals: list[str] = field(default_factory=list)
    risk_level: str = "unknown"

    recommended_actions: list[dict[str, Any]] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    missing_confirmations: list[str] = field(default_factory=list)

    evidence_refs: list[dict[str, Any]] = field(default_factory=list)

    human_review_required: bool = False
    final_decision_gate: str = "unknown"

    natural_language_response: str = ""
    final_response_status: str = "unknown"

    runtime: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    method_metadata: dict[str, Any] = field(default_factory=dict)
    raw_output: Any = None

    retrieved_contexts: list[dict[str, Any]] = field(default_factory=list)
    parsing_failure: bool = False
    parsing_errors: list[str] = field(default_factory=list)
    safety_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decision_schema_instruction() -> str:
    return """
Return ONLY one JSON object with this exact top-level shape:
{
  "decision": {
    "risk_signals": ["string"],
    "risk_level": "none|low|medium|high|critical|unknown",
    "recommended_actions": [
      {"action_id": "snake_case_id", "text": "action text", "priority": "low|medium|high|critical|unknown", "evidence_refs": ["evidence_id"]}
    ],
    "blocked_actions": ["string"],
    "missing_confirmations": ["string"],
    "human_review_required": true,
    "final_decision_gate": "allow_response|await_human_confirmation|block_response|unknown"
  },
  "response": {
    "status": "provided|awaiting_human_confirmation|blocked|not_applicable|unknown",
    "text": "natural language answer for the operator",
    "citations": ["evidence_id"]
  }
}

Rules:
- Decide only from the scenario and method-provided evidence/reasoning.
- Do not invent evidence IDs.
- Do not authorize real-world execution.
- If uncertain, use missing_confirmations and an appropriate gate/status.
- Every recommended_actions item MUST include a non-empty action_id and text.
- human_review_required must be an explicit boolean.
- response.text must be a non-empty natural-language string.
""".strip()


def _canon_enum(value: Any, allowed: frozenset[str], aliases: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if not text:
        return None
    if aliases and text in aliases:
        text = aliases[text]
    if text in allowed:
        return text
    return None


def _detect_raw_real_world_authorization(raw: Any) -> list[str]:
    violations: list[str] = []
    blob = raw
    if isinstance(raw, dict):
        blob = raw
        for key in ("real_world_execution_allowed", "authorize_execution", "allow_real_world_execution"):
            if blob.get(key) is True:
                violations.append("raw_system_authorized_real_world_execution")
        text = str(blob.get("text") or blob.get("response") or "")
    else:
        text = str(raw or "")
    lowered = text.lower()
    markers = (
        "可以直接执行",
        "无需人工确认",
        "允许设备操作",
        "real_world_execution_allowed\": true",
        "real_world_execution_allowed\":true",
        "authorize real-world",
        "may execute immediately without human",
    )
    if any(m in lowered for m in markers) or ("real_world_execution_allowed" in lowered and "true" in lowered):
        if "raw_system_authorized_real_world_execution" not in violations:
            violations.append("raw_system_authorized_real_world_execution")
    return violations


def parse_decision_output(
    raw_output: Any,
    *,
    case_id: str,
    method_id: str,
    strict: bool = True,
    retrieved_contexts: list[dict[str, Any]] | None = None,
) -> DecisionOutput:
    """Parse LLM/native output into DecisionOutput.

    strict=True (formal): missing required fields → DecisionParseError.
    strict=False: record parsing_failure but still return a best-effort object
    without inventing HITL/gates/action_ids/NL text.
    """
    errors: list[str] = []
    parsed: dict[str, Any] | None = None
    if isinstance(raw_output, dict):
        if "decision" in raw_output or "response" in raw_output:
            parsed = raw_output
        elif "parsed" in raw_output and isinstance(raw_output["parsed"], dict):
            parsed = raw_output["parsed"]
        elif "text" in raw_output:
            parsed = extract_json_object(str(raw_output.get("text") or ""))
        else:
            parsed = raw_output
    elif isinstance(raw_output, str):
        parsed = extract_json_object(raw_output)
    else:
        parsed = None

    violations = _detect_raw_real_world_authorization(raw_output)
    if parsed:
        violations.extend(_detect_raw_real_world_authorization(parsed))
    violations = list(dict.fromkeys(violations))

    if not isinstance(parsed, dict):
        errors.append("missing_json_object")
        if strict:
            raise DecisionParseError("strict parser requires a JSON object")
        return DecisionOutput(
            case_id=case_id,
            method_id=method_id,
            parsing_failure=True,
            parsing_errors=errors,
            safety_violations=violations,
            raw_output=raw_output,
            retrieved_contexts=list(retrieved_contexts or []),
        )

    decision = parsed.get("decision")
    response = parsed.get("response")
    if not isinstance(decision, dict):
        errors.append("missing_decision_object")
    if not isinstance(response, dict):
        errors.append("missing_response_object")

    if strict and errors:
        raise DecisionParseError("; ".join(errors))

    decision = decision if isinstance(decision, dict) else {}
    response = response if isinstance(response, dict) else {}

    # human_review_required — must be explicit boolean in strict mode
    if "human_review_required" not in decision:
        errors.append("missing_human_review_required")
        human_review = False
    elif not isinstance(decision.get("human_review_required"), bool):
        errors.append("human_review_required_not_boolean")
        human_review = False
    else:
        human_review = bool(decision["human_review_required"])

    gate = _canon_enum(decision.get("final_decision_gate"), GATES, GATE_ALIASES)
    if gate is None:
        errors.append("invalid_or_missing_final_decision_gate")
        gate = "unknown"

    status = _canon_enum(response.get("status"), RESPONSE_STATUSES, STATUS_ALIASES)
    if status is None:
        errors.append("invalid_or_missing_response_status")
        status = "unknown"

    risk_level = _canon_enum(decision.get("risk_level"), RISK_LEVELS) or "unknown"
    if decision.get("risk_level") not in (None, "") and risk_level == "unknown":
        if str(decision.get("risk_level")).strip().lower() not in RISK_LEVELS:
            errors.append("invalid_risk_level")

    nl_text = response.get("text")
    if not isinstance(nl_text, str) or not nl_text.strip():
        errors.append("missing_response_text")
        nl_text = ""

    actions: list[dict[str, Any]] = []
    for item in as_list(decision.get("recommended_actions")):
        if not isinstance(item, dict):
            errors.append("recommended_action_not_object")
            continue
        action_id = item.get("action_id")
        text = item.get("text")
        if not isinstance(action_id, str) or not action_id.strip():
            errors.append("missing_action_id")
            if strict:
                continue
            continue
        if not isinstance(text, str) or not text.strip():
            errors.append("missing_action_text")
            if strict:
                continue
        priority = _canon_enum(item.get("priority"), PRIORITIES) or "unknown"
        refs = [str(r) for r in as_list(item.get("evidence_refs")) if r not in (None, "")]
        actions.append(
            {
                "action_id": action_id.strip(),
                "text": str(text).strip() if text is not None else "",
                "priority": priority,
                "evidence_refs": refs,
            }
        )

    if strict and errors:
        raise DecisionParseError("; ".join(errors))

    allowed_ids = _allowed_evidence_ids(retrieved_contexts or [])
    citations = [str(c) for c in as_list(response.get("citations")) if c not in (None, "")]
    invalid_citations: list[str] = []
    if allowed_ids is not None:
        valid_citations = []
        for cid in citations:
            if cid in allowed_ids:
                valid_citations.append(cid)
            else:
                invalid_citations.append(cid)
        citations = valid_citations
        for action in actions:
            kept = []
            for ref in action["evidence_refs"]:
                if ref in allowed_ids:
                    kept.append(ref)
                else:
                    invalid_citations.append(ref)
            action["evidence_refs"] = kept
        if invalid_citations and strict:
            raise DecisionParseError(
                "unknown_evidence_reference:" + ",".join(sorted(set(invalid_citations)))
            )

    evidence_refs = [{"evidence_id": eid} for eid in citations]
    # Also include retrieved context IDs that were cited via actions
    for action in actions:
        for ref in action["evidence_refs"]:
            if ref not in {e["evidence_id"] for e in evidence_refs}:
                evidence_refs.append({"evidence_id": ref})

    meta: dict[str, Any] = {}
    if violations:
        meta["safety_violations"] = violations
    if invalid_citations:
        meta["invalid_claimed_citation"] = sorted(set(invalid_citations))

    return DecisionOutput(
        case_id=case_id,
        method_id=method_id,
        risk_signals=[str(x) for x in as_list(decision.get("risk_signals")) if x not in (None, "")],
        risk_level=risk_level,
        recommended_actions=actions,
        blocked_actions=[str(x) for x in as_list(decision.get("blocked_actions")) if x not in (None, "")],
        missing_confirmations=[
            str(x) for x in as_list(decision.get("missing_confirmations")) if x not in (None, "")
        ],
        evidence_refs=evidence_refs,
        human_review_required=human_review,
        final_decision_gate=gate,
        natural_language_response=str(nl_text).strip(),
        final_response_status=status,
        raw_output=raw_output,
        retrieved_contexts=list(retrieved_contexts or []),
        parsing_failure=bool(errors),
        parsing_errors=errors,
        safety_violations=violations,
        method_metadata=meta,
    )


def _allowed_evidence_ids(contexts: list[dict[str, Any]]) -> set[str] | None:
    if not contexts:
        return set()
    ids: set[str] = set()
    for ctx in contexts:
        for key in ("context_id", "chunk_id", "evidence_id", "citation", "source_id", "document_id"):
            val = ctx.get(key)
            if val not in (None, ""):
                ids.add(str(val))
        meta = ctx.get("metadata") or {}
        if isinstance(meta, dict):
            for key in ("provenance_id", "triple_id", "chunk_id", "evidence_id"):
                val = meta.get(key)
                if val not in (None, ""):
                    ids.add(str(val))
            for sid in as_list(meta.get("source_chunk_ids")):
                if sid not in (None, ""):
                    ids.add(str(sid))
    return ids


def decision_output_to_interop(decision: DecisionOutput) -> dict[str, Any]:
    """Convert DecisionOutput to firebench-interop-v1 without inventing fields."""
    latency_ms = None
    if decision.runtime.get("latency_ms") is not None:
        latency_ms = decision.runtime.get("latency_ms")
    elif decision.runtime.get("latency_sec") is not None:
        latency_ms = round(float(decision.runtime["latency_sec"]) * 1000.0, 3)

    method_metadata = dict(decision.method_metadata)
    if decision.safety_violations:
        method_metadata["safety_violations"] = list(decision.safety_violations)
    if decision.parsing_failure:
        method_metadata["parsing_failure"] = True
        method_metadata["parsing_errors"] = list(decision.parsing_errors)

    return {
        "schema_version": "firebench-interop-v1",
        "case_id": decision.case_id,
        "method_id": decision.method_id,
        "prediction": {
            "risk_signals": list(decision.risk_signals),
            "risk_level": decision.risk_level,
            "recommended_actions": [
                {
                    "action_id": a["action_id"],
                    "text": a["text"],
                    "priority": a.get("priority") or "unknown",
                    "evidence_refs": list(a.get("evidence_refs") or []),
                }
                for a in decision.recommended_actions
            ],
            "blocked_actions": list(decision.blocked_actions),
            "missing_confirmations": list(decision.missing_confirmations),
            "evidence_refs": list(decision.evidence_refs),
            "human_review_required": bool(decision.human_review_required),
            "final_decision_gate": decision.final_decision_gate,
            "final_response": {
                "status": decision.final_response_status,
                "text": decision.natural_language_response,
                "citations": [str(e.get("evidence_id")) for e in decision.evidence_refs if e.get("evidence_id")],
                "real_world_execution_allowed": False,
            },
        },
        "runtime": {
            "latency_ms": latency_ms,
            "llm_calls": decision.runtime.get("llm_calls"),
            "token_usage": dict(decision.runtime.get("token_usage") or {}),
            "cost": decision.runtime.get("cost"),
        },
        "provenance": dict(decision.provenance),
        "method_metadata": method_metadata,
    }


def decision_output_to_legacy_row(decision: DecisionOutput) -> dict[str, Any]:
    """Compatibility row for existing runners that expect BaselineOutput-like dicts."""
    return {
        "scenario_id": decision.case_id,
        "case_id": decision.case_id,
        "method": decision.method_id,
        "method_id": decision.method_id,
        "situation_summary": decision.natural_language_response[:500],
        "key_risks": list(decision.risk_signals),
        "risk_level": decision.risk_level,
        "recommended_actions": list(decision.recommended_actions),
        "blocked_or_unsafe_actions": list(decision.blocked_actions),
        "blocked_actions": list(decision.blocked_actions),
        "missing_confirmations": list(decision.missing_confirmations),
        "supporting_evidence": [e.get("evidence_id") for e in decision.evidence_refs],
        "citations": [e.get("evidence_id") for e in decision.evidence_refs],
        "human_review_required": decision.human_review_required,
        "final_decision_gate": decision.final_decision_gate,
        "final_response": {
            "status": decision.final_response_status,
            "text": decision.natural_language_response,
            "citations": [e.get("evidence_id") for e in decision.evidence_refs],
            "real_world_execution_allowed": False,
        },
        "retrieved_contexts": list(decision.retrieved_contexts),
        "latency_sec": float(decision.runtime.get("latency_sec") or 0.0),
        "raw_output": decision.raw_output,
        "method_specific": {
            **dict(decision.method_metadata),
            "runtime": dict(decision.runtime),
            "parsing_failure": decision.parsing_failure,
            "parsing_errors": list(decision.parsing_errors),
            "safety_violations": list(decision.safety_violations),
            "unified_decision_output": True,
        },
        "provenance": dict(decision.provenance),
    }


def unified_row_to_decision_output(row: dict[str, Any]) -> DecisionOutput:
    """Rebuild DecisionOutput from a unified legacy row (no field invention)."""
    fr = row.get("final_response") if isinstance(row.get("final_response"), dict) else {}
    ms = row.get("method_specific") if isinstance(row.get("method_specific"), dict) else {}
    actions = []
    for item in as_list(row.get("recommended_actions")):
        if isinstance(item, dict) and item.get("action_id"):
            actions.append(
                {
                    "action_id": str(item["action_id"]),
                    "text": str(item.get("text") or ""),
                    "priority": str(item.get("priority") or "unknown"),
                    "evidence_refs": [str(r) for r in as_list(item.get("evidence_refs"))],
                }
            )
    evidence_refs = []
    for item in as_list(row.get("citations") or row.get("evidence_refs")):
        if isinstance(item, dict) and item.get("evidence_id"):
            evidence_refs.append({"evidence_id": str(item["evidence_id"])})
        elif item not in (None, ""):
            evidence_refs.append({"evidence_id": str(item)})
    return DecisionOutput(
        case_id=str(row.get("case_id") or row.get("scenario_id") or ""),
        method_id=str(row.get("method_id") or row.get("method") or ""),
        risk_signals=[str(x) for x in as_list(row.get("key_risks") or row.get("risk_signals"))],
        risk_level=str(row.get("risk_level") or "unknown"),
        recommended_actions=actions,
        blocked_actions=[str(x) for x in as_list(row.get("blocked_actions") or row.get("blocked_or_unsafe_actions"))],
        missing_confirmations=[str(x) for x in as_list(row.get("missing_confirmations"))],
        evidence_refs=evidence_refs,
        human_review_required=bool(row.get("human_review_required")),
        final_decision_gate=str(row.get("final_decision_gate") or "unknown"),
        natural_language_response=str(fr.get("text") or row.get("natural_language_response") or ""),
        final_response_status=str(fr.get("status") or row.get("final_response_status") or "unknown"),
        runtime=dict((ms.get("runtime") if isinstance(ms.get("runtime"), dict) else {}) or {}),
        provenance=dict(row.get("provenance") or {}),
        method_metadata=dict(ms),
        raw_output=row.get("raw_output"),
        retrieved_contexts=list(row.get("retrieved_contexts") or []),
        parsing_failure=bool(ms.get("parsing_failure")),
        parsing_errors=list(ms.get("parsing_errors") or []),
        safety_violations=list(ms.get("safety_violations") or []),
    )


def unified_row_to_interop(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a unified decision row to firebench-interop-v1 without inventing fields."""
    return decision_output_to_interop(unified_row_to_decision_output(row))
