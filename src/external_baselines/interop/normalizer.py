from __future__ import annotations

"""Neutral prediction-field normalizer for firebench-interop-v1.

Never reads gold / expected / annotation labels. Never invents safer gates
from empty fields. Unmapped taxonomy values go to diagnostics.
"""

import hashlib
import re
from typing import Any

from external_baselines.common.text_utils import as_list
from external_baselines.interop.taxonomy import TAXONOMY_VERSION, canonicalize, canonical_id

NORMALIZER_VERSION = "external_baselines_neutral_v1"

_RISK_LEVELS = frozenset({"none", "low", "medium", "high", "critical", "unknown"})

_RISK_LEVEL_ALIASES = {
    "n/a": "none",
    "na": "none",
    "no_risk": "none",
    "norisk": "none",
    "med": "medium",
    "moderate": "medium",
    "severe": "high",
    "extreme": "critical",
    "crit": "critical",
}

_GATE_CANONICAL = frozenset(
    {"allow_response", "await_human_confirmation", "block_response", "unknown"}
)

_STATUS_CANONICAL = frozenset(
    {
        "provided",
        "awaiting_human_confirmation",
        "blocked",
        "not_applicable",
        "unknown",
    }
)

_GATE_TO_STATUS = {
    "allow_response": "provided",
    "await_human_confirmation": "awaiting_human_confirmation",
    "block_response": "blocked",
}


def _reject_gold(row: dict[str, Any]) -> None:
    for key in ("gold", "expected", "expected_output", "labels", "annotation"):
        if key in row and row[key] not in (None, {}, [], ""):
            raise ValueError(
                f"Neutral normalizer refuses gold-bearing field {key!r}; "
                "pass prediction-only rows."
            )


def _as_str_list(values: Any) -> list[str]:
    out: list[str] = []
    for item in as_list(values):
        if isinstance(item, dict):
            text = str(
                item.get("id")
                or item.get("action_id")
                or item.get("signal")
                or item.get("text")
                or item.get("action")
                or item
            ).strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def _unmapped_action_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"unmapped:{digest}"


def _normalize_risk_level(raw: Any) -> str:
    if raw is None or str(raw).strip() == "":
        return "unknown"
    text = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if text in _RISK_LEVELS:
        return text
    if text in _RISK_LEVEL_ALIASES:
        return _RISK_LEVEL_ALIASES[text]
    return "unknown"


def _normalize_priority(raw: Any) -> str:
    if raw is None or str(raw).strip() == "":
        return "unknown"
    text = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if text in {"low", "medium", "high", "critical", "unknown"}:
        return text
    if text in {"med", "moderate"}:
        return "medium"
    return "unknown"


def _map_decision_gate(raw: Any) -> tuple[str, str | None]:
    """Map free-form gate strings to the interop enum. Never invent safer gates."""
    if raw is None:
        return "unknown", None
    raw_text = str(raw).strip()
    if not raw_text:
        return "unknown", raw_text
    lowered = raw_text.lower().replace("-", "_").replace(" ", "_")
    if lowered in _GATE_CANONICAL:
        return lowered, raw_text
    # Explicit block signals first.
    if "block_response" in lowered or re.search(r"\bblock(?:ed|ing)?\b", lowered):
        if "allow" not in lowered:
            return "block_response", raw_text
    if any(
        token in lowered
        for token in (
            "await_human_confirmation",
            "await",
            "human_confirm",
            "human_confirmation",
            "missing_confirm",
            "requires_human",
            "critical_information_missing",
        )
    ):
        return "await_human_confirmation", raw_text
    if "allow_response" in lowered or lowered in {
        "allow",
        "baseline_response_without_explicit_gate",
    }:
        return "allow_response", raw_text
    return "unknown", raw_text


def _status_from_gate(gate: str, status_raw: Any) -> tuple[str, str | None]:
    raw_text = None if status_raw is None else str(status_raw).strip()
    if raw_text:
        lowered = raw_text.lower().replace("-", "_").replace(" ", "_")
        if lowered in _STATUS_CANONICAL:
            return lowered, raw_text
        if lowered in {"awaiting", "await_confirmation", "needs_confirmation"}:
            return "awaiting_human_confirmation", raw_text
        if lowered in {"ok", "ready", "complete", "completed"}:
            return "provided", raw_text
        if lowered in {"n/a", "na", "not_applicable"}:
            return "not_applicable", raw_text
    return _GATE_TO_STATUS.get(gate, "unknown"), raw_text


def _normalize_recommended_actions(raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    actions: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for item in as_list(raw):
        if isinstance(item, dict):
            text = str(
                item.get("text") or item.get("action") or item.get("description") or ""
            ).strip()
            candidate = str(item.get("action_id") or item.get("id") or "").strip()
            priority = item.get("priority")
            refs = [str(r) for r in as_list(item.get("evidence_refs") or [])]
        else:
            text = str(item).strip()
            candidate = ""
            priority = None
            refs = []
        if not text and not candidate:
            continue
        mapped = canonical_id("required_actions", candidate) if candidate else None
        if mapped is None and text:
            mapped = canonical_id("required_actions", text)
        if mapped is None:
            source = candidate or text
            if source and source not in unmapped:
                unmapped.append(source)
            if not text:
                text = candidate
            # Do not invent action_1; keep free text and mark unmapped for scoring.
            action_id = _unmapped_action_id(text) if text else "unmapped"
            actions.append(
                {
                    "action_id": action_id,
                    "text": text,
                    "priority": _normalize_priority(priority),
                    "evidence_refs": refs,
                }
            )
        else:
            actions.append(
                {
                    "action_id": mapped,
                    "text": text or mapped,
                    "priority": _normalize_priority(priority),
                    "evidence_refs": refs,
                }
            )
    return actions, unmapped


def normalize_prediction_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize prediction-facing fields without gold and without safer defaults."""
    if not isinstance(row, dict):
        raise TypeError("normalize_prediction_fields expects a dict row")
    _reject_gold(row)

    # Prefer prediction nested payload when present; else top-level baseline fields.
    pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
    source = {**row, **pred}

    risk_raw = source.get("risk_signals") or source.get("key_risks") or []
    risk_mapped, risk_unmapped = canonicalize("risk_signals", _as_str_list(risk_raw))

    blocked_raw = (
        source.get("blocked_actions")
        or source.get("blocked_or_unsafe_actions")
        or []
    )
    blocked_mapped, blocked_unmapped = canonicalize(
        "blocked_actions", _as_str_list(blocked_raw)
    )

    missing_raw = source.get("missing_confirmations") or []
    missing_mapped, missing_unmapped = canonicalize(
        "missing_confirmations", _as_str_list(missing_raw)
    )

    actions_raw = source.get("recommended_actions") or []
    recommended_actions, action_unmapped = _normalize_recommended_actions(actions_raw)

    gate_raw_value = source.get("final_decision_gate")
    gate, gate_raw = _map_decision_gate(gate_raw_value)

    status_source = None
    final_response = source.get("final_response")
    if isinstance(final_response, dict):
        status_source = final_response.get("status")
    status_source = status_source or source.get("final_response_status")
    status, status_raw = _status_from_gate(gate, status_source)

    risk_level = _normalize_risk_level(source.get("risk_level"))

    return {
        "risk_signals": risk_mapped,
        "risk_level": risk_level,
        "recommended_actions": recommended_actions,
        "blocked_actions": blocked_mapped,
        "missing_confirmations": missing_mapped,
        "final_decision_gate": gate,
        "final_response_status": status,
        "diagnostics": {
            "unmapped_risk_signals": risk_unmapped,
            "unmapped_recommended_actions": action_unmapped,
            "unmapped_blocked_actions": blocked_unmapped,
            "unmapped_missing_confirmations": missing_unmapped,
            "normalizer_version": NORMALIZER_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "gate_raw": gate_raw,
            "status_raw": status_raw,
        },
    }
