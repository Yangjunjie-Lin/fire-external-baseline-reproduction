from __future__ import annotations

"""firebench-interop-v1 adapter.

Maps independent baseline outputs into the shared canonical prediction schema.
The adapter only parses what the baseline actually produced. It never invents
blocked actions, missing confirmations, or final-gate values from gold/target
outputs or from target-system Safety Checker logic.
"""

import json
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_json
from external_baselines.common.guards import method_leaderboard_eligibility
from external_baselines.common.text_utils import as_list

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "firebench_interop_v1_prediction.schema.json"

METHOD_ID_ALIASES = {
    "vanilla_rag": "bm25_rag",
    "ekell": "ekell_style_faithful",
    "ekell_style": "ekell_style_faithful",
    "e-kell-style": "ekell_style_faithful",
    "graphrag": "microsoft_graphrag",
}


def canonicalize_method_id(method: str) -> str:
    mid = str(method or "").strip().lower()
    return METHOD_ID_ALIASES.get(mid, mid)


def _action_objects(actions: Any, evidence_refs: list[str] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(as_list(actions)):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("action") or item.get("description") or "")
            action_id = str(item.get("action_id") or item.get("id") or f"action_{i+1}")
            priority = item.get("priority")
            refs = as_list(item.get("evidence_refs") or evidence_refs or [])
            out.append({
                "action_id": action_id,
                "text": text,
                "priority": str(priority) if priority is not None else None,
                "evidence_refs": [str(r) for r in refs],
            })
        else:
            out.append({
                "action_id": f"action_{i+1}",
                "text": str(item),
                "priority": None,
                "evidence_refs": list(evidence_refs or []),
            })
    return out


def _blocked_objects(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(as_list(items)):
        if isinstance(item, dict):
            out.append({
                "action_id": str(item.get("action_id") or item.get("id") or f"blocked_{i+1}"),
                "text": str(item.get("text") or item.get("action") or item),
            })
        else:
            out.append({"action_id": f"blocked_{i+1}", "text": str(item)})
    return out


def _evidence_refs_from_baseline(row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("supporting_evidence", "citations", "evidence_refs"):
        for item in as_list(row.get(key)):
            if isinstance(item, dict):
                refs.append(str(item.get("id") or item.get("context_id") or item.get("citation") or item))
            else:
                refs.append(str(item))
    for ctx in as_list(row.get("retrieved_contexts")):
        if isinstance(ctx, dict):
            cid = ctx.get("context_id") or ctx.get("source_id") or ctx.get("citation")
            if cid:
                refs.append(str(cid))
    # Preserve order, drop empties/dupes.
    seen: set[str] = set()
    ordered: list[str] = []
    for r in refs:
        r = r.strip()
        if r and r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def _risk_level_from_signals(signals: list[Any], explicit: Any) -> str | None:
    if explicit not in (None, ""):
        return str(explicit)
    # Do not invent a risk level; leave null if baseline did not provide one.
    return None


def _parsing_status(row: dict[str, Any]) -> str:
    raw = row.get("raw_output")
    ms = row.get("method_specific") or {}
    if ms.get("parsing_failure") or ms.get("parsing_status") == "failed":
        return "failed"
    if isinstance(raw, dict):
        parsed = raw.get("parsed")
        if parsed == {} or parsed is None:
            text = raw.get("text")
            if text and not parsed:
                return "partial_or_unparsed"
        if parsed:
            return "ok"
    if row.get("situation_summary") or row.get("recommended_actions") or row.get("key_risks"):
        return "ok"
    return "unknown"


def baseline_row_to_interop(row: dict[str, Any], *, bundle_checksum: str | None = None) -> dict[str, Any]:
    """Convert a legacy BaselineOutput dict into firebench-interop-v1."""
    method_id = canonicalize_method_id(str(row.get("method") or row.get("method_id") or "unknown"))
    case_id = str(row.get("scenario_id") or row.get("case_id") or "unknown")
    evidence_refs = _evidence_refs_from_baseline(row)
    risk_signals = [str(x) for x in as_list(row.get("key_risks") or row.get("risk_signals"))]
    risk_level = _risk_level_from_signals(risk_signals, row.get("risk_level"))
    gate = row.get("final_decision_gate")
    # Preserve empty/missing gate as-is; do not invent a safer gate.
    if gate in (None, ""):
        gate = "not_provided_by_baseline"
    else:
        gate = str(gate)

    missing = [str(x) for x in as_list(row.get("missing_confirmations"))]
    blocked = _blocked_objects(row.get("blocked_or_unsafe_actions") or row.get("blocked_actions"))
    human_review = bool(row.get("human_review_required"))
    if not human_review and gate not in {
        "not_provided_by_baseline",
        "not_applicable_or_not_provided",
        "baseline_response_without_explicit_gate",
    }:
        # Reflect baseline-stated gate only; do not force True from gold.
        human_review = "human" in gate.lower() or "confirm" in gate.lower() or "missing" in gate.lower()

    summary = str(row.get("situation_summary") or "")
    parsing_status = _parsing_status(row)
    final_status = "ok" if parsing_status == "ok" else ("parsing_failure" if parsing_status == "failed" else "partial")

    ms = dict(row.get("method_specific") or {})
    eligibility = method_leaderboard_eligibility(method_id, ms)

    latency_sec = float(row.get("latency_sec") or 0.0)
    runtime_meta = ms.get("runtime") or {}
    token_usage = dict(runtime_meta.get("token_usage") or row.get("token_usage") or {})
    llm_calls = int(runtime_meta.get("llm_calls") or row.get("llm_calls") or token_usage.get("llm_calls") or 0)
    cost = runtime_meta.get("cost", row.get("cost", None))

    prediction = {
        "risk_signals": risk_signals,
        "risk_level": risk_level,
        "recommended_actions": _action_objects(row.get("recommended_actions"), evidence_refs=evidence_refs),
        "blocked_actions": blocked,
        "missing_confirmations": missing,
        "evidence_refs": evidence_refs,
        "human_review_required": human_review,
        "final_decision_gate": gate,
        "final_response": {
            "status": final_status,
            "text": summary,
            "citations": [str(x) for x in as_list(row.get("citations"))],
            # Baselines never authorize real-world execution.
            "real_world_execution_allowed": False,
        },
        "parsing_status": parsing_status,
        "raw_output": row.get("raw_output"),
    }

    record = {
        "case_id": case_id,
        "method_id": method_id,
        "prediction": prediction,
        "runtime": {
            "latency_ms": round(latency_sec * 1000.0, 3),
            "llm_calls": llm_calls,
            "token_usage": token_usage,
            "cost": cost,
        },
        "provenance": {
            "schema": "firebench-interop-v1",
            "source_baseline_schema": "external_baselines.BaselineOutput",
            "bundle_checksum": bundle_checksum,
            "raw_output_preserved": row.get("raw_output") is not None,
            "normalizer_policy_injection": False,
            "record_sha256": None,
        },
        "method_metadata": {
            **ms,
            "legacy_method": row.get("method"),
            "retrieved_contexts": row.get("retrieved_contexts") or [],
            "leaderboard_eligibility": eligibility,
            "reproduction_class": ms.get("reproduction_class")
            or ("faithful" if "faithful" in method_id else ("enhanced" if "enhanced" in method_id else ("fallback" if eligibility["smoke_or_fallback_only"] else "baseline"))),
        },
    }
    record["provenance"]["record_sha256"] = sha256_json({
        "case_id": case_id,
        "method_id": method_id,
        "prediction": prediction,
    })
    return record


def validate_interop_record(record: dict[str, Any]) -> list[str]:
    """Lightweight structural validation without requiring jsonschema package."""
    errors: list[str] = []
    for key in ("case_id", "method_id", "prediction", "runtime", "provenance", "method_metadata"):
        if key not in record:
            errors.append(f"missing:{key}")
    pred = record.get("prediction") or {}
    for key in (
        "risk_signals",
        "risk_level",
        "recommended_actions",
        "blocked_actions",
        "missing_confirmations",
        "evidence_refs",
        "human_review_required",
        "final_decision_gate",
        "final_response",
    ):
        if key not in pred:
            errors.append(f"missing:prediction.{key}")
    fr = pred.get("final_response") or {}
    for key in ("status", "text", "citations", "real_world_execution_allowed"):
        if key not in fr:
            errors.append(f"missing:prediction.final_response.{key}")
    runtime = record.get("runtime") or {}
    for key in ("latency_ms", "llm_calls", "token_usage", "cost"):
        if key not in runtime:
            errors.append(f"missing:runtime.{key}")
    for action in pred.get("recommended_actions") or []:
        if not isinstance(action, dict) or "action_id" not in action or "text" not in action:
            errors.append("invalid:recommended_actions_item")
            break
    return errors


def load_schema() -> dict[str, Any]:
    if not SCHEMA_PATH.exists():
        return {}
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
