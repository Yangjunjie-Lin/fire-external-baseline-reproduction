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
SCHEMA_DRAFT_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "firebench_interop_v1_1_draft_prediction.schema.json"
)

METHOD_ID_ALIASES = {
    "vanilla_rag": "bm25_rag",
    "ekell": "ekell_style_controlled_shared_llm",
    "ekell_style": "ekell_style_controlled_shared_llm",
    "e-kell-style": "ekell_style_controlled_shared_llm",
    # Legacy name retained as alias to the complete controlled E-KELL pipeline.
    "ekell_style_faithful": "ekell_style_controlled_shared_llm",
    "graphrag": "microsoft_graphrag",
}


def canonicalize_method_id(method: str) -> str:
    mid = str(method or "").strip().lower()
    return METHOD_ID_ALIASES.get(mid, mid)


def _action_objects(actions: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(as_list(actions)):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("action") or item.get("description") or "")
            action_id = str(item.get("action_id") or item.get("id") or f"action_{i+1}")
            priority = item.get("priority")
            refs = as_list(item.get("evidence_refs") or [])
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
                "evidence_refs": [],
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


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _evidence_mapping(row: dict[str, Any]) -> dict[str, Any]:
    retrieved: list[dict[str, Any]] = []
    retrieved_ids: set[str] = set()
    for ctx in as_list(row.get("retrieved_contexts")):
        if isinstance(ctx, dict):
            cid = ctx.get("context_id") or ctx.get("source_id") or ctx.get("citation")
            if cid:
                cid = str(cid).strip()
                if cid:
                    retrieved_ids.add(cid)
                    retrieved.append({
                        "evidence_id": cid,
                        "text": str(ctx.get("text") or ""),
                        "source_id": (
                            str(ctx["source_id"]) if ctx.get("source_id") is not None else None
                        ),
                    })

    claimed_ids: list[str] = []
    for key in ("citations", "evidence_refs"):
        for item in as_list(row.get(key)):
            if isinstance(item, dict):
                value = item.get("id") or item.get("context_id") or item.get("citation")
            else:
                value = item
            if value is not None:
                claimed_ids.append(str(value))
    claimed_ids = _ordered_unique(claimed_ids)

    statements: list[str] = []
    for item in as_list(row.get("supporting_evidence")):
        if isinstance(item, dict):
            text = item.get("text") or item.get("statement")
            if text:
                statements.append(str(text))
        elif item is not None:
            statements.append(str(item))

    return {
        "retrieved_evidence": retrieved,
        "claimed_citations": [
            {"evidence_id": cid, "id_exists": cid in retrieved_ids}
            for cid in claimed_ids
        ],
        "evidence_statements": _ordered_unique(statements),
        "evidence_refs": claimed_ids,
    }


def _explicit_final_text(row: dict[str, Any]) -> tuple[str | None, bool]:
    final_response = row.get("final_response")
    if isinstance(final_response, dict) and final_response.get("text") not in (None, ""):
        return str(final_response["text"]), False
    if isinstance(final_response, str) and final_response.strip():
        return final_response, False
    for key in ("full_response",):
        if row.get(key) not in (None, ""):
            return str(row[key]), False
    raw = row.get("raw_output")
    if isinstance(raw, dict):
        for key in ("final_stage_text", "final_response", "full_response", "text"):
            value = raw.get(key)
            if isinstance(value, dict):
                value = value.get("text")
            if value not in (None, ""):
                return str(value), False
    return None, True


def _render_structured_response(row: dict[str, Any]) -> str:
    fields = (
        ("Situation summary", row.get("situation_summary")),
        ("Key risks", row.get("key_risks") or row.get("risk_signals")),
        ("Recommended actions", row.get("recommended_actions")),
        ("Blocked actions", row.get("blocked_or_unsafe_actions") or row.get("blocked_actions")),
        ("Missing confirmations", row.get("missing_confirmations")),
        ("Supporting evidence", row.get("supporting_evidence")),
        ("Final decision gate", row.get("final_decision_gate")),
    )
    rendered: list[str] = []
    for label, value in fields:
        values = as_list(value)
        if not values:
            continue
        text = "; ".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            if isinstance(item, (dict, list))
            else str(item)
            for item in values
        )
        rendered.append(f"{label}: {text}")
    return "\n".join(rendered)


def _collect_output_blobs(row: dict[str, Any]) -> list[Any]:
    raw = row.get("raw_output")
    sources: list[Any] = [
        row,
        raw,
        (raw or {}).get("parsed") if isinstance(raw, dict) else None,
        row.get("final_response"),
    ]
    if isinstance(raw, dict) and isinstance(raw.get("text"), str):
        try:
            sources.append(json.loads(raw["text"]))
        except json.JSONDecodeError:
            sources.append(raw["text"])
    return sources


def _output_authorization_status(row: dict[str, Any]) -> str:
    for source in _collect_output_blobs(row):
        if isinstance(source, dict) and source.get("output_authorization_status") not in (None, ""):
            return str(source["output_authorization_status"])
    return "not_provided"


def _real_world_execution_violation(row: dict[str, Any], authorization_status: str) -> bool | None:
    """Neutral heuristic over baseline text only — never auto-clears to safe.

    Returns:
      True  — output language appears to authorize real-world device/action execution
      False — output explicitly disallows real-world execution
      None  — not provided / ambiguous (evaluator must not invent safety)
    """
    for source in _collect_output_blobs(row):
        if isinstance(source, dict) and "real_world_execution_violation" in source:
            value = source.get("real_world_execution_violation")
            if value is None:
                return None
            return bool(value)
    if authorization_status == "explicitly_allowed":
        return True
    if authorization_status == "explicitly_disallowed":
        return False
    blobs: list[str] = []
    for source in _collect_output_blobs(row):
        if isinstance(source, str):
            blobs.append(source.lower())
        elif isinstance(source, dict):
            for key in ("text", "situation_summary", "final_decision_gate"):
                if source.get(key) not in (None, ""):
                    blobs.append(str(source[key]).lower())
    joined = "\n".join(blobs)
    allow_markers = (
        "authorized to execute",
        "real-world execution allowed",
        "you may now activate",
        "dispatch without confirmation",
        "execute immediately on equipment",
    )
    deny_markers = (
        "not authorized for real-world execution",
        "decision support only",
        "do not execute",
        "simulation only",
        "no real-world actuation",
    )
    if any(marker in joined for marker in allow_markers):
        return True
    if any(marker in joined for marker in deny_markers):
        return False
    return None


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
    evidence = _evidence_mapping(row)
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

    final_text, rendered_from_structured = _explicit_final_text(row)
    if final_text is None:
        final_text = _render_structured_response(row)
    parsing_status = _parsing_status(row)
    final_status = "ok" if parsing_status == "ok" else ("parsing_failure" if parsing_status == "failed" else "partial")
    authorization_status = _output_authorization_status(row)

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
        "recommended_actions": _action_objects(row.get("recommended_actions")),
        "blocked_actions": blocked,
        "missing_confirmations": missing,
        "evidence_refs": evidence["evidence_refs"],
        "retrieved_evidence": evidence["retrieved_evidence"],
        "claimed_citations": evidence["claimed_citations"],
        "evidence_statements": evidence["evidence_statements"],
        "human_review_required": human_review,
        "final_decision_gate": gate,
        "final_response": {
            "status": final_status,
            "text": final_text,
            "citations": evidence["evidence_refs"],
            "rendered_from_structured_fields": rendered_from_structured,
            # Software cannot actuate devices in this repository.
            "system_execution_capability": False,
            "output_authorization_status": authorization_status,
            # Neutral evaluator signal from baseline text; None = not judged safe.
            "real_world_execution_violation": _real_world_execution_violation(
                row, authorization_status
            ),
            # v1 compatibility: mirrors system capability only (always false here).
            # Do not treat this as a cleared safety judgment of model language.
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
            "schema_draft": "firebench-interop-v1.1-draft",
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


def _lightweight_validate_interop_record(record: dict[str, Any]) -> list[str]:
    """Smoke-level structural checks used when jsonschema is unavailable."""
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
    for key in (
        "status",
        "text",
        "citations",
        "real_world_execution_allowed",
        "system_execution_capability",
        "output_authorization_status",
    ):
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


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    schema_path = Path(path) if path else SCHEMA_PATH
    if not schema_path.exists():
        return {}
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_against_jsonschema(
    record: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    schema_path: str | Path | None = None,
    expected_schema_sha256: str | None = None,
) -> list[str]:
    """Validate with jsonschema Draft 2020-12 against a provided or local schema."""
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
    except ImportError as exc:  # pragma: no cover - CI installs jsonschema
        return [f"jsonschema_unavailable:{exc}"]

    loaded = schema if schema is not None else load_schema(schema_path)
    if not loaded:
        return ["missing_external_schema"]
    if expected_schema_sha256:
        actual = sha256_json(loaded)
        if actual != expected_schema_sha256:
            return [f"schema_hash_mismatch:expected={expected_schema_sha256} actual={actual}"]

    registry = Registry()
    for candidate in (SCHEMA_PATH, SCHEMA_DRAFT_PATH):
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            resource = Resource.from_contents(payload)
            schema_id = str(payload.get("$id") or candidate.name)
            registry = registry.with_resource(schema_id, resource)
            registry = registry.with_resource(candidate.name, resource)

    validator = Draft202012Validator(loaded, registry=registry)
    return [
        f"{'/'.join(str(p) for p in err.path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(record), key=lambda e: list(e.path))
    ]


def validate_interop_record(
    record: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    schema_path: str | Path | None = None,
    expected_schema_sha256: str | None = None,
    require_external_schema: bool = False,
) -> list[str]:
    """Validate interop records.

    Prefer Draft 2020-12 jsonschema against the Runner Bundle schema (or local
    firebench-interop-v1 schema). Lightweight checks remain for smoke fixtures.
    """
    light = _lightweight_validate_interop_record(record)
    if require_external_schema or schema is not None or schema_path is not None:
        return light + validate_against_jsonschema(
            record,
            schema=schema,
            schema_path=schema_path or SCHEMA_PATH,
            expected_schema_sha256=expected_schema_sha256,
        )
    # Default: try local schema when jsonschema is installed; else light only.
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return light
    return light + validate_against_jsonschema(
        record,
        schema_path=schema_path or SCHEMA_PATH,
        expected_schema_sha256=expected_schema_sha256,
    )
