from __future__ import annotations

from typing import Any

from external_baselines.common.text_utils import as_list, normalize_text


def combined_output_text(output: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in [
        "situation_summary",
        "key_risks",
        "recommended_actions",
        "blocked_or_unsafe_actions",
        "missing_confirmations",
        "supporting_evidence",
        "final_decision_gate",
    ]:
        value = output.get(key)
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    return "\n".join(parts)


def infer_structured_safety_fields(output: dict[str, Any]) -> dict[str, Any]:
    """Legacy compatibility normalizer that may append inferred safety text.

    WARNING: This invents blocked/missing/gate content that the baseline did not
    explicitly generate. It must remain OFF for paper/interop runs. Prefer
    ``maybe_infer_structured_safety_fields`` which defaults to no injection.
    """
    text = normalize_text(combined_output_text(output))
    blocked = list(as_list(output.get("blocked_or_unsafe_actions")))
    missing = list(as_list(output.get("missing_confirmations")))

    if "water" in text and ("electrical" in text or "power" in text) and not any("water" in normalize_text(x) for x in blocked):
        blocked.append("Inferred: water-based suppression is unsafe until electrical isolation is confirmed.")
    if "respiratory" in text and "smoke" in text and not any("respiratory" in normalize_text(x) for x in missing):
        missing.append("Inferred: respiratory protection / SCBA readiness requires confirmation.")
    if "power" in text and not any("power" in normalize_text(x) or "isolation" in normalize_text(x) for x in missing):
        missing.append("Inferred: power isolation status requires confirmation.")

    gate = output.get("final_decision_gate") or "not_applicable_or_not_provided"
    if gate == "not_applicable_or_not_provided" and (missing or blocked):
        gate = "critical_risk_requires_human_confirmation"

    output = dict(output)
    output["blocked_or_unsafe_actions"] = [str(x) for x in blocked]
    output["missing_confirmations"] = [str(x) for x in missing]
    output["final_decision_gate"] = str(gate)
    ms = dict(output.get("method_specific") or {})
    ms["structured_safety_fields"] = "inferred_from_text"
    ms["normalizer_policy_injection"] = True
    output["method_specific"] = ms
    return output


def maybe_infer_structured_safety_fields(output: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply safety-field inference only when explicitly enabled.

    Default is OFF so interop/paper runs never invent missing safety content.
    """
    config = config or {}
    enabled = bool(config.get("normalization", {}).get("infer_structured_safety_fields", False))
    if not enabled:
        out = dict(output)
        ms = dict(out.get("method_specific") or {})
        ms.setdefault("structured_safety_fields", "baseline_generated_only")
        ms["normalizer_policy_injection"] = False
        out["method_specific"] = ms
        return out
    return infer_structured_safety_fields(output)
