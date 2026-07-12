"""Deterministic FireBench ID character normalization + exact alias mapping.

No fuzzy matching, embedding similarity, regex semantic extraction, or LLM
reclassification. Chinese / free-text phrases are never inferred into IDs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from external_baselines.common.firebench_taxonomy import (
    alias_map,
    membership_set,
    ordered_ids,
)

CaseMode = Literal["lower", "upper", "preserve"]


@dataclass
class NormalizeResult:
    value: str | None
    raw_value: str
    normalized_characters: str
    alias_applied: bool = False
    alias_source: str | None = None
    reason: str | None = None


@dataclass
class TaxonomyNormalizeReport:
    aliases_applied: list[dict[str, str]] = field(default_factory=list)
    unmapped: list[dict[str, str]] = field(default_factory=list)

    def record_alias(self, *, field_name: str, source: str, target: str) -> None:
        self.aliases_applied.append(
            {"field": field_name, "source": source, "target": target}
        )

    def record_unmapped(
        self,
        *,
        field_name: str,
        raw_value: str,
        normalized_value: str,
        reason: str = "not_in_firebench_taxonomy",
    ) -> None:
        self.unmapped.append(
            {
                "field": field_name,
                "raw_value": raw_value,
                "normalized_value": normalized_value,
                "reason": reason,
            }
        )


_FULLWIDTH_SPACE = "\u3000"


def normalize_identifier_characters(
    value: str,
    *,
    case: CaseMode = "lower",
) -> str:
    """Character-level ID normalization only (no semantic mapping)."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace(_FULLWIDTH_SPACE, " ")
    text = text.strip()
    if not text:
        return ""
    # Spaces and hyphens become underscores; keep existing underscores.
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    if case == "lower":
        text = text.lower()
    elif case == "upper":
        text = text.upper()
    return text


def normalize_evidence_id(value: Any) -> str | None:
    """Evidence IDs: NFKC + strip only; preserve case; no lowercasing."""
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).replace(_FULLWIDTH_SPACE, " ").strip()
    return text or None


def _lookup(
    raw: Any,
    *,
    kind: str,
    case: CaseMode,
    field_name: str,
    strict: bool,
    report: TaxonomyNormalizeReport | None,
    dev_aliases_enabled: bool = False,
) -> NormalizeResult:
    raw_text = "" if raw is None else str(raw)
    chars = normalize_identifier_characters(raw_text, case=case)
    if not chars:
        result = NormalizeResult(
            value=None,
            raw_value=raw_text,
            normalized_characters="",
            reason="empty_after_normalization",
        )
        if report is not None:
            report.record_unmapped(
                field_name=field_name,
                raw_value=raw_text,
                normalized_value="",
                reason="empty_after_normalization",
            )
        if strict:
            return result
        return result

    allowed = membership_set(kind)
    aliases = alias_map(kind, dev_aliases_enabled=dev_aliases_enabled)

    # Exact membership on character-normalized form.
    if chars in allowed:
        return NormalizeResult(value=chars, raw_value=raw_text, normalized_characters=chars)

    # Exact alias on character-normalized source key (aliases stored lowercase-ish).
    alias_key = chars if case != "upper" else chars.lower()
    # Also try original character-normalized form for blocked aliases stored lowercase.
    mapped = aliases.get(alias_key) or aliases.get(chars)
    if mapped is None:
        # Alias table keys are typically lower snake; for blocked, sources are lower.
        mapped = aliases.get(chars.lower())
    if mapped is not None and mapped in allowed:
        if report is not None:
            report.record_alias(field_name=field_name, source=raw_text, target=mapped)
        return NormalizeResult(
            value=mapped,
            raw_value=raw_text,
            normalized_characters=chars,
            alias_applied=True,
            alias_source=raw_text,
        )

    reason = "not_in_firebench_taxonomy"
    if report is not None:
        report.record_unmapped(
            field_name=field_name,
            raw_value=raw_text,
            normalized_value=chars,
            reason=reason,
        )
    return NormalizeResult(
        value=None,
        raw_value=raw_text,
        normalized_characters=chars,
        reason=reason,
    )


def normalize_risk_signal(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
    dev_aliases_enabled: bool = False,
) -> str | None:
    return _lookup(
        value,
        kind="risk_signals",
        case="lower",
        field_name="risk_signals",
        strict=strict,
        report=report,
        dev_aliases_enabled=dev_aliases_enabled,
    ).value


def normalize_action_id(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
    dev_aliases_enabled: bool = False,
) -> str | None:
    return _lookup(
        value,
        kind="recommended_action_ids",
        case="lower",
        field_name="recommended_actions.action_id",
        strict=strict,
        report=report,
        dev_aliases_enabled=dev_aliases_enabled,
    ).value


def normalize_blocked_action_id(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
    dev_aliases_enabled: bool = False,
) -> str | None:
    return _lookup(
        value,
        kind="blocked_action_ids",
        case="upper",
        field_name="blocked_actions",
        strict=strict,
        report=report,
        dev_aliases_enabled=dev_aliases_enabled,
    ).value


def normalize_confirmation_id(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
    dev_aliases_enabled: bool = False,
) -> str | None:
    return _lookup(
        value,
        kind="confirmation_ids",
        case="lower",
        field_name="missing_confirmations",
        strict=strict,
        report=report,
        dev_aliases_enabled=dev_aliases_enabled,
    ).value


def normalize_risk_level(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
    dev_aliases_enabled: bool = False,
) -> str | None:
    return _lookup(
        value,
        kind="risk_levels",
        case="lower",
        field_name="risk_level",
        strict=strict,
        report=report,
        dev_aliases_enabled=dev_aliases_enabled,
    ).value


def normalize_priority(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
    dev_aliases_enabled: bool = False,
) -> str | None:
    return _lookup(
        value,
        kind="priorities",
        case="lower",
        field_name="recommended_actions.priority",
        strict=strict,
        report=report,
        dev_aliases_enabled=dev_aliases_enabled,
    ).value


def normalize_final_gate(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
    dev_aliases_enabled: bool = False,
) -> str | None:
    return _lookup(
        value,
        kind="final_decision_gates",
        case="lower",
        field_name="final_decision_gate",
        strict=strict,
        report=report,
        dev_aliases_enabled=dev_aliases_enabled,
    ).value


def normalize_response_status(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
    dev_aliases_enabled: bool = False,
) -> str | None:
    return _lookup(
        value,
        kind="final_response_statuses",
        case="lower",
        field_name="final_response.status",
        strict=strict,
        report=report,
        dev_aliases_enabled=dev_aliases_enabled,
    ).value


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def sort_by_taxonomy_order(values: list[str], kind: str) -> list[str]:
    order = {item: idx for idx, item in enumerate(ordered_ids(kind))}
    unique = dedupe_preserve_order(values)
    return sorted(unique, key=lambda v: (order.get(v, 10**9), v))


def _canonical_field_error(
    *,
    field: str,
    raw: str,
    canonical: str | None,
    case_id: str | None = None,
    line: int | None = None,
) -> dict[str, str | int | None]:
    if canonical is None:
        return {
            "line": line,
            "case_id": case_id,
            "field": field,
            "value": raw,
            "canonical_value": None,
            "error": "invalid_taxonomy_id",
        }
    chars = normalize_identifier_characters(raw, case="preserve")
    if chars != canonical and raw != canonical:
        return {
            "line": line,
            "case_id": case_id,
            "field": field,
            "value": raw,
            "canonical_value": canonical,
            "error": "noncanonical_alias_in_final_output",
        }
    if canonical not in membership_set(
        {
            "risk_signals": "risk_signals",
            "recommended_actions.action_id": "recommended_action_ids",
            "blocked_actions": "blocked_action_ids",
            "missing_confirmations": "confirmation_ids",
            "risk_level": "risk_levels",
            "recommended_actions.priority": "priorities",
            "final_decision_gate": "final_decision_gates",
            "final_response.status": "final_response_statuses",
        }.get(field, field)
    ):
        return {
            "line": line,
            "case_id": case_id,
            "field": field,
            "value": raw,
            "canonical_value": canonical,
            "error": "invalid_taxonomy_id",
        }
    return {}


def validate_canonical_field_value(
    raw: Any,
    *,
    field: str,
    normalizer,
    case_id: str | None = None,
    line: int | None = None,
    dev_aliases_enabled: bool = False,
) -> dict[str, str | int | None] | None:
    raw_text = str(raw or "")
    if not raw_text.strip():
        return {
            "line": line,
            "case_id": case_id,
            "field": field,
            "value": raw_text,
            "canonical_value": None,
            "error": "invalid_taxonomy_id",
        }
    canonical_formal = normalizer(raw_text, strict=False, dev_aliases_enabled=False)
    if canonical_formal is not None and raw_text == canonical_formal:
        return None
    canonical_any = normalizer(raw_text, strict=False, dev_aliases_enabled=True)
    if canonical_any is not None and raw_text != canonical_any:
        return {
            "line": line,
            "case_id": case_id,
            "field": field,
            "value": raw_text,
            "canonical_value": canonical_any,
            "error": "noncanonical_alias_in_final_output",
        }
    err = _canonical_field_error(
        field=field,
        raw=raw_text,
        canonical=canonical_formal,
        case_id=case_id,
        line=line,
    )
    return err or None


def validate_canonical_interop_record(
    record: dict[str, Any],
    *,
    line: int | None = None,
    dev_aliases_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Ensure final interop record contains canonical taxonomy IDs only."""
    errors: list[dict[str, Any]] = []
    case_id = str(record.get("case_id") or "")
    pred = record.get("prediction") or {}

    for signal in pred.get("risk_signals") or []:
        err = validate_canonical_field_value(
            signal,
            field="risk_signals",
            normalizer=normalize_risk_signal,
            case_id=case_id,
            line=line,
            dev_aliases_enabled=dev_aliases_enabled,
        )
        if err:
            errors.append(err)

    err = validate_canonical_field_value(
        pred.get("risk_level"),
        field="risk_level",
        normalizer=normalize_risk_level,
        case_id=case_id,
        line=line,
        dev_aliases_enabled=dev_aliases_enabled,
    )
    if err:
        errors.append(err)

    for action in pred.get("recommended_actions") or []:
        if not isinstance(action, dict):
            continue
        err = validate_canonical_field_value(
            action.get("action_id"),
            field="recommended_actions.action_id",
            normalizer=normalize_action_id,
            case_id=case_id,
            line=line,
            dev_aliases_enabled=dev_aliases_enabled,
        )
        if err:
            errors.append(err)
        err = validate_canonical_field_value(
            action.get("priority"),
            field="recommended_actions.priority",
            normalizer=normalize_priority,
            case_id=case_id,
            line=line,
            dev_aliases_enabled=dev_aliases_enabled,
        )
        if err:
            errors.append(err)

    for blocked in pred.get("blocked_actions") or []:
        err = validate_canonical_field_value(
            blocked,
            field="blocked_actions",
            normalizer=normalize_blocked_action_id,
            case_id=case_id,
            line=line,
            dev_aliases_enabled=dev_aliases_enabled,
        )
        if err:
            errors.append(err)

    for conf in pred.get("missing_confirmations") or []:
        err = validate_canonical_field_value(
            conf,
            field="missing_confirmations",
            normalizer=normalize_confirmation_id,
            case_id=case_id,
            line=line,
            dev_aliases_enabled=dev_aliases_enabled,
        )
        if err:
            errors.append(err)

    err = validate_canonical_field_value(
        pred.get("final_decision_gate"),
        field="final_decision_gate",
        normalizer=normalize_final_gate,
        case_id=case_id,
        line=line,
        dev_aliases_enabled=dev_aliases_enabled,
    )
    if err:
        errors.append(err)

    fr = pred.get("final_response") or {}
    err = validate_canonical_field_value(
        fr.get("status"),
        field="final_response.status",
        normalizer=normalize_response_status,
        case_id=case_id,
        line=line,
        dev_aliases_enabled=dev_aliases_enabled,
    )
    if err:
        errors.append(err)

    return errors


def assert_canonical_interop_record(
    record: dict[str, Any],
    *,
    line: int | None = None,
    dev_aliases_enabled: bool = False,
) -> None:
    errors = validate_canonical_interop_record(
        record,
        line=line,
        dev_aliases_enabled=dev_aliases_enabled,
    )
    if errors:
        raise ValueError(
            "noncanonical_taxonomy_in_final_output: "
            + "; ".join(
                f"{e.get('field')}={e.get('value')!r} ({e.get('error')})" for e in errors
            )
        )
