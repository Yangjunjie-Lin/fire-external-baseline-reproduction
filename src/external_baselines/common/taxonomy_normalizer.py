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
    aliases = alias_map(kind)

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
) -> str | None:
    return _lookup(
        value,
        kind="risk_signals",
        case="lower",
        field_name="risk_signals",
        strict=strict,
        report=report,
    ).value


def normalize_action_id(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
) -> str | None:
    return _lookup(
        value,
        kind="recommended_action_ids",
        case="lower",
        field_name="recommended_actions.action_id",
        strict=strict,
        report=report,
    ).value


def normalize_blocked_action_id(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
) -> str | None:
    return _lookup(
        value,
        kind="blocked_action_ids",
        case="upper",
        field_name="blocked_actions",
        strict=strict,
        report=report,
    ).value


def normalize_confirmation_id(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
) -> str | None:
    return _lookup(
        value,
        kind="confirmation_ids",
        case="lower",
        field_name="missing_confirmations",
        strict=strict,
        report=report,
    ).value


def normalize_risk_level(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
) -> str | None:
    return _lookup(
        value,
        kind="risk_levels",
        case="lower",
        field_name="risk_level",
        strict=strict,
        report=report,
    ).value


def normalize_priority(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
) -> str | None:
    return _lookup(
        value,
        kind="priorities",
        case="lower",
        field_name="recommended_actions.priority",
        strict=strict,
        report=report,
    ).value


def normalize_final_gate(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
) -> str | None:
    return _lookup(
        value,
        kind="final_decision_gates",
        case="lower",
        field_name="final_decision_gate",
        strict=strict,
        report=report,
    ).value


def normalize_response_status(
    value: str,
    *,
    strict: bool = True,
    report: TaxonomyNormalizeReport | None = None,
) -> str | None:
    return _lookup(
        value,
        kind="final_response_statuses",
        case="lower",
        field_name="final_response.status",
        strict=strict,
        report=report,
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
