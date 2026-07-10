"""Load and validate the FireBench taxonomy snapshot (no fire_agent_demo import)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file

ROOT = Path(__file__).resolve().parents[3]
TAXONOMY_PATH = ROOT / "configs" / "contracts" / "firebench_taxonomy_v1.json"
ALIASES_PATH = ROOT / "configs" / "contracts" / "firebench_taxonomy_aliases_v1.json"

TAXONOMY_VERSION = "firebench-taxonomy-v1"


class TaxonomyError(ValueError):
    """Raised when taxonomy or alias tables are invalid."""


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, Any]:
    if not TAXONOMY_PATH.is_file():
        raise TaxonomyError(f"Missing taxonomy snapshot: {TAXONOMY_PATH}")
    data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    _validate_taxonomy_payload(data)
    return data


@lru_cache(maxsize=1)
def load_aliases() -> dict[str, Any]:
    if not ALIASES_PATH.is_file():
        raise TaxonomyError(f"Missing alias table: {ALIASES_PATH}")
    data = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    _validate_alias_payload(data, load_taxonomy())
    return data


def taxonomy_sha256() -> str:
    return sha256_file(TAXONOMY_PATH)


def alias_sha256() -> str:
    return sha256_file(ALIASES_PATH)


def taxonomy_provenance() -> dict[str, Any]:
    tax = load_taxonomy()
    return {
        "taxonomy_version": tax.get("taxonomy_version") or TAXONOMY_VERSION,
        "taxonomy_source_repository": tax.get("taxonomy_source_repository"),
        "taxonomy_source_branch": tax.get("taxonomy_source_branch"),
        "taxonomy_source_path": tax.get("taxonomy_source_path"),
        "taxonomy_schema_path": tax.get("taxonomy_schema_path"),
        "taxonomy_snapshot_date": tax.get("taxonomy_snapshot_date"),
        "taxonomy_source_commit": tax.get("taxonomy_source_commit"),
        "taxonomy_sha256": taxonomy_sha256(),
        "alias_sha256": alias_sha256(),
        "taxonomy_path": str(TAXONOMY_PATH.as_posix()),
        "alias_path": str(ALIASES_PATH.as_posix()),
    }


def ordered_ids(kind: str) -> list[str]:
    tax = load_taxonomy()
    key = {
        "risk_signals": "risk_signals",
        "recommended_action_ids": "recommended_action_ids",
        "action_ids": "recommended_action_ids",
        "blocked_action_ids": "blocked_action_ids",
        "blocked_actions": "blocked_action_ids",
        "confirmation_ids": "confirmation_ids",
        "missing_confirmations": "confirmation_ids",
        "risk_levels": "risk_levels",
        "priorities": "priorities",
        "final_decision_gates": "final_decision_gates",
        "final_response_statuses": "final_response_statuses",
    }.get(kind, kind)
    values = tax.get(key)
    if not isinstance(values, list):
        raise TaxonomyError(f"Unknown taxonomy kind: {kind}")
    return [str(v) for v in values]


def membership_set(kind: str) -> frozenset[str]:
    return frozenset(ordered_ids(kind))


def alias_map(kind: str) -> dict[str, str]:
    aliases = load_aliases()
    key = {
        "risk_signals": "risk_signals",
        "recommended_action_ids": "recommended_action_ids",
        "action_ids": "recommended_action_ids",
        "blocked_action_ids": "blocked_action_ids",
        "blocked_actions": "blocked_action_ids",
        "confirmation_ids": "confirmation_ids",
        "missing_confirmations": "confirmation_ids",
        "final_decision_gates": "final_decision_gates",
        "final_response_statuses": "final_response_statuses",
        "risk_levels": "risk_levels",
        "priorities": "priorities",
    }.get(kind, kind)
    block = aliases.get(key) or {}
    if not isinstance(block, dict):
        raise TaxonomyError(f"Alias block must be object: {key}")
    return {str(k): str(v) for k, v in block.items()}


def taxonomy_prompt_block() -> str:
    """Compact taxonomy instruction for LLM prompts."""
    tax = load_taxonomy()
    return "\n".join(
        [
            "Allowed risk_signals:",
            ", ".join(tax["risk_signals"]),
            "",
            "Allowed recommended action_id:",
            ", ".join(tax["recommended_action_ids"]),
            "",
            "Allowed blocked_actions:",
            ", ".join(tax["blocked_action_ids"]),
            "",
            "Allowed missing_confirmations:",
            ", ".join(tax["confirmation_ids"]),
            "",
            "Allowed risk_level:",
            ", ".join(tax["risk_levels"]),
            "",
            "Allowed priority:",
            ", ".join(tax["priorities"]),
            "",
            "Allowed final_decision_gate:",
            ", ".join(tax["final_decision_gates"]),
            "",
            "Allowed response.status:",
            ", ".join(tax["final_response_statuses"]),
            "",
            "Use IDs exactly as written.",
            "Do not translate IDs.",
            "Do not add punctuation.",
            "Do not use Chinese phrases in ID fields.",
            "Natural-language explanations belong only in action.text and response.text.",
            "Do not invent new IDs. If unsupported, omit the ID or use an applicable missing_confirmations ID.",
        ]
    )


def _validate_taxonomy_payload(data: dict[str, Any]) -> None:
    required = (
        "risk_signals",
        "recommended_action_ids",
        "blocked_action_ids",
        "confirmation_ids",
        "risk_levels",
        "priorities",
        "final_decision_gates",
        "final_response_statuses",
    )
    for key in required:
        values = data.get(key)
        if not isinstance(values, list) or not values:
            raise TaxonomyError(f"taxonomy missing non-empty list: {key}")
        if len(values) != len(set(values)):
            raise TaxonomyError(f"taxonomy has duplicate values: {key}")
        for item in values:
            if not isinstance(item, str) or not item.strip():
                raise TaxonomyError(f"taxonomy item must be non-empty string in {key}")
    for blocked in data["blocked_action_ids"]:
        if blocked != blocked.upper():
            raise TaxonomyError(f"blocked action must be uppercase: {blocked}")


def _validate_alias_payload(aliases: dict[str, Any], taxonomy: dict[str, Any]) -> None:
    mapping = {
        "risk_signals": "risk_signals",
        "recommended_action_ids": "recommended_action_ids",
        "blocked_action_ids": "blocked_action_ids",
        "confirmation_ids": "confirmation_ids",
        "final_decision_gates": "final_decision_gates",
        "final_response_statuses": "final_response_statuses",
        "risk_levels": "risk_levels",
        "priorities": "priorities",
    }
    for alias_key, tax_key in mapping.items():
        block = aliases.get(alias_key) or {}
        if not isinstance(block, dict):
            raise TaxonomyError(f"alias block must be object: {alias_key}")
        allowed = set(taxonomy[tax_key])
        sources = list(block.keys())
        if len(sources) != len(set(sources)):
            raise TaxonomyError(f"duplicate alias sources in {alias_key}")
        for source, target in block.items():
            if not isinstance(source, str) or not source.strip():
                raise TaxonomyError(f"empty alias source in {alias_key}")
            if " " in source.strip() and len(source.strip().split()) > 4:
                raise TaxonomyError(f"alias source looks like free text: {source!r}")
            if target not in allowed:
                raise TaxonomyError(
                    f"alias target not in taxonomy ({alias_key}): {source!r} -> {target!r}"
                )


def validate_alias_table() -> dict[str, Any]:
    """Return alias integrity report (raises TaxonomyError on conflict)."""
    tax = load_taxonomy()
    aliases = load_aliases()
    _validate_alias_payload(aliases, tax)
    counts = {
        key: len(aliases.get(key) or {})
        for key in (
            "risk_signals",
            "recommended_action_ids",
            "blocked_action_ids",
            "confirmation_ids",
            "final_decision_gates",
            "final_response_statuses",
        )
    }
    return {
        "ok": True,
        "alias_counts": counts,
        "alias_total": sum(counts.values()),
        "conflicts": [],
        "all_targets_valid": True,
    }
