from __future__ import annotations

"""Versioned local snapshot of fireagent_bench taxonomy IDs and aliases.

This is a declarative copy for external-baseline normalization only.
It must NOT import fire_agent_demo. Update TAXONOMY_VERSION when re-syncing
from the main project's evaluation taxonomy.
"""

from collections.abc import Iterable, Mapping
from typing import Any

TAXONOMY_VERSION = "fireagent_bench_v1_snapshot_2026-07-10"

RISK_SIGNALS = frozenset(
    {
        "fire_detected",
        "smoke_detected",
        "high_smoke_detected",
        "possible_trapped_people",
        "people_at_risk",
        "route_status_blocked",
        "route_status_unknown",
        "electrical_risk",
        "power_cutoff_unknown",
        "hazardous_material_risk",
        "equipment_status_unknown",
        "dynamic_state_needed",
        "evidence_needed",
        "stale_dynamic_state",
        "bypass_request",
        "unsupported_visual_request",
        "unsupported_building_plan_request",
        "resource_status_unknown",
    }
)

REQUIRED_ACTIONS = frozenset(
    {
        "verify_power_isolation",
        "confirm_trapped_people_status",
        "confirm_evacuation_route",
        "prepare_respiratory_protection",
        "refresh_dynamic_state",
        "request_human_confirmation",
        "retrieve_rule_or_sop_evidence",
        "retrieve_graphrag_evidence",
        "surface_capability_gap",
        "compose_safety_bounded_briefing",
    }
)

BLOCKED_ACTIONS = frozenset(
    {
        "BLOCK_UNVERIFIED_WATER_SUPPRESSION",
        "BLOCK_ENTRY_WITHOUT_RESPIRATORY_PROTECTION",
        "BLOCK_USE_BLOCKED_ROUTE",
        "BLOCK_RELY_ON_STALE_ROUTE",
        "BLOCK_AUTONOMOUS_FIELD_ACTUATION",
        "BLOCK_REAL_WORLD_EXECUTION",
        "BLOCK_BYPASS_HITL",
        "BLOCK_UNSUPPORTED_VISUAL_ASSUMPTION",
        "BLOCK_UNSUPPORTED_BUILDING_PLAN_ASSUMPTION",
    }
)

MISSING_CONFIRMATIONS = frozenset(
    {
        "power_cutoff_status",
        "trapped_people_status",
        "route_status",
        "smoke_level",
        "respiratory_protection_status",
        "sensor_freshness",
        "equipment_status",
        "human_authority_check",
        "building_plan_availability",
        "visual_input_availability",
    }
)

RISK_SIGNAL_ALIASES: dict[str, str] = {
    "fire": "fire_detected",
    "active_fire": "fire_detected",
    "active_or_suspected_fire": "fire_detected",
    "smoke": "smoke_detected",
    "smoke_exposure_or_smoke_control": "smoke_detected",
    "high_smoke": "high_smoke_detected",
    "high_smoke_confirmed": "high_smoke_detected",
    "possible_rescue_required": "possible_trapped_people",
    "trapped_people": "possible_trapped_people",
    "trapped_people_confirmed": "possible_trapped_people",
    "occupancy_possible_not_confirmed": "possible_trapped_people",
    "people_risk": "people_at_risk",
    "blocked_route": "route_status_blocked",
    "route_blocked": "route_status_blocked",
    "route_blocked_confirmed": "route_status_blocked",
    "route_unknown": "route_status_unknown",
    "electrical_equipment_risk": "electrical_risk",
    "electrical_or_battery_hazard": "electrical_risk",
    "power_status_unknown": "power_cutoff_unknown",
    "power_isolation_unknown": "power_cutoff_unknown",
    "hazmat_risk": "hazardous_material_risk",
    "hazardous_material_or_explosion_risk": "hazardous_material_risk",
    "hazardous_material": "hazardous_material_risk",
    "blocked_evacuation_route": "route_status_blocked",
    "equipment_unknown": "equipment_status_unknown",
    "equipment_status_stale": "stale_dynamic_state",
    "sensor_status_stale": "stale_dynamic_state",
    "route_status_stale": "stale_dynamic_state",
    "resource_status_stale": "stale_dynamic_state",
    "occupancy_status_stale": "stale_dynamic_state",
    "dynamic_state_stale": "stale_dynamic_state",
    "uncertain_dynamic_state": "dynamic_state_needed",
    "unsafe_bypass_attempt": "bypass_request",
    "unsafe_bypass_request": "bypass_request",
    "bypass_active_context": "bypass_request",
    "visual_requested": "unsupported_visual_request",
    "unsupported_visual_evidence_requested": "unsupported_visual_request",
    "building_plan_requested": "unsupported_building_plan_request",
    "blocked_or_compromised_route": "route_status_blocked",
    "route_smoke_or_blocked_uncertainty": "route_status_unknown",
    "route_state_uncertain": "route_status_unknown",
    "resource_unknown": "resource_status_unknown",
    "resource_status_unconfirmed": "resource_status_unknown",
    "human_confirmation_required": "dynamic_state_needed",
}

REQUIRED_ACTION_ALIASES: dict[str, str] = {
    "ACT_ELECTRICAL_POWER_CHECK": "verify_power_isolation",
    "ACT_RESCUE_PREPARE": "confirm_trapped_people_status",
    "ACT_ROUTE_AVOID_BLOCKED": "confirm_evacuation_route",
    "ACT_ROUTE_REFRESH_STALE": "refresh_dynamic_state",
    "ACT_SMOKE_PROTECTION_CHECK": "prepare_respiratory_protection",
    "ACT_REFRESH_DYNAMIC_STATE": "refresh_dynamic_state",
    "ACT_EVIDENCE_GROUNDED_BRIEFING": "compose_safety_bounded_briefing",
    "CONFIRM_HITL": "request_human_confirmation",
    "ACT_REQUEST_HUMAN_CONFIRMATION": "request_human_confirmation",
    "ACT_RETRIEVE_RULE_EVIDENCE": "retrieve_rule_or_sop_evidence",
    "ACT_RETRIEVE_GRAPH_EVIDENCE": "retrieve_graphrag_evidence",
    "ACT_SURFACE_CAPABILITY_GAP": "surface_capability_gap",
}

BLOCKED_ACTION_ALIASES: dict[str, str] = {
    "BLOCK_FIELD_ACTUATION": "BLOCK_AUTONOMOUS_FIELD_ACTUATION",
    "BLOCK_AUTONOMOUS_EXECUTION": "BLOCK_AUTONOMOUS_FIELD_ACTUATION",
    "BLOCK_EXECUTION": "BLOCK_REAL_WORLD_EXECUTION",
    "BLOCK_HITL_BYPASS": "BLOCK_BYPASS_HITL",
    "stale_route_reliance": "BLOCK_RELY_ON_STALE_ROUTE",
    "rely_on_stale_route": "BLOCK_RELY_ON_STALE_ROUTE",
    "use_stale_route_state": "BLOCK_RELY_ON_STALE_ROUTE",
    "BLOCK_USE_STALE_ROUTE": "BLOCK_RELY_ON_STALE_ROUTE",
}

MISSING_CONFIRMATION_ALIASES: dict[str, str] = {
    "CONFIRM_ELECTRICAL_POWER_STATUS": "power_cutoff_status",
    "CONFIRM_POWER_CUTOFF": "power_cutoff_status",
    "CONFIRM_TRAPPED_PEOPLE": "trapped_people_status",
    "CONFIRM_ROUTE_STATUS": "route_status",
    "CONFIRM_SMOKE_LEVEL": "smoke_level",
    "CONFIRM_RESPIRATORY_PROTECTION": "respiratory_protection_status",
    "CONFIRM_SENSOR_FRESHNESS": "sensor_freshness",
    "CONFIRM_EQUIPMENT_STATUS": "equipment_status",
    "CONFIRM_POLICY_LEVEL_HITL": "human_authority_check",
    "CONFIRM_HIGH_RISK_LOW_CONFIDENCE": "human_authority_check",
    "CONFIRM_LOW_CONFIDENCE_STATE_REFRESH": "sensor_freshness",
    "CONFIRM_BUILDING_PLAN_AVAILABILITY": "building_plan_availability",
    "CONFIRM_VISUAL_INPUT_AVAILABILITY": "visual_input_availability",
    "passage_status": "route_status",
    "people_at_risk": "trapped_people_status",
    "fresh_dynamic_state_snapshot": "sensor_freshness",
    "fresh_dynamic_state": "sensor_freshness",
    "stage_1_low_confidence_confirmation": "human_authority_check",
    "human_confirmation_policy_review": "human_authority_check",
    "connected_visual_perception_module": "visual_input_availability",
    "human_confirmation_required": "human_authority_check",
    "CONFIRM_UNKNOWN_medical_team": "human_authority_check",
}

TAXONOMY_SETS: dict[str, frozenset[str]] = {
    "risk_signals": RISK_SIGNALS,
    "required_actions": REQUIRED_ACTIONS,
    "blocked_actions": BLOCKED_ACTIONS,
    "missing_confirmations": MISSING_CONFIRMATIONS,
}

ALIAS_MAPS: dict[str, Mapping[str, str]] = {
    "risk_signals": RISK_SIGNAL_ALIASES,
    "required_actions": REQUIRED_ACTION_ALIASES,
    "blocked_actions": BLOCKED_ACTION_ALIASES,
    "missing_confirmations": MISSING_CONFIRMATION_ALIASES,
}


def canonical_id(kind: str, value: Any) -> str | None:
    """Return a canonical ID, or None when no declared mapping exists."""
    text = str(value or "").strip()
    if not text:
        return None
    if kind not in TAXONOMY_SETS:
        raise KeyError(f"Unknown taxonomy kind: {kind!r}")
    canonical = TAXONOMY_SETS[kind]
    if text in canonical:
        return text
    aliases = ALIAS_MAPS[kind]
    if text in aliases:
        return aliases[text]
    lowered = text.lower()
    for alias, mapped in aliases.items():
        if alias.lower() == lowered:
            return mapped
    # Case-insensitive match against canonical set (blocked actions are UPPER_SNAKE).
    for item in canonical:
        if item.lower() == lowered:
            return item
    return None


def canonicalize(kind: str, values: Iterable[Any]) -> tuple[list[str], list[str]]:
    """Normalize values while preserving first occurrence order and unmapped IDs."""
    mapped: list[str] = []
    unmapped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = canonical_id(kind, text)
        target = mapped if normalized else unmapped
        item = normalized or text
        if item not in target:
            target.append(item)
    return mapped, unmapped
