from __future__ import annotations

import json
from typing import Any

from external_baselines.common.llm_client import LLMClient
from external_baselines.common.text_utils import extract_json_object, normalize_text

SCENARIO_PARSE_SCHEMA_KEYS = [
    "incident_type",
    "location",
    "building_type",
    "hazards",
    "affected_people",
    "resources_or_equipment",
    "emergency_stage",
    "information_gaps",
]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def deterministic_parse(scenario_text: str) -> dict[str, Any]:
    text = normalize_text(scenario_text)
    hazards: list[str] = []
    gaps: list[str] = []
    people: list[str] = []
    resources: list[str] = []

    incident_type = "unspecified_fire_emergency"
    if any(x in text for x in ["electrical", "power", "电气", "电力", "配电", "电"]):
        incident_type = "electrical_fire"
        hazards.append("energized_equipment_or_power_uncertainty")
        gaps.append("power isolation status")
    elif any(x in text for x in ["chemical", "hazmat", "gas", "化学", "危化", "燃气"]):
        incident_type = "hazmat_or_gas_related_fire"
        hazards.append("hazardous_material_or_gas_risk")
        gaps.append("hazardous material or gas identity")
    elif "fire" in text or "火" in text:
        incident_type = "fire_emergency"

    location = "unspecified_location"
    building_type = "unspecified_building_type"
    if any(x in text for x in ["shopping mall", "mall", "commercial complex", "商场", "商业综合体"]):
        location = "shopping_mall_or_commercial_complex"
        building_type = "public_commercial_building"
        people.append("public_occupants")
    elif "electrical room" in text or "配电室" in text:
        location = "electrical_room"
        building_type = "building_service_room"
    elif "warehouse" in text or "仓库" in text:
        location = "warehouse"
        building_type = "warehouse"
    elif "building" in text or "建筑" in text:
        location = "building"

    if "smoke" in text or "烟" in text:
        hazards.append("smoke_exposure")
        resources.append("respiratory_protection")
        if "high" in text or "heavy" in text or "浓" in text or "大量" in text:
            hazards.append("high_smoke_detected")
        gaps.append("respiratory protection readiness")
    if "unknown" in text or "requires confirmation" in text or "unconfirmed" in text or "待确认" in text or "未知" in text:
        gaps.append("unconfirmed critical incident information")
    if "trapped" in text or "injured" in text or "被困" in text or "受伤" in text:
        people.append("affected_or_trapped_people")
        gaps.append("victim location and condition")
    if "sprinkler" in text or "extinguisher" in text or "hose" in text or "灭火器" in text or "消防栓" in text:
        resources.append("fire_suppression_equipment")

    return {
        "incident_type": incident_type,
        "location": location,
        "building_type": building_type,
        "hazards": _unique(hazards),
        "affected_people": _unique(people),
        "resources_or_equipment": _unique(resources),
        "emergency_stage": "initial_response",
        "information_gaps": _unique(gaps),
        "parser_mode": "deterministic",
        "parser_fallback_used": False,
    }


def normalize_parsed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in SCENARIO_PARSE_SCHEMA_KEYS:
        value = payload.get(key)
        if key in {"hazards", "affected_people", "resources_or_equipment", "information_gaps"}:
            if value is None:
                normalized[key] = []
            elif isinstance(value, list):
                normalized[key] = [str(v) for v in value if v not in (None, "")]
            else:
                normalized[key] = [str(value)]
        else:
            normalized[key] = "" if value is None else str(value)
    normalized["parser_mode"] = str(payload.get("parser_mode") or "llm_json")
    normalized["parser_fallback_used"] = bool(payload.get("parser_fallback_used", False))
    return normalized


def parse_scenario(scenario_text: str, *, llm: LLMClient | None = None, use_llm: bool = False) -> dict[str, Any]:
    if not use_llm or llm is None:
        return deterministic_parse(scenario_text)

    system = (
        "You are performing E-KELL-style emergency scenario understanding for an external baseline. "
        "Return valid JSON only. Do not call or emulate target-project SAFE modules."
    )
    user = f"""
Scenario parsing task.
Scenario:
{scenario_text}

Parser output schema:
{json.dumps({
    "incident_type": "",
    "location": "",
    "building_type": "",
    "hazards": [],
    "affected_people": [],
    "resources_or_equipment": [],
    "emergency_stage": "",
    "information_gaps": [],
}, indent=2)}

Return only JSON matching this schema.
""".strip()
    try:
        raw = llm.complete(system=system, user=user, temperature=0.0, max_tokens=700)
        parsed = extract_json_object(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM did not return a JSON object")
        parsed["parser_mode"] = "llm_json"
        parsed["parser_fallback_used"] = False
        return normalize_parsed_payload(parsed)
    except Exception as exc:
        fallback = deterministic_parse(scenario_text)
        fallback["parser_mode"] = "deterministic_after_llm_failure"
        fallback["parser_fallback_used"] = True
        fallback["parser_error"] = str(exc)
        return fallback
