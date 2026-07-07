from __future__ import annotations

from typing import Any

from external_baselines.common.llm_client import LLMClient
from external_baselines.common.text_utils import extract_json_object, normalize_text


def deterministic_parse(scenario_text: str) -> dict[str, Any]:
    text = normalize_text(scenario_text)
    hazards: list[str] = []
    gaps: list[str] = []
    people: list[str] = []

    incident_type = "unspecified_fire_emergency"
    if "electrical" in text or "power" in text:
        incident_type = "electrical_fire"
        hazards.append("energized_equipment_or_power_uncertainty")
        gaps.append("power isolation status")
    elif "chemical" in text or "hazmat" in text:
        incident_type = "hazmat_fire"
        hazards.append("hazardous_material")
        gaps.append("hazardous material identity")
    elif "fire" in text:
        incident_type = "fire_emergency"

    location = "unspecified_location"
    if "shopping mall" in text or "mall" in text:
        location = "shopping_mall"
        people.append("public_occupants")
    elif "electrical room" in text:
        location = "electrical_room"
    elif "warehouse" in text:
        location = "warehouse"
    elif "building" in text:
        location = "building"

    if "smoke" in text:
        hazards.append("smoke_exposure")
        if "high" in text:
            hazards.append("high_smoke_detected")
        gaps.append("respiratory protection readiness")
    if "unknown" in text or "requires confirmation" in text or "unconfirmed" in text:
        gaps.append("unconfirmed critical incident information")
    if "trapped" in text or "injured" in text:
        people.append("affected_or_trapped_people")

    return {
        "incident_type": incident_type,
        "location": location,
        "hazards": list(dict.fromkeys(hazards)),
        "affected_people": list(dict.fromkeys(people)),
        "emergency_stage": "initial_response",
        "information_gaps": list(dict.fromkeys(gaps)),
    }


def parse_scenario(scenario_text: str, *, llm: LLMClient | None = None, use_llm: bool = False) -> dict[str, Any]:
    if not use_llm or llm is None:
        return deterministic_parse(scenario_text)
    system = "You reproduce E-KELL-style scenario parsing. Return valid JSON only."
    user = f"""
Scenario parsing task.
Scenario: {scenario_text}
Return JSON with incident_type, location, hazards, affected_people, emergency_stage, information_gaps.
""".strip()
    raw = llm.complete(system=system, user=user, temperature=0.0, max_tokens=600)
    parsed = extract_json_object(raw)
    return parsed or deterministic_parse(scenario_text)
