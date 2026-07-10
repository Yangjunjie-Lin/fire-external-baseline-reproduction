from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

SCHEMA_VERSION = "ekell-style-fire-v1"

# Paper-informed top-down decision-demand taxonomy. Labels marked substituted are
# fire-domain adaptations for this independent reproduction, not official E-KELL data.
DECISION_DEMANDS: dict[str, tuple[str, ...]] = {
    "incident_assessment": ("incident_identification", "severity_assessment", "development_forecast"),
    "hazard_assessment": ("hazard_identification", "exposure_assessment", "secondary_hazard_assessment"),
    "people_protection": ("evacuation", "rescue", "shelter_or_protection"),
    "response_objectives": ("life_safety", "incident_stabilization", "property_conservation"),
    "resource_coordination": ("personnel_assignment", "equipment_assignment", "mutual_aid"),
    "tactical_action": ("suppression_or_control", "containment_or_isolation", "search_and_access"),
    "information_coordination": ("monitoring", "communication"),
    "recovery_and_continuity": ("recovery", "service_continuity"),
}

ENTITY_TYPES = (
    "emergency_event", "hazard", "location", "person_or_group", "organization",
    "resource", "action", "decision_demand", "condition", "outcome", "information",
)
RELATION_TYPES = (
    "has_hazard", "occurs_at", "affects", "requires", "supports", "prevents",
    "uses", "performed_by", "depends_on", "has_condition", "results_in",
    "part_of", "related_to",
)
ENTITY_ATTRIBUTES = ("entity_id", "name", "entity_type", "aliases", "description")

ReviewStatus = Literal["candidate", "approved", "rejected"]


@dataclass(frozen=True)
class KGTriple:
    triple_id: str
    subject: str
    predicate: str
    object: str
    source_id: str
    chunk_id: str
    source_text: str
    extraction_method: str
    confidence: float
    review_status: ReviewStatus = "candidate"
    subject_type: str | None = None
    object_type: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KGTriple":
        required = (
            "triple_id", "subject", "predicate", "object", "source_id", "chunk_id",
            "source_text", "extraction_method", "confidence",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"missing triple fields: {', '.join(missing)}")
        return cls(
            **{key: value[key] for key in required},
            review_status=value.get("review_status", "candidate"),
            subject_type=value.get("subject_type"),
            object_type=value.get("object_type"),
            attributes=dict(value.get("attributes") or {}),
        )


def schema_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "basis": "Paper Section 3.1 top-down emergency decision-demand structure",
        "fidelity_note": (
            "Independent paper-informed scaffold; fire-domain labels are substitutions "
            "and must not be represented as official E-KELL schema or triples."
        ),
        "decision_demands": [
            {
                "primary": primary,
                "subclasses": list(subclasses),
                "substituted_for_fire_domain": True,
            }
            for primary, subclasses in DECISION_DEMANDS.items()
        ],
        "primary_demand_count": len(DECISION_DEMANDS),
        "subclass_demand_count": sum(map(len, DECISION_DEMANDS.values())),
        "entity_types": list(ENTITY_TYPES),
        "relation_types": list(RELATION_TYPES),
        "entity_attributes": list(ENTITY_ATTRIBUTES),
        "triple_fields": [
            "triple_id", "subject", "predicate", "object", "source_id", "chunk_id",
            "source_text", "extraction_method", "confidence", "review_status",
        ],
    }
