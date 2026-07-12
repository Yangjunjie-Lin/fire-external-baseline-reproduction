"""Shared LLM generation identity for fair five-method comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GENERATION_IDENTITY_FIELDS = (
    "provider",
    "model",
    "model_version",
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "enable_thinking",
)

LLM_OVERRIDE_FIELDS = GENERATION_IDENTITY_FIELDS


@dataclass(frozen=True)
class GenerationIdentity:
    provider: str
    model: str
    model_version: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None
    enable_thinking: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_generation_identity(config: dict[str, Any]) -> GenerationIdentity:
    llm = config.get("llm") or {}
    seed_raw = llm.get("seed")
    seed = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None
    return GenerationIdentity(
        provider=str(llm.get("provider") or ""),
        model=str(llm.get("model") or ""),
        model_version=str(llm.get("model_version") or llm.get("version") or ""),
        temperature=float(llm.get("temperature") if llm.get("temperature") is not None else 0.0),
        top_p=float(llm.get("top_p") if llm.get("top_p") is not None else 1.0),
        max_tokens=int(llm.get("max_tokens") if llm.get("max_tokens") is not None else 0),
        seed=seed,
        enable_thinking=bool(llm.get("enable_thinking", False)),
    )


def generation_identities_match(left: GenerationIdentity, right: GenerationIdentity) -> bool:
    return left == right


def validate_shared_generation_identity(
    *,
    method_ids: list[str],
    method_configs: dict[str, dict[str, Any]],
    reference_method_id: str | None = None,
) -> dict[str, Any]:
    """Ensure all methods share identical generation model identity."""
    if not method_ids:
        return {"ok": True, "reference": None, "methods": {}, "mismatches": []}

    ref_id = reference_method_id or method_ids[0]
    reference = extract_generation_identity(method_configs.get(ref_id) or {})
    by_method: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []

    for method_id in method_ids:
        identity = extract_generation_identity(method_configs.get(method_id) or {})
        by_method[method_id] = identity.to_dict()
        if identity == reference:
            continue
        for field in GENERATION_IDENTITY_FIELDS:
            if getattr(identity, field) != getattr(reference, field):
                mismatches.append(
                    {
                        "error": "shared_generation_identity_mismatch",
                        "field": field,
                        "reference_method": ref_id,
                        "reference_value": getattr(reference, field),
                        "method_id": method_id,
                        "method_value": getattr(identity, field),
                    }
                )

    return {
        "ok": not mismatches,
        "reference": reference.to_dict(),
        "methods": by_method,
        "mismatches": mismatches,
    }


def validate_runtime_generation_identity(
    *,
    method_ids: list[str],
    method_evidences: dict[str, Any],
    reference_method_id: str | None = None,
) -> dict[str, Any]:
    """Compare runtime LLM identity recorded after client initialization."""
    if not method_ids:
        return {"ok": True, "reference": None, "methods": {}, "mismatches": []}

    ref_id = reference_method_id or method_ids[0]
    ref_ev = method_evidences.get(ref_id)
    reference = {
        "provider": getattr(ref_ev, "llm_provider", None) if ref_ev else None,
        "model": getattr(ref_ev, "llm_model", None) if ref_ev else None,
        "model_version": getattr(ref_ev, "llm_model_version", None) if ref_ev else None,
    }
    by_method: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []

    for method_id in method_ids:
        ev = method_evidences.get(method_id)
        current = {
            "provider": getattr(ev, "llm_provider", None) if ev else None,
            "model": getattr(ev, "llm_model", None) if ev else None,
            "model_version": getattr(ev, "llm_model_version", None) if ev else None,
        }
        by_method[method_id] = current
        for field in ("provider", "model", "model_version"):
            if current[field] != reference[field]:
                mismatches.append(
                    {
                        "error": "runtime_generation_identity_mismatch",
                        "field": field,
                        "reference_method": ref_id,
                        "reference_value": reference[field],
                        "method_id": method_id,
                        "method_value": current[field],
                    }
                )

    return {
        "ok": not mismatches,
        "reference": reference,
        "methods": by_method,
        "mismatches": mismatches,
    }


def detect_method_llm_overrides(
    *,
    shared_config: dict[str, Any],
    method_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Detect method-level llm fields that differ from shared model config."""
    shared = extract_generation_identity(shared_config)
    method = extract_generation_identity(method_config)
    overrides: list[dict[str, Any]] = []
    for field in LLM_OVERRIDE_FIELDS:
        if getattr(method, field) != getattr(shared, field):
            overrides.append(
                {
                    "error": "method_config_overrides_shared_llm",
                    "field": field,
                    "shared_value": getattr(shared, field),
                    "method_value": getattr(method, field),
                }
            )
    return overrides
