"""Shared LLM generation identity for fair five-method comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from external_baselines.common.strict_config_types import (
    read_exact_bool,
    read_exact_nonempty_string,
    require_exact_bool,
    require_exact_int,
    require_exact_number,
)

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
    formal = read_exact_bool(config, "paper_final", field="paper_final", default=False)
    seed_raw = llm.get("seed")
    if formal:
        seed = require_exact_int(seed_raw, field="llm.seed")
        provider = read_exact_nonempty_string(llm, "provider", field="llm.provider")
        model = read_exact_nonempty_string(llm, "model", field="llm.model")
        if "model_version" in llm:
            model_version = read_exact_nonempty_string(
                llm,
                "model_version",
                field="llm.model_version",
            )
        else:
            model_version = read_exact_nonempty_string(
                llm,
                "version",
                field="llm.model_version",
            )
        temperature = require_exact_number(llm.get("temperature"), field="llm.temperature")
        top_p = require_exact_number(llm.get("top_p"), field="llm.top_p")
        max_tokens = require_exact_int(llm.get("max_tokens"), field="llm.max_tokens")
        enable_thinking = (
            require_exact_bool(llm["enable_thinking"], field="llm.enable_thinking")
            if "enable_thinking" in llm
            else False
        )
    else:
        seed = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None
        provider = str(llm.get("provider") or "")
        model = str(llm.get("model") or "")
        model_version = str(llm.get("model_version") or llm.get("version") or "")
        temperature = float(llm.get("temperature") if llm.get("temperature") is not None else 0.0)
        top_p = float(llm.get("top_p") if llm.get("top_p") is not None else 1.0)
        max_tokens = int(llm.get("max_tokens") if llm.get("max_tokens") is not None else 0)
        enable_thinking = bool(llm.get("enable_thinking", False))
    return GenerationIdentity(
        provider=provider,
        model=model,
        model_version=model_version,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        enable_thinking=enable_thinking,
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


def runtime_identity_from_evidence(evidence: Any) -> GenerationIdentity:
    seed_raw = getattr(evidence, "llm_seed", None)
    seed = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None
    top_p = getattr(evidence, "llm_top_p", None)
    max_tokens = getattr(evidence, "llm_max_tokens", None)
    temperature = getattr(evidence, "llm_temperature", None)
    return GenerationIdentity(
        provider=str(getattr(evidence, "llm_provider", None) or ""),
        model=str(getattr(evidence, "llm_model", None) or ""),
        model_version=str(getattr(evidence, "llm_model_version", None) or ""),
        temperature=float(temperature if temperature is not None else 0.0),
        top_p=float(top_p if top_p is not None else 1.0),
        max_tokens=int(max_tokens if max_tokens is not None else 0),
        seed=seed,
        enable_thinking=bool(getattr(evidence, "llm_enable_thinking", False)),
    )


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
    reference = runtime_identity_from_evidence(ref_ev).to_dict() if ref_ev else None
    by_method: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []

    for method_id in method_ids:
        ev = method_evidences.get(method_id)
        current = runtime_identity_from_evidence(ev).to_dict() if ev else None
        by_method[method_id] = current
        if reference is None or current is None:
            mismatches.append(
                {
                    "error": "runtime_generation_identity_missing",
                    "method_id": method_id,
                }
            )
            continue
        for field in GENERATION_IDENTITY_FIELDS:
            if current.get(field) != reference.get(field):
                mismatches.append(
                    {
                        "error": "runtime_generation_identity_mismatch",
                        "field": field,
                        "reference_method": ref_id,
                        "reference_value": reference.get(field),
                        "method_id": method_id,
                        "method_value": current.get(field),
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
