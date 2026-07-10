"""Validate formal experiment and method configs before paper-facing runs.

LOCAL GUARD — rejects heuristic LLM, smoke embeddings, placeholders, and legacy IDs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from external_baselines.common.experiment_manifest import build_method_config, load_experiment_manifest
from external_baselines.common.io import read_yaml
from external_baselines.method_registry import (
    canonicalize_method_id,
    fallback_methods,
    legacy_methods,
    main_table_methods,
    paper_fidelity_methods,
)

PLACEHOLDER_TOKENS = frozenset(
    {
        "REQUIRED_BEFORE_FORMAL_RUN",
        "TBD",
        "TODO",
        "path/to/",
        "<required>",
        "CHANGEME",
    }
)

SMOKE_LLM_PROVIDERS = frozenset({"heuristic", "local", "smoke", ""})
SMOKE_EMBEDDING_BACKENDS = frozenset(
    {
        "smoke",
        "hash",
        "smoke_hash",
        "hash_smoke",
        "deterministic_hash_smoke",
        "smoke_hash_embedding",
    }
)

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9]{16,}", re.I),
)


class FormalConfigError(ValueError):
    """Raised when a formal config violates paper-facing safety rules."""


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    upper = text.upper()
    if upper in PLACEHOLDER_TOKENS:
        return True
    return any(token in text for token in PLACEHOLDER_TOKENS)


def _walk_strings(obj: Any, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(_walk_strings(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found.extend(_walk_strings(item, f"{path}[{i}]"))
    elif isinstance(obj, str):
        found.append((path, obj))
    return found


def _check_no_secrets(config: dict[str, Any]) -> None:
    for path, value in _walk_strings(config):
        for pattern in SECRET_PATTERNS:
            if pattern.search(value):
                raise FormalConfigError(f"Possible secret at {path}; use env vars instead.")


def _llm_block(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm", {})
    return dict(llm) if isinstance(llm, dict) else {}


def validate_llm_for_formal(config: dict[str, Any], *, allow_placeholders: bool = False) -> None:
    llm = _llm_block(config)
    provider = str(llm.get("provider", "heuristic")).lower().strip()
    model = str(llm.get("model") or "").strip()
    model_version = str(llm.get("model_version") or llm.get("version") or "").strip()

    if provider in SMOKE_LLM_PROVIDERS:
        raise FormalConfigError(f"Formal config rejects smoke LLM provider: {provider!r}")
    if _is_placeholder(model) and not allow_placeholders:
        raise FormalConfigError("Formal config requires llm.model (non-placeholder).")
    if _is_placeholder(model_version) and not allow_placeholders:
        raise FormalConfigError("Formal config requires llm.model_version (non-placeholder).")


def validate_ekell_vector_for_formal(
    config: dict[str, Any], *, allow_placeholders: bool = False
) -> None:
    vector = config.get("ekell_vector") or (config.get("ekell_style") or {}).get("vector") or {}
    if not isinstance(vector, dict) or not vector:
        raise FormalConfigError("Formal E-KELL config requires explicit ekell_vector block.")
    backend = str(vector.get("backend", "smoke")).casefold().replace("-", "_")
    if backend in SMOKE_EMBEDDING_BACKENDS:
        raise FormalConfigError(f"Formal E-KELL rejects smoke/hash backend: {backend!r}")
    if not bool(vector.get("reject_smoke", False)):
        raise FormalConfigError("Formal E-KELL requires ekell_vector.reject_smoke=true.")
    for field in ("model_name", "model_version", "dimension"):
        value = vector.get(field)
        if _is_placeholder(value) and not allow_placeholders:
            raise FormalConfigError(f"Formal E-KELL requires ekell_vector.{field} (non-placeholder).")


def validate_method_config(
    config: dict[str, Any],
    *,
    method_id: str | None = None,
    allow_placeholders: bool = False,
    require_formal: bool = True,
) -> list[str]:
    """Return warnings; raise FormalConfigError on hard violations."""
    warnings: list[str] = []
    mid = canonicalize_method_id(method_id or str(config.get("method_id") or ""))
    if mid in legacy_methods():
        raise FormalConfigError(f"Legacy method {mid!r} cannot be used in formal configs.")
    if mid in fallback_methods():
        raise FormalConfigError(f"Fallback method {mid!r} cannot enter formal main table.")

    if str(config.get("method_id") or "") != mid and config.get("method_id"):
        if str(config.get("method_id")) != mid:
            warnings.append(f"method_id canonicalized {config.get('method_id')!r} → {mid!r}")

    paper_final = bool(config.get("paper_final", False))
    if require_formal or paper_final:
        validate_llm_for_formal(config, allow_placeholders=allow_placeholders)
        if mid == "ekell_style_controlled_shared_llm":
            validate_ekell_vector_for_formal(config, allow_placeholders=allow_placeholders)
        enhanced = config.get("ekell_style") or {}
        if mid == "ekell_style_controlled_shared_llm":
            for flag in (
                "dense_entity_retrieval",
                "hybrid_subgraph_ranking",
                "reranker",
                "self_consistency",
                "structured_verification",
            ):
                if bool(enhanced.get(flag)):
                    raise FormalConfigError(
                        f"Controlled E-KELL forbids enhanced hook ekell_style.{flag}=true"
                    )

    _check_no_secrets(config)
    return warnings


def validate_experiment_manifest(
    path: str | Path,
    *,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    manifest = load_experiment_manifest(path)
    raw = manifest.get("raw") or read_yaml(path)
    if not isinstance(raw, dict):
        raise FormalConfigError(f"Experiment manifest must be a mapping: {path}")

    errors: list[str] = []
    run_mode = str(raw.get("run_mode", "formal")).lower()
    if run_mode != "formal":
        errors.append(f"run_mode must be 'formal' for paper-facing manifest (got {run_mode!r})")

    if not bool(raw.get("paper_final", False)):
        errors.append("paper_final must be true for formal experiment manifest.")

    for key in (
        "require_bundle_checksum",
        "require_external_schema",
        "require_complete_case_match",
        "fail_on_schema_error",
        "fail_on_duplicate_case_id",
        "fail_on_missing_case",
        "fail_on_extra_case",
    ):
        if not bool(raw.get(key, False)):
            errors.append(f"{key} must be true for formal manifest.")

    freeze_status = str(raw.get("freeze_status", "")).lower()
    if freeze_status in {"frozen", "paper_ready", "empirically_validated"}:
        errors.append(f"freeze_status must remain provisional (got {freeze_status!r}).")

    shared = str(raw.get("shared_model_config") or "")
    if not shared:
        errors.append("shared_model_config is required.")
    elif "smoke" in shared.lower() or "heuristic" in shared.lower():
        errors.append("shared_model_config must not reference smoke/heuristic configs.")

    methods = raw.get("methods") or []
    if not isinstance(methods, list) or not methods:
        errors.append("methods list is required and non-empty.")

    seen_ids: set[str] = set()
    for entry in methods:
        if isinstance(entry, str):
            entry = {"method_id": entry}
        if not isinstance(entry, dict):
            errors.append(f"Invalid method entry: {entry!r}")
            continue
        mid = canonicalize_method_id(str(entry.get("method_id") or ""))
        if mid in seen_ids:
            errors.append(f"Duplicate method_id in manifest: {mid}")
        seen_ids.add(mid)
        if str(entry.get("method_id") or "") == "ekell_style_faithful":
            errors.append("Manifest must use canonical ekell_style_controlled_shared_llm, not faithful.")
        if bool(entry.get("enabled", True)) and mid in main_table_methods():
            if not (entry.get("config") or entry.get("method_config")):
                errors.append(f"Main-table method {mid} missing config path.")
            else:
                merged = build_method_config(manifest, entry)
                merged["paper_final"] = True
                try:
                    validate_method_config(
                        merged,
                        method_id=mid,
                        allow_placeholders=allow_placeholders,
                        require_formal=True,
                    )
                except FormalConfigError as exc:
                    errors.append(f"{mid} merged config: {exc}")

    declared_main = [canonicalize_method_id(str(m)) for m in (raw.get("main_table_methods") or [])]
    expected_main = list(main_table_methods())
    enabled_main = [
        canonicalize_method_id(str(e.get("method_id")))
        for e in methods
        if isinstance(e, dict) and e.get("enabled", True)
        and canonicalize_method_id(str(e.get("method_id") or "")) in expected_main
    ]
    if enabled_main and declared_main != expected_main:
        errors.append(f"main_table_methods must be exactly {expected_main} (got {declared_main})")

    pf_expected = list(paper_fidelity_methods())
    declared_pf = [canonicalize_method_id(str(m)) for m in (raw.get("paper_fidelity_methods") or [])]
    enabled_pf = [
        canonicalize_method_id(str(e.get("method_id")))
        for e in methods
        if isinstance(e, dict) and e.get("enabled", True)
        and canonicalize_method_id(str(e.get("method_id") or "")) in pf_expected
    ]
    if enabled_pf and declared_pf != pf_expected:
        errors.append(f"paper_fidelity_methods must be exactly {pf_expected} (got {declared_pf})")

    if errors:
        raise FormalConfigError("Formal manifest validation failed:\n- " + "\n- ".join(errors))

    return {
        "path": str(path),
        "experiment_id": raw.get("experiment_id"),
        "valid": True,
        "allow_placeholders": allow_placeholders,
    }


# Repo root helper for relative config paths (unused after manifest merge; kept for imports)
ROOT_REL = Path(__file__).resolve().parents[3]
