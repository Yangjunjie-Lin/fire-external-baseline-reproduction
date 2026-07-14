from __future__ import annotations

"""Experiment manifest loader for fair paper runs.

Formal runs use a single experiment manifest that:
- references one shared model config
- assigns each method its own method config
- distinguishes main-table vs supplemental methods
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_json
from external_baselines.common.io import load_config, read_json, read_yaml
from external_baselines.common.path_resolution import PathContext, resolve_declared_path
from external_baselines.common.strict_config_types import require_exact_nonempty_string
from external_baselines.method_registry import (
    canonicalize_method_id,
    comparison_suite_methods,
    main_table_methods,
    paper_fidelity_methods,
    supplemental_methods,
)

MAIN_TABLE_METHODS = main_table_methods()
SUPPLEMENTAL_METHODS = supplemental_methods()
PAPER_FIDELITY_METHODS = paper_fidelity_methods()
COMPARISON_SUITE_METHODS = comparison_suite_methods()
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
METHOD_SETS = {
    "main_table": MAIN_TABLE_METHODS,
    "comparison_suite": COMPARISON_SUITE_METHODS,
}


class MethodEntryError(ValueError):
    """Raised when an experiment manifest lacks a usable method entry."""


EXPERIMENT_FREEZE_LIFECYCLE_FIELDS = frozenset({"freeze_status", "freeze_manifest"})


def canonical_experiment_core(experiment_raw: dict[str, Any]) -> dict[str, Any]:
    """Return experiment semantics while excluding only freeze lifecycle state."""
    if not isinstance(experiment_raw, dict):
        raise TypeError("canonical_experiment_core requires a mapping")
    core = deepcopy(experiment_raw)
    for field in EXPERIMENT_FREEZE_LIFECYCLE_FIELDS:
        core.pop(field, None)
    return core


def experiment_core_sha256(experiment_raw: dict[str, Any]) -> str:
    """Identity stable across provisional-to-frozen manifest finalization."""
    return sha256_json(canonical_experiment_core(experiment_raw))


def canonical_method_config_for_freeze(config: dict[str, Any]) -> dict[str, Any]:
    """Return a merged method config without experiment freeze lifecycle state."""
    if not isinstance(config, dict):
        raise TypeError("canonical_method_config_for_freeze requires a mapping")
    canonical = deepcopy(config)
    experiment = canonical.get("experiment")
    if isinstance(experiment, dict):
        for field in EXPERIMENT_FREEZE_LIFECYCLE_FIELDS:
            experiment.pop(field, None)
    return canonical


def merged_method_config_sha256(config: dict[str, Any]) -> str:
    return sha256_json(canonical_method_config_for_freeze(config))


def _manifest_exact_bool(raw: dict[str, Any], key: str, *, path: str) -> None:
    if key in raw and type(raw[key]) is not bool:
        raise ValueError(f"Experiment manifest {path} must be an exact boolean")


def _method_enabled(entry: dict[str, Any], *, index: int) -> bool:
    if "enabled" not in entry:
        return True
    raw_enabled = entry["enabled"]
    if type(raw_enabled) is not bool:
        raise ValueError(
            f"Experiment manifest methods[{index}].enabled must be an exact boolean"
        )
    return raw_enabled


def _manifest_string(raw: dict[str, Any], key: str, *, default: str | None = None) -> str:
    if key not in raw:
        if default is None:
            raise ValueError(f"Experiment manifest requires {key}")
        return default
    return require_exact_nonempty_string(raw[key], field=key)


def _optional_manifest_string(raw: dict[str, Any], key: str) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    return require_exact_nonempty_string(raw[key], field=key)


def get_method_entry(
    manifest: dict[str, Any],
    method_id: str,
    *,
    require_enabled: bool = True,
) -> dict[str, Any]:
    """Return the resolved method entry dict for a canonical method_id."""
    mid = canonicalize_method_id(method_id)
    for entry in manifest.get("methods") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("method_id") == mid:
            if require_enabled and not entry.get("enabled", True):
                raise MethodEntryError(f"Experiment manifest method entry disabled: {mid}")
            return entry
    raise MethodEntryError(f"Experiment manifest does not define method entry: {mid}")


def _resolve_experiment_resource(
    declared: str,
    *,
    manifest_path: Path,
) -> tuple[str, str]:
    """Prefer manifest-relative resources, then deterministic repository-relative ones."""
    candidate = Path(declared)
    context = PathContext(
        repository_root=REPOSITORY_ROOT,
        experiment_manifest_path=manifest_path,
    )
    if candidate.is_absolute():
        return str(candidate.resolve(strict=False)), "absolute"
    experiment_candidate = resolve_declared_path(
        declared,
        context=context,
        policy="experiment_relative",
        must_exist=False,
    )
    if experiment_candidate.exists():
        return str(experiment_candidate), "experiment_relative"
    repository_candidate = resolve_declared_path(
        declared,
        context=context,
        policy="repository_relative",
        must_exist=False,
    )
    return str(repository_candidate), "repository_relative"

def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = resolve_declared_path(
            path,
            context=PathContext(repository_root=REPOSITORY_ROOT),
            policy="repository_relative",
            must_exist=False,
        )
    else:
        path = path.resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(path)
    name = path.name.lower()
    if path.suffix.lower() in {".yaml", ".yml"} or name.endswith(".yaml.example") or name.endswith(".yml.example"):
        raw = read_yaml(path)
    else:
        raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment manifest must be a mapping: {path}")

    shared_model_declared = _manifest_string(raw, "shared_model_config")
    shared_model, shared_policy = _resolve_experiment_resource(
        shared_model_declared,
        manifest_path=path,
    )
    base_declared = _optional_manifest_string(raw, "base_config") or "configs/default.yaml"
    base_config, base_policy = _resolve_experiment_resource(
        base_declared,
        manifest_path=path,
    )

    methods = raw.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Experiment manifest requires non-empty methods list")

    for key in ("paper_final", "require_bundle_checksum", "require_external_schema"):
        _manifest_exact_bool(raw, key, path=key)

    resolved: list[dict[str, Any]] = []
    for index, entry in enumerate(methods):
        if isinstance(entry, str):
            entry = {"method_id": entry}
        if not isinstance(entry, dict) or "method_id" not in entry:
            raise ValueError(f"Invalid method entry in experiment manifest: {entry}")
        method_id = canonicalize_method_id(
            require_exact_nonempty_string(
                entry["method_id"],
                field=f"methods[{index}].method_id",
            )
        )
        if entry.get("paper_table_role"):
            role = require_exact_nonempty_string(
                entry["paper_table_role"],
                field=f"methods[{index}].paper_table_role",
            )
        elif method_id in MAIN_TABLE_METHODS:
            role = "main_table"
        elif method_id in PAPER_FIDELITY_METHODS:
            role = "paper_fidelity"
        else:
            role = "supplemental_extended"
        if "config" in entry:
            method_config_path = entry["config"]
        else:
            method_config_path = entry.get("method_config")
        if method_config_path is not None:
            method_config_path = require_exact_nonempty_string(
                method_config_path,
                field=f"methods[{index}].config",
            )
            method_config_declared = method_config_path
            method_config_path, method_config_policy = _resolve_experiment_resource(
                method_config_declared,
                manifest_path=path,
            )
        else:
            method_config_declared = None
            method_config_policy = None
        resolved.append({
            "method_id": method_id,
            "config": method_config_path,
            "config_declared": method_config_declared,
            "config_path_policy": method_config_policy,
            "paper_table_role": role,
            "enabled": _method_enabled(entry, index=index),
        })

    return {
        "manifest_path": str(path),
        "experiment_id": _optional_manifest_string(raw, "experiment_id") or path.stem,
        "schema_version": _optional_manifest_string(raw, "schema_version") or "firebench-interop-v1",
        "track": _optional_manifest_string(raw, "track") or "A_shared_outcome",
        "shared_model_config": str(shared_model),
        "shared_model_config_declared": shared_model_declared,
        "base_config": base_declared,
        "base_config_resolved": base_config,
        "base_config_declared": base_declared,
        "path_provenance": {
            "base_config": {
                "declared_path": base_declared,
                "resolved_path": base_config,
                "path_policy": base_policy,
            },
            "shared_model_config": {
                "declared_path": shared_model_declared,
                "resolved_path": shared_model,
                "path_policy": shared_policy,
            },
            "method_configs": {
                entry["method_id"]: {
                    "declared_path": entry.get("config_declared"),
                    "resolved_path": entry.get("config"),
                    "path_policy": entry.get("config_path_policy"),
                }
                for entry in resolved
                if entry.get("config")
            },
        },
        "methods": resolved,
        "main_table_methods": list(raw.get("main_table_methods") or MAIN_TABLE_METHODS),
        "comparison_suite_methods": list(raw.get("comparison_suite_methods") or COMPARISON_SUITE_METHODS),
        "comparison_suite_methods_explicit": (
            "comparison_suite_methods" in raw
            and raw.get("comparison_suite_methods") is not None
        ),
        "supplemental_methods": list(raw.get("supplemental_methods") or SUPPLEMENTAL_METHODS),
        "bundle": _optional_manifest_string(raw, "bundle"),
        "freeze_manifest": _optional_manifest_string(raw, "freeze_manifest"),
        "expected_bundle_checksum": raw.get("expected_bundle_checksum"),
        "output": _optional_manifest_string(raw, "output") or "outputs/firebench_interop_v1_predictions.jsonl",
        "legacy_output": _optional_manifest_string(raw, "legacy_output") or "outputs/baseline_outputs_legacy.jsonl",
        "run_manifest": _optional_manifest_string(raw, "run_manifest") or "outputs/interop_run_manifest.json",
        "limit": raw.get("limit"),
        "paper_final": raw.get("paper_final", False),
        "require_bundle_checksum": raw.get("require_bundle_checksum", True),
        "notes": raw.get("notes") or [],
        "freeze_status": _optional_manifest_string(raw, "freeze_status") or "provisional",
        "raw": raw,
    }


def build_method_config(manifest: dict[str, Any], method_entry: dict[str, Any]) -> dict[str, Any]:
    """Merge order (later wins): base → shared_model → method_config → manifest paper flags."""
    if not isinstance(method_entry, dict):
        raise TypeError(
            "build_method_config requires a method entry mapping, "
            f"got {type(method_entry).__name__}"
        )
    if not method_entry.get("method_id"):
        raise TypeError("build_method_config requires method_entry with method_id")
    paths = [
        manifest.get("base_config_resolved") or manifest["base_config"],
        manifest["shared_model_config"],
    ]
    if method_entry.get("config"):
        paths.append(method_entry["config"])
    config = load_config(*paths)
    config["paper_final"] = manifest.get("paper_final", config.get("paper_final", False))
    config["require_bundle_checksum"] = manifest.get(
        "require_bundle_checksum", config.get("require_bundle_checksum", False)
    )
    config.setdefault("experiment", {})
    config["experiment"]["experiment_id"] = manifest.get("experiment_id")
    config["experiment"]["paper_table_role"] = method_entry.get("paper_table_role")
    config["experiment"]["freeze_status"] = manifest.get("freeze_status")
    return config


def resolve_method_set(
    manifest: dict[str, Any],
    *,
    method_set: str = "main_table",
    include_supplemental: bool = False,
) -> list[str]:
    """Resolve ordered method IDs for a named method set."""
    name = str(method_set or "main_table").strip().lower()
    if include_supplemental and name == "main_table":
        # Deprecated path: main + supplemental_extended entries.
        name = "comparison_suite"
    if name == "main_table":
        declared = [canonicalize_method_id(str(m)) for m in (manifest.get("main_table_methods") or MAIN_TABLE_METHODS)]
        return declared
    if name == "comparison_suite":
        declared = [
            canonicalize_method_id(str(m))
            for m in (manifest.get("comparison_suite_methods") or COMPARISON_SUITE_METHODS)
        ]
        forbidden = set(paper_fidelity_methods()) | {"ekell_style_enhanced", "lightrag", "microsoft_graphrag", "fallback_graph_retrieval"}
        cleaned = [m for m in declared if m not in forbidden]
        if cleaned != list(COMPARISON_SUITE_METHODS) and not manifest.get("comparison_suite_methods"):
            return list(COMPARISON_SUITE_METHODS)
        return cleaned
    raise ValueError(f"Unknown method_set={method_set!r}; use main_table or comparison_suite.")


def enabled_methods(
    manifest: dict[str, Any],
    *,
    include_supplemental: bool = False,
    method_set: str | None = None,
) -> list[dict[str, Any]]:
    selected_ids = None
    if method_set:
        selected_ids = set(resolve_method_set(manifest, method_set=method_set, include_supplemental=include_supplemental))
    elif include_supplemental:
        selected_ids = set(resolve_method_set(manifest, method_set="comparison_suite"))

    out: list[dict[str, Any]] = []
    by_id = {entry["method_id"]: entry for entry in manifest["methods"]}
    if selected_ids is not None:
        for mid in resolve_method_set(
            manifest,
            method_set=method_set or ("comparison_suite" if include_supplemental else "main_table"),
        ):
            entry = by_id.get(mid)
            if entry is None:
                raise ValueError(f"method_set references missing method entry: {mid}")
            # comparison_suite may enable dense/hybrid even if enabled=false in template.
            out.append({**entry, "enabled": True})
        return out

    for entry in manifest["methods"]:
        if not entry.get("enabled", True):
            continue
        role = entry.get("paper_table_role")
        if role == "supplemental_extended" and not include_supplemental:
            continue
        out.append(entry)
    return out
