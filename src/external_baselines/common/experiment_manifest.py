from __future__ import annotations

"""Experiment manifest loader for fair paper runs.

Formal runs use a single experiment manifest that:
- references one shared model config
- assigns each method its own method config
- distinguishes main-table vs supplemental methods
"""

from pathlib import Path
from typing import Any

from external_baselines.common.io import deep_merge, load_config, read_json, read_yaml
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
METHOD_SETS = {
    "main_table": MAIN_TABLE_METHODS,
    "comparison_suite": COMPARISON_SUITE_METHODS,
}

def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    name = path.name.lower()
    if path.suffix.lower() in {".yaml", ".yml"} or name.endswith(".yaml.example") or name.endswith(".yml.example"):
        raw = read_yaml(path)
    else:
        raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment manifest must be a mapping: {path}")

    shared_model = raw.get("shared_model_config")
    if not shared_model:
        raise ValueError("Experiment manifest requires shared_model_config")

    methods = raw.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Experiment manifest requires non-empty methods list")

    resolved: list[dict[str, Any]] = []
    for entry in methods:
        if isinstance(entry, str):
            entry = {"method_id": entry}
        if not isinstance(entry, dict) or not entry.get("method_id"):
            raise ValueError(f"Invalid method entry in experiment manifest: {entry}")
        method_id = canonicalize_method_id(str(entry["method_id"]))
        if entry.get("paper_table_role"):
            role = str(entry["paper_table_role"])
        elif method_id in MAIN_TABLE_METHODS:
            role = "main_table"
        elif method_id in PAPER_FIDELITY_METHODS:
            role = "paper_fidelity"
        else:
            role = "supplemental_extended"
        method_config_path = entry.get("config") or entry.get("method_config")
        resolved.append({
            "method_id": method_id,
            "config": method_config_path,
            "paper_table_role": role,
            "enabled": bool(entry.get("enabled", True)),
        })

    return {
        "manifest_path": str(path),
        "experiment_id": raw.get("experiment_id") or path.stem,
        "schema_version": raw.get("schema_version") or "firebench-interop-v1",
        "track": raw.get("track") or "A_shared_outcome",
        "shared_model_config": str(shared_model),
        "base_config": raw.get("base_config") or "configs/default.yaml",
        "methods": resolved,
        "main_table_methods": list(raw.get("main_table_methods") or MAIN_TABLE_METHODS),
        "comparison_suite_methods": list(raw.get("comparison_suite_methods") or COMPARISON_SUITE_METHODS),
        "supplemental_methods": list(raw.get("supplemental_methods") or SUPPLEMENTAL_METHODS),
        "bundle": raw.get("bundle"),
        "freeze_manifest": raw.get("freeze_manifest"),
        "expected_bundle_checksum": raw.get("expected_bundle_checksum"),
        "output": raw.get("output") or "outputs/firebench_interop_v1_predictions.jsonl",
        "legacy_output": raw.get("legacy_output") or "outputs/baseline_outputs_legacy.jsonl",
        "run_manifest": raw.get("run_manifest") or "outputs/interop_run_manifest.json",
        "limit": raw.get("limit"),
        "paper_final": bool(raw.get("paper_final", False)),
        "require_bundle_checksum": bool(raw.get("require_bundle_checksum", True)),
        "notes": raw.get("notes") or [],
        "freeze_status": raw.get("freeze_status") or "provisional",
        "raw": raw,
    }


def build_method_config(manifest: dict[str, Any], method_entry: dict[str, Any]) -> dict[str, Any]:
    """Merge order (later wins): base → shared_model → method_config → manifest paper flags."""
    paths = [manifest["base_config"], manifest["shared_model_config"]]
    if method_entry.get("config"):
        paths.append(method_entry["config"])
    config = load_config(*paths)
    config["paper_final"] = bool(manifest.get("paper_final", config.get("paper_final", False)))
    config["require_bundle_checksum"] = bool(
        manifest.get("require_bundle_checksum", config.get("require_bundle_checksum", False))
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
