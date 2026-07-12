"""Formal execution guard for the five-method decision comparison suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_baselines.common.formal_config_validator import (
    FormalConfigError,
    validate_experiment_manifest,
    validate_method_config,
)
from external_baselines.method_registry import canonicalize_method_id

FORMAL_EXECUTION_METHODS: tuple[str, ...] = (
    "direct_llm",
    "bm25_rag",
    "dense_rag",
    "hybrid_rag",
    "ekell_style_controlled_shared_llm",
)

MANIFEST_REQUIRED_MESSAGE = (
    "Formal execution requires a real experiment manifest. "
    "Heuristic/smoke defaults are only permitted for smoke or dry-run fixtures."
)

FORMAL_LIMIT_FORBIDDEN_MESSAGE = (
    "Formal execution forbids --limit. "
    "Formal runs must process the complete Runner Bundle case set. "
    "Use dry_run for partial-case diagnostics."
)

SMOKE_CONFIG_FORBIDDEN_MESSAGE = (
    "Formal execution cannot use the built-in smoke configuration."
)


class FormalSuiteExecutionError(ValueError):
    """Raised when formal suite preconditions are not met."""


class FormalConfigurationError(FormalConfigError):
    """Alias for formal configuration failures in the decision suite."""


def _is_example_manifest(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".example") or ".example" in name


def validate_decision_suite_execution(
    *,
    execution_stage: str,
    experiment_manifest: Path | None,
    method_ids: list[str],
    runner_bundle: Path | None = None,
    limit: int | None = None,
) -> None:
    """Validate suite execution contract before any LLM or index loading."""
    stage = str(execution_stage or "dry_run").strip().lower()
    if stage == "formal" and limit is not None:
        raise FormalSuiteExecutionError(FORMAL_LIMIT_FORBIDDEN_MESSAGE)
    if stage != "formal":
        return

    if experiment_manifest is None:
        raise FormalSuiteExecutionError(MANIFEST_REQUIRED_MESSAGE)
    if not experiment_manifest.is_file():
        raise FormalSuiteExecutionError(
            f"Formal execution requires a real experiment manifest (missing file: {experiment_manifest})."
        )
    if _is_example_manifest(experiment_manifest):
        raise FormalSuiteExecutionError(
            f"Formal execution rejects .example manifest paths: {experiment_manifest}"
        )

    canonical_methods = [canonicalize_method_id(m) for m in method_ids]
    expected = sorted(FORMAL_EXECUTION_METHODS)
    if sorted(canonical_methods) != expected:
        raise FormalSuiteExecutionError(
            f"Formal execution requires the exact five-method comparison set {expected}; "
            f"got {sorted(canonical_methods)}."
        )

    validate_experiment_manifest(
        experiment_manifest,
        validation_stage="formal",
        method_set="comparison_suite",
        runtime_bundle_path=runner_bundle,
    )

    from external_baselines.common.firebench_taxonomy import validate_formal_alias_table

    validate_formal_alias_table()


def assert_formal_smoke_config_forbidden(*, execution_stage: str) -> None:
    if str(execution_stage).strip().lower() == "formal":
        raise FormalConfigurationError(SMOKE_CONFIG_FORBIDDEN_MESSAGE)


def validate_formal_method_configs(
    *,
    method_ids: list[str],
    experiment_manifest: Path,
    runner_bundle: Path | None = None,
) -> dict[str, Any]:
    """Load and validate per-method configs from a formal manifest before runtime init."""
    from external_baselines.common.experiment_manifest import (
        build_method_config,
        get_method_entry,
        load_experiment_manifest,
    )

    experiment = load_experiment_manifest(experiment_manifest)
    reports: dict[str, Any] = {}
    for method_id in method_ids:
        method_entry = get_method_entry(experiment, method_id)
        cfg = build_method_config(experiment, method_entry)
        cfg["execution_stage"] = "formal"
        cfg["paper_final"] = True
        cfg["strict_decision_parse"] = True
        cfg["dev_aliases_enabled"] = False
        cfg["unified_decision_output"] = True
        cfg.setdefault("normalization", {})["infer_structured_safety_fields"] = False
        validate_method_config(
            cfg,
            method_id=method_id,
            allow_placeholders=False,
            require_formal=True,
            validation_stage="formal",
        )
        reports[method_id] = {
            "method_id": method_id,
            "config_validated": True,
            "dev_aliases_enabled": False,
            "strict_decision_parse": True,
        }
    return {
        "experiment_manifest": str(experiment_manifest),
        "runner_bundle": str(runner_bundle) if runner_bundle else None,
        "methods": reports,
    }
