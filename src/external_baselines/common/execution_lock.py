"""Deferred execution locks for dry-run and formal interop runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from external_baselines.common.formal_config_validator import _is_placeholder
from external_baselines.common.io import read_yaml
from external_baselines.common.main_project_readiness import assess_main_project_readiness

ExecutionStage = Literal["dry_run", "formal"]

LOCK_MESSAGE = (
    "Formal execution is currently locked.\n"
    "The baseline configuration is prepared, but the main project v1 Runner Bundle\n"
    "has not been approved for cross-repository testing."
)

DRY_RUN_LOCK_MESSAGE = (
    "Dry-run execution is currently locked.\n"
    "The baseline configuration is prepared, but readiness gates for a real dry run\n"
    "have not been satisfied."
)

DRY_RUN_LIMIT_MIN = 1
DRY_RUN_LIMIT_MAX = 10
DRY_RUN_OUTPUT_PREFIX = "outputs/dry_run/"
FORBIDDEN_DRY_RUN_PREFIXES = ("outputs/interop/", "outputs/formal/", "outputs/paper/")


class ExecutionLockError(SystemExit):
    """Raised when execution is blocked by readiness gates."""


def _bundle_is_placeholder(bundle: str | None) -> bool:
    if not bundle:
        return True
    return _is_placeholder(bundle)


def _normalize_output_path(path: str | Path | None) -> str:
    if not path:
        return ""
    return str(path).replace("\\", "/").lstrip("./")


def assert_execution_allowed(
    *,
    experiment_manifest: dict[str, Any] | None = None,
    bundle_path: str | None = None,
    resources_path: str | Path | None = None,
    execution_stage: ExecutionStage = "formal",
    limit: int | None = None,
    output_path: str | Path | None = None,
    allow_partial: bool = False,
    override_readiness_lock: bool = False,
) -> dict[str, Any]:
    """Gate dry_run vs formal interop execution.

    Returns an audit dict when execution is allowed.
    """
    stage = str(execution_stage or "formal").strip().lower()
    if stage not in {"dry_run", "formal"}:
        raise ExecutionLockError(f"Unknown execution_stage={execution_stage!r}; use dry_run or formal.")

    if override_readiness_lock:
        warning = (
            "WARNING: readiness lock bypassed via --override-readiness-lock. "
            "Not for CI or automation. Override does not make the run paper-valid."
        )
        print(warning, file=sys.stderr)
        return {
            "execution_lock_overridden": True,
            "execution_stage": stage,
            "warning": warning,
            "paper_valid": False,
        }

    resources_file = Path(resources_path or "configs/local/experiment_resources.yaml")
    resources: dict[str, Any] = {}
    reasons: list[str] = []
    if resources_file.is_file():
        resources = read_yaml(resources_file)
    else:
        reasons.append("experiment_resources_missing")

    execution = dict(resources.get("execution") or {})
    status = dict(resources.get("status") or {})
    bundle = bundle_path or (experiment_manifest or {}).get("bundle")
    output_norm = _normalize_output_path(output_path or (experiment_manifest or {}).get("output"))

    if _bundle_is_placeholder(str(bundle or "")):
        reasons.append("bundle_placeholder")
    if not resources_file.is_file() or execution.get("allow_real_model_calls") is False:
        reasons.append("allow_real_model_calls_false")
    if not resources_file.is_file() or execution.get("allow_cross_repo_test") is False:
        reasons.append("allow_cross_repo_test_false")

    readiness: dict[str, Any] = {}
    if resources_file.is_file():
        readiness = assess_main_project_readiness(resources_file)
        if not readiness.get("main_project_v1_ready"):
            reasons.append("main_project_v1_not_ready")

    if stage == "dry_run":
        if resources_file.is_file() and not readiness.get("safe_to_run_real_dry_run"):
            reasons.append("safe_to_run_real_dry_run_false")
        if limit is None:
            reasons.append("dry_run_limit_required")
        elif not (DRY_RUN_LIMIT_MIN <= int(limit) <= DRY_RUN_LIMIT_MAX):
            reasons.append("dry_run_limit_out_of_range")
        if not output_norm.startswith(DRY_RUN_OUTPUT_PREFIX):
            reasons.append("dry_run_output_path_invalid")
        if any(output_norm.startswith(prefix) for prefix in FORBIDDEN_DRY_RUN_PREFIXES):
            reasons.append("dry_run_output_in_formal_directory")
        # allow_partial defaults false; permitted only as explicit debug for dry_run.
    else:
        # formal
        if not resources_file.is_file() or execution.get("allow_formal_evaluation") is False:
            reasons.append("allow_formal_evaluation_false")
        if not bool(status.get("configs_frozen")):
            reasons.append("configs_not_frozen")
        if not bool(status.get("real_dry_run_completed")):
            reasons.append("real_dry_run_not_completed")
        if resources_file.is_file() and not readiness.get("safe_to_run_formal_experiment"):
            reasons.append("safe_to_run_formal_experiment_false")
        if limit is not None:
            reasons.append("formal_limit_forbidden")
        if allow_partial:
            reasons.append("formal_allow_partial_forbidden")
        if output_norm and not (
            output_norm.startswith("outputs/interop/")
            or output_norm.startswith("outputs/formal/")
        ):
            # Manifest formal outputs should live under interop/formal trees.
            if output_norm.startswith(DRY_RUN_OUTPUT_PREFIX):
                reasons.append("formal_output_in_dry_run_directory")

    if reasons:
        message = DRY_RUN_LOCK_MESSAGE if stage == "dry_run" else LOCK_MESSAGE
        detail = {
            "locked": True,
            "execution_stage": stage,
            "reasons": sorted(set(reasons)),
            "readiness": readiness,
        }
        raise ExecutionLockError(f"{message}\nLock detail: {detail}")

    return {
        "execution_lock_overridden": False,
        "execution_stage": stage,
        "readiness": readiness,
        "paper_valid": stage == "formal",
    }


def assert_formal_execution_allowed(
    *,
    experiment_manifest: dict[str, Any] | None = None,
    bundle_path: str | None = None,
    resources_path: str | Path | None = None,
    override_readiness_lock: bool = False,
    execution_stage: ExecutionStage = "formal",
    limit: int | None = None,
    output_path: str | Path | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper; defaults to formal stage."""
    return assert_execution_allowed(
        experiment_manifest=experiment_manifest,
        bundle_path=bundle_path,
        resources_path=resources_path,
        execution_stage=execution_stage,
        limit=limit,
        output_path=output_path,
        allow_partial=allow_partial,
        override_readiness_lock=override_readiness_lock,
    )
