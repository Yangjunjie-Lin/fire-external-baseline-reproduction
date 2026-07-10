"""Deferred execution locks for formal interop runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_baselines.common.formal_config_validator import _is_placeholder
from external_baselines.common.io import read_yaml
from external_baselines.common.main_project_readiness import assess_main_project_readiness

LOCK_MESSAGE = (
    "Formal execution is currently locked.\n"
    "The baseline configuration is prepared, but the main project v1 Runner Bundle\n"
    "has not been approved for cross-repository testing."
)


class ExecutionLockError(SystemExit):
    """Raised when formal execution is blocked by readiness gates."""


def _bundle_is_placeholder(bundle: str | None) -> bool:
    if not bundle:
        return True
    return _is_placeholder(bundle)


def assert_formal_execution_allowed(
    *,
    experiment_manifest: dict[str, Any] | None = None,
    bundle_path: str | None = None,
    resources_path: str | Path | None = None,
    override_readiness_lock: bool = False,
) -> dict[str, Any]:
    """Block formal interop when preparation locks are active.

    Returns a small audit dict when execution is allowed.
    """
    if override_readiness_lock:
        return {
            "execution_lock_overridden": True,
            "warning": "Readiness lock bypassed manually; not for CI or automation.",
        }

    resources_file = Path(resources_path or "configs/local/experiment_resources.yaml")
    resources: dict[str, Any] = {}
    reasons: list[str] = []
    if resources_file.is_file():
        resources = read_yaml(resources_file)
    else:
        reasons.append("experiment_resources_missing")

    execution = dict(resources.get("execution") or {})
    bundle = bundle_path or (experiment_manifest or {}).get("bundle")

    if _bundle_is_placeholder(str(bundle or "")):
        reasons.append("bundle_placeholder")
    if not resources_file.is_file() or execution.get("allow_real_model_calls") is False:
        reasons.append("allow_real_model_calls_false")
    if not resources_file.is_file() or execution.get("allow_cross_repo_test") is False:
        reasons.append("allow_cross_repo_test_false")
    if execution.get("allow_formal_evaluation") is False and bool(
        (experiment_manifest or {}).get("paper_final")
    ):
        reasons.append("allow_formal_evaluation_false")

    readiness: dict[str, Any] = {}
    if resources_file.is_file():
        readiness = assess_main_project_readiness(resources_file)
        if not readiness.get("main_project_v1_ready"):
            reasons.append("main_project_v1_not_ready")
        if not readiness.get("safe_to_run_real_dry_run"):
            reasons.append("safe_to_run_real_dry_run_false")

    if reasons:
        detail = {
            "locked": True,
            "reasons": sorted(set(reasons)),
            "readiness": readiness,
        }
        raise ExecutionLockError(f"{LOCK_MESSAGE}\nLock detail: {detail}")

    return {
        "execution_lock_overridden": False,
        "readiness": readiness,
    }
