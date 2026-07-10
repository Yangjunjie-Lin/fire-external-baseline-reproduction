"""Main-project v1 readiness checks (read-only; never modifies fire-agent-demo)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file
from external_baselines.common.io import read_yaml

ROOT = Path(__file__).resolve().parents[3]

READINESS_MARKER_CANDIDATES = (
    "artifacts/status/first_model_v1_ready.json",
    "artifacts/firebench_interop_v1/runner_v1/manifest.json",
)

BUNDLE_MANIFEST_NAMES = ("manifest.json", "runner_manifest.json", "bundle_manifest.json")
INPUT_CASES_NAMES = ("input_cases.jsonl", "scenarios.jsonl")
SCHEMA_NAMES = ("prediction_schema.json", "schema.json")


def _git(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _resolve_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _discover_runner_bundle(main_repo: Path, resources: dict[str, Any]) -> Path | None:
    """Informational discovery only — never used to mark v1 ready."""
    main_cfg = resources.get("main_project") or {}
    for rel in main_cfg.get("runner_bundle_candidates") or []:
        candidate = _resolve_path(main_repo, str(rel))
        if candidate and candidate.is_dir():
            return candidate

    for rel in READINESS_MARKER_CANDIDATES:
        marker = main_repo / rel
        if not marker.is_file():
            continue
        if marker.name == "manifest.json" and "runner" in rel:
            return marker.parent
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        bundle_ref = payload.get("runner_bundle_path") or payload.get("bundle_path")
        if bundle_ref:
            candidate = _resolve_path(main_repo, str(bundle_ref))
            if candidate and candidate.is_dir():
                return candidate
        if payload.get("ready") is True and "runner_v1" in rel:
            return marker.parent

    seed = main_repo / "artifacts" / "firebench_interop_v1" / "runner_seed_curated"
    if seed.is_dir():
        return seed
    runner_v1 = main_repo / "artifacts" / "firebench_interop_v1" / "runner_v1"
    if runner_v1.is_dir():
        return runner_v1
    return None


def _bundle_artifacts(bundle_root: Path) -> dict[str, Any]:
    manifest = _first_existing(bundle_root, BUNDLE_MANIFEST_NAMES)
    input_cases = _first_existing(bundle_root, INPUT_CASES_NAMES)
    if not input_cases:
        for sub in ("input", "cases", "scenarios"):
            nested = bundle_root / sub
            if nested.is_dir():
                found = _first_existing(nested, INPUT_CASES_NAMES)
                if found:
                    input_cases = found
                    break
    schema = _first_existing(bundle_root, SCHEMA_NAMES)
    if not schema:
        for sub in ("schema", "interop"):
            nested = bundle_root / sub
            if nested.is_dir():
                found = _first_existing(nested, SCHEMA_NAMES)
                if found:
                    schema = found
                    break
    return {
        "bundle_root": str(bundle_root),
        "manifest_path": str(manifest) if manifest else None,
        "input_cases_path": str(input_cases) if input_cases else None,
        "prediction_schema_path": str(schema) if schema else None,
    }


def _validate_bundle_checksum(bundle_root: Path) -> dict[str, Any]:
    try:
        from external_baselines.interop.bundle import load_runner_bundle, validate_bundle_checksum

        bundle = load_runner_bundle(bundle_root)
        report = validate_bundle_checksum(bundle)
        return {
            "ok": bool(report.get("ok")),
            "runner_bundle_checksum": bundle.get("consumer_computed_bundle_hash")
            or bundle.get("producer_declared_checksum"),
            "schema_checksum": sha256_file(bundle["prediction_schema_path"])
            if bundle.get("prediction_schema_path")
            else None,
            "report": report,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}


def _readiness_marker_present(main_repo: Path) -> bool:
    marker = main_repo / "artifacts" / "status" / "first_model_v1_ready.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return bool(payload.get("ready") or payload.get("first_model_v1_ready"))
        except json.JSONDecodeError:
            return True
    runner_manifest = main_repo / "artifacts" / "firebench_interop_v1" / "runner_v1" / "manifest.json"
    return runner_manifest.is_file()


def load_experiment_resources(path: str | Path) -> dict[str, Any]:
    return read_yaml(path)


def assess_main_project_readiness(
    resources_path: str | Path,
    *,
    require_v1_marker: bool = False,
) -> dict[str, Any]:
    """Return readiness report without secrets or API key material.

    Manual ``status.main_project_v1_ready`` is an approval signal only.
    It cannot override repository, branch, bundle, schema, or checksum failures.
    """
    resources = load_experiment_resources(resources_path)
    repository_reasons: list[str] = []
    bundle_reasons: list[str] = []
    approval_reasons: list[str] = []

    main_cfg = dict(resources.get("main_project") or {})
    status = dict(resources.get("status") or {})
    execution = dict(resources.get("execution") or {})
    repo_rel = str(main_cfg.get("repository_path") or "../fire-agent-demo")
    main_repo = _resolve_path(ROOT, repo_rel)
    expected_branch = str(main_cfg.get("expected_branch") or "evaluation/benchmark-v1")

    if not main_repo or not main_repo.is_dir():
        repository_reasons.append("main_project_repository_missing")
        return _build_report(
            resources=resources,
            repository_reasons=repository_reasons,
            bundle_reasons=["runner_bundle_path_not_configured"],
            approval_reasons=["approval_missing"],
            main_repo=None,
            branch=None,
            commit=None,
            bundle_root=None,
            v1_marker=False,
            manual_approval=bool(status.get("main_project_v1_ready")),
            bundle_report={},
        )

    branch = _git(["branch", "--show-current"], main_repo)
    if branch != expected_branch:
        repository_reasons.append("main_project_branch_mismatch")

    commit = _git(["rev-parse", "HEAD"], main_repo) or None
    v1_marker = _readiness_marker_present(main_repo)
    if require_v1_marker and not v1_marker:
        approval_reasons.append("main_project_v1_marker_missing")

    configured_bundle = main_cfg.get("runner_bundle_path")
    bundle_root: Path | None = None
    bundle_report: dict[str, Any] = {}

    if configured_bundle in (None, "", "null"):
        bundle_reasons.append("runner_bundle_path_not_configured")
        discovered = _discover_runner_bundle(main_repo, resources)
        if discovered and discovered.is_dir():
            hint = _bundle_artifacts(discovered)
            bundle_report = {"discovered_candidate_bundle": hint.get("bundle_root")}
    else:
        bundle_root = _resolve_path(ROOT, str(configured_bundle))
        if not bundle_root or not bundle_root.is_dir():
            bundle_reasons.append("runner_bundle_path_missing")
            bundle_root = None
        else:
            artifacts = _bundle_artifacts(bundle_root)
            if not artifacts.get("manifest_path"):
                bundle_reasons.append("runner_bundle_manifest_missing")
            if not artifacts.get("input_cases_path"):
                bundle_reasons.append("runner_bundle_input_cases_missing")
            if not artifacts.get("prediction_schema_path"):
                bundle_reasons.append("runner_bundle_prediction_schema_missing")
            checksum = _validate_bundle_checksum(bundle_root)
            bundle_report = {**artifacts, **checksum}
            if not checksum.get("ok"):
                bundle_reasons.append("runner_bundle_checksum_failed")

    manual_approval = bool(status.get("main_project_v1_ready"))
    marker_approval = bool(v1_marker)
    approval_present = manual_approval or marker_approval
    if not approval_present:
        approval_reasons.append("approval_missing")

    return _build_report(
        resources=resources,
        repository_reasons=repository_reasons,
        bundle_reasons=bundle_reasons,
        approval_reasons=approval_reasons,
        main_repo=main_repo,
        branch=branch or None,
        commit=commit,
        bundle_root=bundle_root,
        v1_marker=v1_marker,
        manual_approval=manual_approval,
        bundle_report=bundle_report,
        expected_branch=expected_branch,
        execution=execution,
        status=status,
    )


def _build_report(
    *,
    resources: dict[str, Any],
    repository_reasons: list[str],
    bundle_reasons: list[str],
    approval_reasons: list[str],
    main_repo: Path | None,
    branch: str | None,
    commit: str | None,
    bundle_root: Path | None,
    v1_marker: bool,
    manual_approval: bool,
    bundle_report: dict[str, Any],
    expected_branch: str | None = None,
    execution: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = status if status is not None else dict(resources.get("status") or {})
    execution = execution if execution is not None else dict(resources.get("execution") or {})
    expected_branch = expected_branch or str(
        (resources.get("main_project") or {}).get("expected_branch") or "evaluation/benchmark-v1"
    )

    repository_valid = (
        main_repo is not None
        and main_repo.is_dir()
        and branch == expected_branch
        and not repository_reasons
    )
    bundle_valid = bundle_root is not None and not bundle_reasons
    approval_present = manual_approval or v1_marker

    main_project_v1_ready = repository_valid and bundle_valid and approval_present

    reasons = sorted(set(repository_reasons + bundle_reasons + approval_reasons))
    if main_project_v1_ready:
        reasons = []

    safe_dry_run = (
        main_project_v1_ready
        and bool(execution.get("allow_real_model_calls"))
        and bool(execution.get("allow_cross_repo_test"))
    )
    safe_formal = (
        safe_dry_run
        and bool(execution.get("allow_formal_evaluation"))
        and bool(status.get("configs_frozen"))
        and bool(status.get("real_dry_run_completed"))
    )

    out: dict[str, Any] = {
        "main_project_v1_ready": main_project_v1_ready,
        "reasons": reasons,
        "repository_reasons": sorted(set(repository_reasons)),
        "bundle_reasons": sorted(set(bundle_reasons)),
        "approval_reasons": sorted(set(approval_reasons)),
        "manual_approval": manual_approval,
        "marker_approval": v1_marker,
        "repository_valid": repository_valid,
        "bundle_valid": bundle_valid,
        "safe_to_run_real_dry_run": safe_dry_run,
        "safe_to_run_formal_experiment": safe_formal,
        "main_project_repository_available": bool(main_repo and main_repo.is_dir()),
        "main_project_branch": branch,
        "main_project_commit": commit,
        "main_project_v1_marker_present": v1_marker,
        "runner_bundle_path": str(bundle_root) if bundle_root else None,
    }
    if bundle_report:
        out["runner_bundle_checksum"] = bundle_report.get("runner_bundle_checksum")
        out["schema_checksum"] = bundle_report.get("schema_checksum")
        out["bundle_artifacts"] = {
            k: bundle_report.get(k)
            for k in ("manifest_path", "input_cases_path", "prediction_schema_path")
        }
        if bundle_report.get("discovered_candidate_bundle"):
            out["discovered_candidate_bundle"] = bundle_report["discovered_candidate_bundle"]
    return out
