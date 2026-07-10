from __future__ import annotations

"""Runner Bundle loader for firebench-interop-v1.

Baselines may only read the Runner Bundle (input scenarios, allowed corpus/KG
snapshots, experiment config, manifests/checksums). They must not read gold,
expected labels, annotation notes, or target outputs from an Evaluator Bundle.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import directory_manifest, sha256_file
from external_baselines.common.io import read_json, read_yaml


FORBIDDEN_BUNDLE_KEYS = {
    "expected",
    "gold",
    "ground_truth",
    "labels",
    "annotations",
    "annotation_notes",
    "target_outputs",
    "evaluator_hints",
    "expert_scores",
}

FORBIDDEN_FILENAME_MARKERS = (
    "gold",
    "expected",
    "labels",
    "annotations",
    "evaluator_hints",
    "target_outputs",
)
CHECKSUM_FILE_SUFFIXES = {
    ".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".tsv",
    ".md", ".xml", ".ttl", ".nt", ".parquet", ".npy", ".npz",
}
DETACHED_METADATA_FILES = {"manifest.json"}


class BundleIntegrityError(RuntimeError):
    """Raised when a Runner Bundle fails an integrity invariant."""


def _assert_no_forbidden_keys(obj: Any, *, location: str) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).strip().lower() in FORBIDDEN_BUNDLE_KEYS:
                raise PermissionError(
                    f"Runner Bundle contains forbidden evaluator key {key!r} at {location}"
                )
            _assert_no_forbidden_keys(value, location=f"{location}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            _assert_no_forbidden_keys(value, location=f"{location}[{index}]")


def assert_path_inside_bundle(
    path: str | Path,
    bundle_root: str | Path,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a non-empty path and prove that it remains inside the bundle."""
    if path is None or not str(path).strip():
        raise BundleIntegrityError("Bundle-relative path must be non-empty")
    root_text = str(bundle_root).strip()
    if not root_text:
        raise BundleIntegrityError("Bundle root must be non-empty")
    root = Path(root_text).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (FileNotFoundError, OSError) as exc:
        raise BundleIntegrityError(f"Bundle path cannot be resolved: {path}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BundleIntegrityError(
            f"Path escapes Runner Bundle root: {path} (root={root})"
        ) from exc
    return resolved


def _assert_allowed_bundle_files(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().lower()
        if any(marker in relative for marker in FORBIDDEN_FILENAME_MARKERS):
            raise PermissionError(
                f"Runner Bundle contains forbidden evaluator material: {relative}"
            )


def _assert_no_forbidden_structured_content(root: Path) -> None:
    """Inspect all structured Runner payloads, not only top-level metadata."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        relative = path.relative_to(root).as_posix()
        try:
            if suffix == ".json":
                value = read_json(path)
                _assert_no_forbidden_keys(value, location=relative)
            elif suffix in {".yaml", ".yml"}:
                value = read_yaml(path)
                _assert_no_forbidden_keys(value, location=relative)
            elif suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if line.strip():
                            _assert_no_forbidden_keys(
                                json.loads(line),
                                location=f"{relative}:{line_number}",
                            )
        except PermissionError:
            raise
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise BundleIntegrityError(
                f"Invalid structured Runner Bundle file: {relative}"
            ) from exc


def recompute_bundle_checksum(root: str | Path) -> str:
    """Hash a canonical, sorted manifest of allowed Runner Bundle files."""
    if root is None or not str(root).strip():
        raise BundleIntegrityError("Bundle root must be non-empty")
    try:
        root_path = Path(root).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise BundleIntegrityError(f"Bundle root cannot be resolved: {root}") from exc
    if not root_path.is_dir():
        raise BundleIntegrityError(f"Bundle root is not a directory: {root_path}")
    _assert_allowed_bundle_files(root_path)
    files: list[dict[str, Any]] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CHECKSUM_FILE_SUFFIXES:
            continue
        resolved = assert_path_inside_bundle(path, root_path)
        relative = resolved.relative_to(root_path).as_posix()
        # The declaration manifest is detached metadata; including its own
        # checksum value would make a stable aggregate impossible.
        if relative.lower() in DETACHED_METADATA_FILES:
            continue
        files.append({
            "path": relative,
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        })
    files.sort(key=lambda item: item["path"])
    canonical = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_runner_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Load a runner bundle directory or manifest JSON.

    Expected layout (flexible):
      bundle/
        manifest.json
        experiment_config.json | experiment_config.yaml
        prediction_schema.json          (optional reference)
        resource_access_policy.json     (optional)
        scenarios/  or scenarios.json
        corpus/     (optional snapshot)
    """
    if bundle_path is None or not str(bundle_path).strip():
        raise BundleIntegrityError("Runner Bundle path must be non-empty")
    supplied = Path(bundle_path).resolve(strict=True)
    root = supplied
    if root.is_file():
        manifest = read_json(root)
        manifest_root = manifest.get("bundle_root")
        root = (
            assert_path_inside_bundle(manifest_root, root.parent)
            if manifest_root
            else root.parent.resolve(strict=True)
        )
    else:
        root = root.resolve(strict=True)
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path, default={})
    _assert_allowed_bundle_files(root)
    _assert_no_forbidden_structured_content(root)
    _assert_no_forbidden_keys(manifest, location="manifest")

    experiment_config: dict[str, Any] = {}
    for name in ("experiment_config.yaml", "experiment_config.yml", "experiment_config.json"):
        p = root / name
        if p.exists():
            experiment_config = read_yaml(p) if p.suffix in {".yaml", ".yml"} else read_json(p)
            _assert_no_forbidden_keys(experiment_config, location=name)
            break

    policy_path = root / "resource_access_policy.json"
    policy = read_json(policy_path, default={}) if policy_path.exists() else {}
    _assert_no_forbidden_keys(policy, location="resource_access_policy")

    scenarios_path = None
    for candidate in [
        root / "scenarios" / "scenarios.json",
        root / "scenarios.json",
        root / "scenarios" / "scenario_matrix_v2.json",
        experiment_config.get("paths", {}).get("scenario_file"),
    ]:
        if candidate:
            try:
                resolved = assert_path_inside_bundle(candidate, root)
            except BundleIntegrityError:
                if candidate in {
                    root / "scenarios" / "scenarios.json",
                    root / "scenarios.json",
                    root / "scenarios" / "scenario_matrix_v2.json",
                }:
                    continue
                raise
            if resolved.exists():
                scenarios_path = resolved
                break

    corpus_dir = None
    for candidate in [
        root / "corpus",
        experiment_config.get("paths", {}).get("corpus_dir"),
    ]:
        if candidate:
            try:
                resolved = assert_path_inside_bundle(candidate, root)
            except BundleIntegrityError:
                if candidate == root / "corpus":
                    continue
                raise
            if resolved.exists():
                corpus_dir = resolved
                break

    schema_path = root / "prediction_schema.json"
    if not schema_path.exists():
        schema_path = Path("schemas/firebench_interop_v1_prediction.schema.json")

    checksum = manifest.get("bundle_checksum") or manifest.get("checksum")

    return {
        "bundle_root": str(root),
        "manifest": manifest,
        "experiment_config": experiment_config,
        "resource_access_policy": policy,
        "scenarios_path": str(scenarios_path) if scenarios_path else None,
        "corpus_dir": str(corpus_dir) if corpus_dir else None,
        "prediction_schema_path": str(schema_path),
        "bundle_checksum": checksum,
        "recomputed_bundle_checksum": recompute_bundle_checksum(root),
        "corpus_manifest": directory_manifest(corpus_dir) if corpus_dir else None,
        "forbidden_keys_stripped": False,
    }


def validate_bundle_checksum(bundle: dict[str, Any], *, expected: str | None = None) -> dict[str, Any]:
    """Validate declared checksum against recomputed or provided expected value."""
    declared = str(bundle.get("bundle_checksum") or "")
    root = bundle.get("bundle_root")
    recomputed = recompute_bundle_checksum(root)
    expected_value = str(expected or "")
    ok = bool(declared) and declared == recomputed
    if expected is not None:
        ok = ok and expected_value == recomputed
    return {
        "ok": ok,
        "declared": declared,
        "recomputed": recomputed,
        "expected": expected_value or None,
        "bundle_root": bundle.get("bundle_root"),
        "scenarios_path": bundle.get("scenarios_path"),
        "corpus_dir": bundle.get("corpus_dir"),
        "corpus_aggregate_sha256": (bundle.get("corpus_manifest") or {}).get("aggregate_sha256"),
    }


def assert_no_evaluator_bundle_access(path: str | Path) -> None:
    """Hard fail if a path looks like an evaluator/gold bundle."""
    if path is None or not str(path).strip():
        raise BundleIntegrityError("Runner Bundle path must be non-empty")
    p = Path(path)
    name = p.name.lower()
    markers = ("evaluator_bundle", "gold", "private_test", "target_outputs")
    if any(m in name for m in markers):
        raise PermissionError(
            f"Baselines must not read evaluator/gold bundles: {path}. "
            "Use the Runner Bundle only."
        )
    if p.is_dir():
        _assert_allowed_bundle_files(p)
