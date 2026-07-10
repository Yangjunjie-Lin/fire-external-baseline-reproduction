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
    "target_output",
    "evaluator_hints",
    "expert_scores",
    "reference_answer",
}

# Exact filename stems / prefixes — avoid rejecting benign metadata like label_coverage_version.
FORBIDDEN_FILENAME_EXACT = {
    "gold",
    "expected",
    "labels",
    "annotations",
    "evaluator_hints",
    "target_outputs",
    "target_output",
    "reference_answer",
}
FORBIDDEN_FILENAME_PREFIXES = (
    "gold_",
    "expected_",
    "labels_",
    "annotations_",
    "evaluator_hints_",
    "target_outputs_",
)
FORBIDDEN_FILENAME_SUFFIXES = (
    "_gold",
    "_expected",
    "_labels",
    "_annotations",
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


def _filename_is_forbidden(relative: str) -> bool:
    name = Path(relative).name.lower()
    stem = Path(name).stem.lower()
    if stem in FORBIDDEN_FILENAME_EXACT or name in FORBIDDEN_FILENAME_EXACT:
        return True
    if any(stem.startswith(prefix) for prefix in FORBIDDEN_FILENAME_PREFIXES):
        return True
    if any(stem.endswith(suffix) for suffix in FORBIDDEN_FILENAME_SUFFIXES):
        return True
    return False


def _assert_allowed_bundle_files(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().lower()
        if _filename_is_forbidden(relative):
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
    """Consumer-side diagnostic aggregate hash of allowed Runner Bundle files."""
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


def _resolve_manifest_file(root: Path, relative: str | None, *, required: bool = False) -> Path | None:
    if relative is None or not str(relative).strip():
        if required:
            raise BundleIntegrityError("manifest.files entry is missing or empty")
        return None
    resolved = assert_path_inside_bundle(relative, root, must_exist=True)
    if not resolved.is_file():
        raise BundleIntegrityError(f"manifest.files path is not a file: {relative}")
    return resolved


def _verify_file_checksums(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    declared = manifest.get("checksums") if isinstance(manifest.get("checksums"), dict) else {}
    files_map = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    results: dict[str, Any] = {"ok": True, "checked": {}, "mismatches": []}
    for logical_name, rel in files_map.items():
        if not rel:
            continue
        path = _resolve_manifest_file(root, str(rel), required=True)
        assert path is not None
        actual = sha256_file(path)
        expected = declared.get(str(rel)) or declared.get(Path(str(rel)).name)
        entry = {
            "logical_name": logical_name,
            "path": str(rel),
            "actual_sha256": actual,
            "declared_sha256": expected,
            "match": expected is None or expected == actual,
        }
        results["checked"][logical_name] = entry
        if expected is not None and expected != actual:
            results["ok"] = False
            results["mismatches"].append(entry)
    return results


def load_runner_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Load a firebench-interop-v1 Runner Bundle.

    Formal layout prefers ``manifest.files.input_cases`` (JSONL). Legacy
    scenarios.json layouts remain supported for local smoke only.
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
    if str(manifest.get("bundle_type") or "").lower() == "evaluator":
        raise PermissionError("Baselines must not load evaluator bundles; Runner Bundle only.")
    _assert_allowed_bundle_files(root)
    _assert_no_forbidden_structured_content(root)
    _assert_no_forbidden_keys(manifest, location="manifest")

    files_map = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    file_checksum_report = (
        _verify_file_checksums(root, manifest) if files_map else {"ok": True, "checked": {}, "mismatches": []}
    )

    experiment_config: dict[str, Any] = {}
    exp_rel = files_map.get("experiment_config")
    if exp_rel:
        exp_path = _resolve_manifest_file(root, str(exp_rel), required=True)
        assert exp_path is not None
        experiment_config = read_yaml(exp_path) if exp_path.suffix in {".yaml", ".yml"} else read_json(exp_path)
        _assert_no_forbidden_keys(experiment_config, location=str(exp_rel))
    else:
        for name in ("experiment_config.yaml", "experiment_config.yml", "experiment_config.json"):
            p = root / name
            if p.exists():
                experiment_config = read_yaml(p) if p.suffix in {".yaml", ".yml"} else read_json(p)
                _assert_no_forbidden_keys(experiment_config, location=name)
                break

    policy: dict[str, Any] = {}
    policy_rel = files_map.get("resource_access_policy")
    if policy_rel:
        policy_path = _resolve_manifest_file(root, str(policy_rel), required=True)
        assert policy_path is not None
        policy = read_json(policy_path)
    else:
        fallback_policy = root / "resource_access_policy.json"
        policy = read_json(fallback_policy, default={}) if fallback_policy.exists() else {}
    _assert_no_forbidden_keys(policy, location="resource_access_policy")

    scenarios_path = None
    input_cases_rel = files_map.get("input_cases")
    if input_cases_rel:
        scenarios_path = _resolve_manifest_file(root, str(input_cases_rel), required=True)
    else:
        for candidate in [
            root / "input_cases.jsonl",
            root / "scenarios" / "scenarios.json",
            root / "scenarios.json",
            root / "scenarios" / "scenario_matrix_v2.json",
            experiment_config.get("paths", {}).get("scenario_file"),
        ]:
            if not candidate:
                continue
            try:
                resolved = assert_path_inside_bundle(candidate, root)
            except BundleIntegrityError:
                continue
            if resolved.exists():
                scenarios_path = resolved
                break

    corpus_dir = None
    corpus_manifest_path = None
    corpus_rel = files_map.get("corpus_manifest")
    if corpus_rel:
        corpus_manifest_path = _resolve_manifest_file(root, str(corpus_rel), required=True)
    for candidate in [
        root / "corpus",
        experiment_config.get("paths", {}).get("corpus_dir"),
    ]:
        if not candidate:
            continue
        try:
            resolved = assert_path_inside_bundle(candidate, root, must_exist=False)
        except BundleIntegrityError:
            continue
        if resolved.exists():
            corpus_dir = resolved
            break

    schema_rel = files_map.get("prediction_schema")
    if schema_rel:
        schema_path = _resolve_manifest_file(root, str(schema_rel), required=True)
    else:
        schema_path = root / "prediction_schema.json"
        if not schema_path.exists():
            schema_path = Path("schemas/firebench_interop_v1_prediction.schema.json")

    producer_checksum = manifest.get("bundle_checksum") or manifest.get("checksum")
    consumer_hash = recompute_bundle_checksum(root)
    schema_sha = sha256_file(schema_path) if schema_path and Path(schema_path).exists() else None

    corpus_manifest_payload: Any = None
    if corpus_dir:
        corpus_manifest_payload = directory_manifest(corpus_dir)
    elif corpus_manifest_path:
        corpus_manifest_payload = read_json(corpus_manifest_path)

    return {
        "bundle_root": str(root),
        "manifest": manifest,
        "experiment_config": experiment_config,
        "resource_access_policy": policy,
        "scenarios_path": str(scenarios_path) if scenarios_path else None,
        "input_cases_path": str(scenarios_path) if scenarios_path else None,
        "corpus_dir": str(corpus_dir) if corpus_dir else None,
        "corpus_manifest_path": str(corpus_manifest_path) if corpus_manifest_path else None,
        "prediction_schema_path": str(schema_path),
        "prediction_schema_sha256": schema_sha,
        "bundle_checksum": producer_checksum,
        "producer_declared_checksum": producer_checksum,
        "consumer_computed_bundle_hash": consumer_hash,
        "recomputed_bundle_checksum": consumer_hash,
        "file_checksum_report": file_checksum_report,
        "corpus_manifest": corpus_manifest_payload,
        "forbidden_keys_stripped": False,
        "formal_manifest_files_used": bool(files_map.get("input_cases")),
    }


def validate_bundle_checksum(bundle: dict[str, Any], *, expected: str | None = None) -> dict[str, Any]:
    """Validate checksums without confusing producer vs consumer hashes."""
    file_report = bundle.get("file_checksum_report") or {}
    declared = str(bundle.get("producer_declared_checksum") or bundle.get("bundle_checksum") or "")
    recomputed = str(
        bundle.get("consumer_computed_bundle_hash")
        or bundle.get("recomputed_bundle_checksum")
        or ""
    )
    expected_value = str(expected or "")
    if file_report.get("checked"):
        ok = bool(file_report.get("ok", False))
        if expected is not None and declared:
            ok = ok and (expected_value == declared)
    elif declared:
        ok = declared == recomputed
        if expected is not None:
            ok = ok and expected_value == recomputed
    else:
        ok = True if expected is None else (expected_value == recomputed)
    corpus_manifest = bundle.get("corpus_manifest")
    return {
        "ok": ok,
        "declared": declared or None,
        "recomputed": recomputed,
        "consumer_computed_bundle_hash": recomputed,
        "producer_declared_checksum": declared or None,
        "expected": expected_value or None,
        "file_checksum_report": file_report,
        "bundle_root": bundle.get("bundle_root"),
        "scenarios_path": bundle.get("scenarios_path"),
        "corpus_dir": bundle.get("corpus_dir"),
        "corpus_aggregate_sha256": (
            corpus_manifest.get("aggregate_sha256") if isinstance(corpus_manifest, dict) else None
        ),
        "note": (
            "When only per-file checksums exist, aggregate consumer hash is diagnostic "
            "and must not be confused with a producer-declared bundle checksum."
        ),
    }


def assert_no_evaluator_bundle_access(path: str | Path) -> None:
    """Hard fail if a path looks like an evaluator/gold bundle."""
    if path is None or not str(path).strip():
        raise BundleIntegrityError("Runner Bundle path must be non-empty")
    p = Path(path)
    name = p.name.lower()
    markers = ("evaluator_bundle", "evaluator_seed", "gold", "private_test", "target_outputs")
    if any(m in name for m in markers):
        raise PermissionError(
            f"Baselines must not read evaluator/gold bundles: {path}. "
            "Use the Runner Bundle only."
        )
    if p.is_dir():
        _assert_allowed_bundle_files(p)
        manifest_path = p / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path, default={})
            if str(manifest.get("bundle_type") or "").lower() == "evaluator":
                raise PermissionError(
                    f"Baselines must not read evaluator bundles: {path}"
                )
