from __future__ import annotations

"""Runner Bundle loader for firebench-interop-v1.

Baselines may only read the Runner Bundle (input scenarios, allowed corpus/KG
snapshots, experiment config, manifests/checksums). They must not read gold,
expected labels, annotation notes, or target outputs from an Evaluator Bundle.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import directory_manifest, sha256_file
from external_baselines.common.io import read_json, read_jsonl, read_yaml

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
SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")

SCHEMA_SOURCE_BUNDLE_MANIFEST = "bundle_manifest"
SCHEMA_SOURCE_BUNDLE_IMPLICIT_FILE = "bundle_implicit_file"
SCHEMA_SOURCE_LOCAL_DEVELOPMENT_SNAPSHOT = "local_development_snapshot"


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


def _normalize_manifest_rel(root: Path, relative: str | Path) -> str:
    resolved = assert_path_inside_bundle(relative, root, must_exist=True)
    return resolved.relative_to(root).as_posix()


def _exact_nonempty_manifest_string(value: Any, *, error_code: str) -> str:
    if type(value) is not str or not value.strip():
        raise BundleIntegrityError(error_code)
    return value.strip()


def _declared_checksum_for_rel(
    declared: dict[str, Any],
    rel: str,
    *,
    allow_legacy_basename_checksum: bool,
) -> tuple[Any, str | None]:
    if rel in declared:
        return declared.get(rel), rel
    if allow_legacy_basename_checksum:
        basename = Path(rel).name
        if basename in declared:
            return declared.get(basename), basename
    return None, None


def _checksum_format_ok(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_HEX_RE.fullmatch(value))


def _verify_file_checksums(
    root: Path,
    manifest: dict[str, Any],
    *,
    required_logical_names: set[str] | None = None,
    allow_legacy_basename_checksum: bool = True,
) -> dict[str, Any]:
    declared = manifest.get("checksums") if isinstance(manifest.get("checksums"), dict) else {}
    files_map = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    required = set(required_logical_names or set())
    results: dict[str, Any] = {
        "ok": True,
        "checked": {},
        "mismatches": [],
        "missing_required": [],
        "missing_checksums": [],
        "invalid_checksums": [],
    }
    for logical_name in sorted(required):
        if logical_name not in files_map:
            results["ok"] = False
            results["missing_required"].append(logical_name)
    for logical_name, rel in files_map.items():
        if not rel:
            continue
        path = _resolve_manifest_file(root, str(rel), required=True)
        assert path is not None
        normalized_rel = path.relative_to(root).as_posix()
        actual = sha256_file(path)
        expected, checksum_key = _declared_checksum_for_rel(
            declared,
            normalized_rel,
            allow_legacy_basename_checksum=allow_legacy_basename_checksum,
        )
        required_checksum = logical_name in required
        entry = {
            "logical_name": logical_name,
            "path": normalized_rel,
            "checksum_key": checksum_key,
            "actual_sha256": actual,
            "declared_sha256": expected,
            "checksum_declared": expected is not None,
            "declared_sha256_valid": _checksum_format_ok(expected) if expected is not None else False,
            "match": expected is None or expected == actual,
        }
        results["checked"][logical_name] = entry
        if required_checksum and expected is None:
            results["ok"] = False
            results["missing_checksums"].append(entry)
        elif expected is not None and not _checksum_format_ok(expected):
            results["ok"] = False
            results["invalid_checksums"].append(entry)
        elif expected is not None and expected != actual:
            results["ok"] = False
            results["mismatches"].append(entry)
    return results


def _resolve_prediction_schema(
    root: Path,
    manifest: dict[str, Any],
    *,
    formal: bool,
) -> dict[str, Any]:
    files_raw = manifest.get("files")
    files_map = files_raw if isinstance(files_raw, dict) else {}
    source = ""
    schema_rel: str | None = None
    schema_path: Path | None = None

    if formal:
        if not isinstance(files_raw, dict):
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_missing")
        schema_rel = _exact_nonempty_manifest_string(
            files_raw.get("prediction_schema"),
            error_code="formal_runner_bundle_prediction_schema_missing",
        )
        try:
            schema_path = _resolve_manifest_file(root, schema_rel, required=True)
        except BundleIntegrityError as exc:
            message = str(exc)
            if "escapes Runner Bundle root" in message:
                raise BundleIntegrityError("formal_runner_bundle_prediction_schema_outside_bundle") from exc
            if "path is not a file" in message:
                raise BundleIntegrityError("formal_runner_bundle_prediction_schema_not_file") from exc
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_path_invalid") from exc
        if schema_path is None:
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_missing")
        if not schema_path.is_file():
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_not_file")
        source = SCHEMA_SOURCE_BUNDLE_MANIFEST
    else:
        schema_value = files_map.get("prediction_schema")
        if isinstance(schema_value, str) and schema_value.strip():
            try:
                schema_path = _resolve_manifest_file(root, schema_value, required=True)
                schema_rel = schema_path.relative_to(root).as_posix()
                source = SCHEMA_SOURCE_BUNDLE_MANIFEST
            except BundleIntegrityError:
                schema_path = None
        if schema_path is None:
            implicit = root / "prediction_schema.json"
            if implicit.exists():
                schema_path = assert_path_inside_bundle(implicit, root)
                schema_rel = schema_path.relative_to(root).as_posix()
                source = SCHEMA_SOURCE_BUNDLE_IMPLICIT_FILE
            else:
                schema_path = Path("schemas/firebench_interop_v1_prediction.schema.json")
                schema_rel = None
                source = SCHEMA_SOURCE_LOCAL_DEVELOPMENT_SNAPSHOT

    inside_bundle = False
    normalized_rel = None
    if schema_path is not None:
        try:
            normalized_rel = schema_path.resolve(strict=True).relative_to(root).as_posix()
            inside_bundle = True
        except (FileNotFoundError, OSError, ValueError):
            inside_bundle = False

    declared = manifest.get("checksums") if isinstance(manifest.get("checksums"), dict) else {}
    declared_sha = None
    checksum_key = None
    if normalized_rel is not None:
        declared_sha, checksum_key = _declared_checksum_for_rel(
            declared,
            normalized_rel,
            allow_legacy_basename_checksum=not formal,
        )
    actual_sha = sha256_file(schema_path) if schema_path and schema_path.exists() and schema_path.is_file() else None
    checksum_declared = declared_sha is not None
    checksum_valid = _checksum_format_ok(declared_sha) if declared_sha is not None else False
    checksum_match = bool(actual_sha and declared_sha and actual_sha == declared_sha)

    if formal:
        if not inside_bundle:
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_outside_bundle")
        if actual_sha is None:
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_not_file")
        if not checksum_declared:
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_checksum_missing")
        if not checksum_valid:
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_checksum_invalid")
        if not checksum_match:
            raise BundleIntegrityError("formal_runner_bundle_prediction_schema_checksum_mismatch")

    formal_eligible = (
        source == SCHEMA_SOURCE_BUNDLE_MANIFEST
        and inside_bundle
        and checksum_declared
        and checksum_valid
        and checksum_match
    )
    return {
        "prediction_schema_path": str(schema_path) if schema_path is not None else None,
        "prediction_schema_relpath": normalized_rel or schema_rel,
        "prediction_schema_sha256": actual_sha,
        "prediction_schema_declared_sha256": declared_sha,
        "prediction_schema_checksum_key": checksum_key,
        "prediction_schema_source": source,
        "prediction_schema_inside_bundle": inside_bundle,
        "prediction_schema_checksum_declared": checksum_declared,
        "prediction_schema_checksum_valid": checksum_valid,
        "prediction_schema_checksum_match": checksum_match,
        "prediction_schema_authoritative": formal_eligible,
        "prediction_schema_formal_eligible": formal_eligible,
    }


def load_runner_bundle(bundle_path: str | Path, *, formal: bool = False) -> dict[str, Any]:
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
    schema_provenance = _resolve_prediction_schema(root, manifest, formal=formal)
    file_checksum_report = (
        _verify_file_checksums(
            root,
            manifest,
            required_logical_names={"prediction_schema"} if formal else None,
            allow_legacy_basename_checksum=not formal,
        )
        if files_map
        else {"ok": True, "checked": {}, "mismatches": [], "missing_required": [], "missing_checksums": [], "invalid_checksums": []}
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

    producer_checksum = manifest.get("bundle_checksum") or manifest.get("checksum")
    consumer_hash = recompute_bundle_checksum(root)

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
        **schema_provenance,
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


@dataclass
class RunnerBundleCoverage:
    manifest_case_count: int | None
    input_file_case_count: int
    loaded_case_count: int
    input_case_ids: list[str]
    loaded_case_ids: list[str]
    runner_bundle_case_count_source: str
    scenarios_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_case_count": self.manifest_case_count,
            "input_file_case_count": self.input_file_case_count,
            "loaded_case_count": self.loaded_case_count,
            "input_case_ids": list(self.input_case_ids),
            "loaded_case_ids": list(self.loaded_case_ids),
            "runner_bundle_case_count_source": self.runner_bundle_case_count_source,
            "scenarios_path": self.scenarios_path,
        }


def _manifest_declared_case_count(manifest: dict[str, Any]) -> int | None:
    for key in ("case_count", "input_case_count", "scenario_count"):
        value = manifest.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    stats = manifest.get("stats")
    if isinstance(stats, dict):
        for key in ("case_count", "input_case_count", "scenario_count"):
            value = stats.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return None


def inspect_runner_bundle_case_coverage(
    bundle_path: str | Path,
    *,
    limit: int | None = None,
    formal: bool = False,
) -> RunnerBundleCoverage:
    """Inspect full Runner Bundle case coverage without truncating input_cases.jsonl."""
    from external_baselines.common.io import load_scenarios

    bundle = load_runner_bundle(bundle_path, formal=formal)
    scenarios_path = Path(str(bundle.get("scenarios_path") or ""))
    if not scenarios_path.is_file():
        raise BundleIntegrityError(f"Runner Bundle input cases missing: {scenarios_path}")

    rows = read_jsonl(scenarios_path)
    input_case_ids: list[str] = []
    for row in rows:
        cid = str(row.get("case_id") or row.get("scenario_id") or "").strip()
        if cid:
            input_case_ids.append(cid)

    manifest = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    manifest_case_count = _manifest_declared_case_count(manifest)
    count_source = "manifest" if manifest_case_count is not None else "input_cases_jsonl"

    loaded = load_scenarios(scenarios_path, limit=limit)
    loaded_case_ids = [str(s.get("case_id") or s.get("scenario_id") or "").strip() for s in loaded]
    loaded_case_ids = [cid for cid in loaded_case_ids if cid]

    return RunnerBundleCoverage(
        manifest_case_count=manifest_case_count,
        input_file_case_count=len(input_case_ids),
        loaded_case_count=len(loaded_case_ids),
        input_case_ids=input_case_ids,
        loaded_case_ids=loaded_case_ids,
        runner_bundle_case_count_source=count_source,
        scenarios_path=str(scenarios_path),
    )


def validate_formal_runner_bundle_coverage(coverage: RunnerBundleCoverage) -> None:
    """Hard-fail when formal coverage invariants are violated."""
    if coverage.manifest_case_count is not None and coverage.manifest_case_count != coverage.input_file_case_count:
        raise BundleIntegrityError(
            "Formal Runner Bundle case_count mismatch: "
            f"manifest={coverage.manifest_case_count} input_cases={coverage.input_file_case_count}"
        )
    if coverage.loaded_case_count != coverage.input_file_case_count:
        raise BundleIntegrityError(
            "Formal execution requires loading the complete Runner Bundle case set; "
            f"loaded={coverage.loaded_case_count} expected={coverage.input_file_case_count}"
        )
    if set(coverage.loaded_case_ids) != set(coverage.input_case_ids):
        raise BundleIntegrityError(
            "Formal loaded case_id set does not match input_cases.jsonl case_id set."
        )
    if len(coverage.input_case_ids) != len(set(coverage.input_case_ids)):
        raise BundleIntegrityError("Duplicate case_id values in input_cases.jsonl.")


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
