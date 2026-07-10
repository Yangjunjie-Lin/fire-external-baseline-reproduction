from __future__ import annotations

"""Runner Bundle loader for firebench-interop-v1.

Baselines may only read the Runner Bundle (input scenarios, allowed corpus/KG
snapshots, experiment config, manifests/checksums). They must not read gold,
expected labels, annotation notes, or target outputs from an Evaluator Bundle.
"""

from pathlib import Path
from typing import Any

from external_baselines.common.checksums import directory_manifest, sha256_file, sha256_json
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


def _strip_forbidden(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_forbidden(v)
            for k, v in obj.items()
            if str(k).lower() not in FORBIDDEN_BUNDLE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_forbidden(x) for x in obj]
    return obj


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
    root = Path(bundle_path)
    if root.is_file():
        manifest = read_json(root)
        root = Path(manifest.get("bundle_root") or root.parent)
    else:
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path, default={})

    experiment_config: dict[str, Any] = {}
    for name in ("experiment_config.yaml", "experiment_config.yml", "experiment_config.json"):
        p = root / name
        if p.exists():
            experiment_config = read_yaml(p) if p.suffix in {".yaml", ".yml"} else read_json(p)
            break

    policy_path = root / "resource_access_policy.json"
    policy = read_json(policy_path, default={}) if policy_path.exists() else {}

    scenarios_path = None
    for candidate in [
        root / "scenarios" / "scenarios.json",
        root / "scenarios.json",
        root / "scenarios" / "scenario_matrix_v2.json",
        Path(str(experiment_config.get("paths", {}).get("scenario_file") or "")),
    ]:
        if candidate and Path(candidate).exists():
            scenarios_path = Path(candidate)
            break

    corpus_dir = None
    for candidate in [
        root / "corpus",
        Path(str(experiment_config.get("paths", {}).get("corpus_dir") or "")),
    ]:
        if candidate and Path(candidate).exists():
            corpus_dir = Path(candidate)
            break

    schema_path = root / "prediction_schema.json"
    if not schema_path.exists():
        schema_path = Path("schemas/firebench_interop_v1_prediction.schema.json")

    # Never expose gold/evaluator material even if mistakenly placed in runner bundle.
    safe_manifest = _strip_forbidden(manifest)
    safe_config = _strip_forbidden(experiment_config)
    safe_policy = _strip_forbidden(policy)

    checksum = (
        safe_manifest.get("bundle_checksum")
        or safe_manifest.get("checksum")
        or sha256_json({"manifest": safe_manifest, "config": safe_config})
    )

    return {
        "bundle_root": str(root),
        "manifest": safe_manifest,
        "experiment_config": safe_config,
        "resource_access_policy": safe_policy,
        "scenarios_path": str(scenarios_path) if scenarios_path else None,
        "corpus_dir": str(corpus_dir) if corpus_dir else None,
        "prediction_schema_path": str(schema_path),
        "bundle_checksum": checksum,
        "corpus_manifest": directory_manifest(corpus_dir) if corpus_dir else None,
        "forbidden_keys_stripped": True,
    }


def validate_bundle_checksum(bundle: dict[str, Any], *, expected: str | None = None) -> dict[str, Any]:
    """Validate declared checksum against recomputed or provided expected value."""
    declared = str(bundle.get("bundle_checksum") or "")
    expected = expected or declared
    ok = bool(expected) and declared == expected
    return {
        "ok": ok,
        "declared": declared,
        "expected": expected,
        "bundle_root": bundle.get("bundle_root"),
        "scenarios_path": bundle.get("scenarios_path"),
        "corpus_dir": bundle.get("corpus_dir"),
        "corpus_aggregate_sha256": (bundle.get("corpus_manifest") or {}).get("aggregate_sha256"),
    }


def assert_no_evaluator_bundle_access(path: str | Path) -> None:
    """Hard fail if a path looks like an evaluator/gold bundle."""
    p = Path(path)
    name = p.name.lower()
    markers = ("evaluator_bundle", "gold", "private_test", "target_outputs")
    if any(m in name for m in markers):
        raise PermissionError(
            f"Baselines must not read evaluator/gold bundles: {path}. "
            "Use the Runner Bundle only."
        )
    if p.is_dir():
        for marker in ("gold.json", "expected.json", "target_outputs.jsonl", "annotations.json"):
            if (p / marker).exists():
                raise PermissionError(
                    f"Runner path contains evaluator material ({marker}). Refusing to load: {path}"
                )
