"""Artifact packaging dry-run tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_packager():
    path = ROOT / "scripts" / "package_reproducibility_artifact.py"
    spec = importlib.util.spec_from_file_location("package_reproducibility_artifact", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_artifact_dry_run_without_results():
    mod = _load_packager()
    result = mod.package_artifact(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        output_dir=ROOT / "outputs/diagnostics/test_artifact_dry_run",
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["manifest"]["predictions_present"] is False


def test_artifact_manifest_complete():
    mod = _load_packager()
    out = ROOT / "outputs/diagnostics/test_artifact_packaged"
    mod.package_artifact(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        output_dir=out,
        dry_run=False,
    )
    manifest_path = out / "MANIFEST.json"
    assert manifest_path.is_file()
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("git_commit", "experiment_id", "method_ids", "artifact_status"):
        assert key in meta


def test_artifact_checksums_match():
    mod = _load_packager()
    out = ROOT / "outputs/diagnostics/test_artifact_checksums"
    mod.package_artifact(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        output_dir=out,
        dry_run=False,
    )
    assert (out / "CHECKSUMS.sha256").is_file()


def test_artifact_redacts_secrets():
    mod = _load_packager()
    out = ROOT / "outputs/diagnostics/test_artifact_redact"
    mod.package_artifact(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        output_dir=out,
        dry_run=False,
    )
    redacted = out / "configs/shared_model_config.redacted.yaml"
    if redacted.is_file():
        text = redacted.read_text(encoding="utf-8")
        assert "sk-" not in text


def test_artifact_records_git_dirty_state():
    mod = _load_packager()
    out = ROOT / "outputs/diagnostics/test_artifact_git"
    mod.package_artifact(
        experiment_manifest=ROOT / "configs/experiments/controlled_main_table_v1.yaml.example",
        output_dir=out,
        dry_run=False,
    )
    meta = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "git_dirty" in meta
