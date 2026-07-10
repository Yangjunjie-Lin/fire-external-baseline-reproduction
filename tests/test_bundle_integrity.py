from __future__ import annotations

import json

import pytest

from external_baselines.interop.bundle import (
    BundleIntegrityError,
    assert_path_inside_bundle,
    load_runner_bundle,
    recompute_bundle_checksum,
    validate_bundle_checksum,
)


def test_checksum_detects_tampering(tmp_path):
    (tmp_path / "scenarios.json").write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "x"}]}),
        encoding="utf-8",
    )
    declared = recompute_bundle_checksum(tmp_path)
    bundle = {"bundle_root": str(tmp_path), "bundle_checksum": declared}
    assert validate_bundle_checksum(bundle)["ok"] is True

    (tmp_path / "scenarios.json").write_text(
        json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "tampered"}]}),
        encoding="utf-8",
    )
    assert validate_bundle_checksum(bundle)["ok"] is False


def test_empty_path_is_rejected(tmp_path):
    with pytest.raises(BundleIntegrityError):
        load_runner_bundle("")
    with pytest.raises(BundleIntegrityError):
        assert_path_inside_bundle("", tmp_path)


def test_nested_gold_key_hard_fails(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"metadata": {"gold": {"answer": "forbidden"}}}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        load_runner_bundle(tmp_path)
