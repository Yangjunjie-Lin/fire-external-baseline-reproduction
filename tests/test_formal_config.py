"""Formal config validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from external_baselines.common.formal_config_validator import (  # noqa: E402
    FormalConfigError,
    validate_experiment_manifest,
    validate_method_config,
)
from external_baselines.common.io import read_yaml


FORMAL_MANIFEST = ROOT / "configs/experiments/controlled_main_table_v1.yaml.example"
SMOKE = ROOT / "configs/smoke/deterministic_heuristic.yaml"
EKELL_FROZEN = ROOT / "configs/frozen/ekell_controlled_shared_llm_v1.yaml"


def test_only_one_formal_main_manifest():
    active = [
        p.name
        for p in (ROOT / "configs/experiments").glob("*.yaml*")
        if "controlled_main_table_v1" in p.name
    ]
    assert active == ["controlled_main_table_v1.yaml.example"]
    deprecated = (ROOT / "configs/experiments/paper_main_table_v1.yaml.example").read_text(encoding="utf-8")
    assert "deprecated: true" in deprecated


def test_formal_manifest_uses_canonical_ids():
    text = FORMAL_MANIFEST.read_text(encoding="utf-8")
    assert "ekell_style_controlled_shared_llm" in text
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        if "method_id:" in line or "config:" in line:
            assert "ekell_style_faithful" not in line


def test_formal_manifest_accepts_placeholders():
    result = validate_experiment_manifest(FORMAL_MANIFEST, allow_placeholders=True)
    assert result["valid"] is True


def test_formal_manifest_rejects_placeholders_without_flag():
    with pytest.raises(FormalConfigError):
        validate_experiment_manifest(FORMAL_MANIFEST, allow_placeholders=False)


def test_formal_manifest_requires_schema_and_checksum_flags():
    raw = read_yaml(FORMAL_MANIFEST)
    for key in ("require_bundle_checksum", "require_external_schema", "fail_on_schema_error"):
        assert raw[key] is True


def test_formal_ekell_requires_explicit_vector_backend():
    cfg = read_yaml(EKELL_FROZEN)
    assert "ekell_vector" in cfg
    assert cfg["ekell_vector"]["reject_smoke"] is True


def test_formal_ekell_rejects_smoke_backend():
    cfg = read_yaml(EKELL_FROZEN)
    cfg["llm"] = {"provider": "siliconflow", "model": "m", "model_version": "v"}
    cfg["paper_final"] = True
    cfg["ekell_vector"]["backend"] = "smoke"
    with pytest.raises(FormalConfigError):
        validate_method_config(cfg, method_id="ekell_style_controlled_shared_llm")


def test_formal_ekell_rejects_missing_embedding_model():
    cfg = read_yaml(EKELL_FROZEN)
    cfg["llm"] = {"provider": "siliconflow", "model": "m", "model_version": "v"}
    del cfg["ekell_vector"]["model_name"]
    with pytest.raises(FormalConfigError):
        validate_method_config(cfg, method_id="ekell_style_controlled_shared_llm", allow_placeholders=False)


def test_formal_ekell_rejects_placeholder_values():
    cfg = read_yaml(EKELL_FROZEN)
    cfg["llm"] = {"provider": "siliconflow", "model": "m", "model_version": "v"}
    with pytest.raises(FormalConfigError):
        validate_method_config(cfg, method_id="ekell_style_controlled_shared_llm", allow_placeholders=False)


def test_smoke_config_may_use_hash_embedding():
    smoke = read_yaml(SMOKE)
    assert smoke["llm"]["provider"] == "heuristic"
    assert smoke["paper_final"] is False


def test_formal_main_manifest_uses_canonical_ekell_id():
    raw = read_yaml(FORMAL_MANIFEST)
    ids = [m["method_id"] for m in raw["methods"] if m.get("enabled")]
    assert "ekell_style_controlled_shared_llm" in ids
    assert "ekell_style_faithful" not in ids


def test_formal_manifest_rejects_smoke_llm():
    raw = read_yaml(FORMAL_MANIFEST)
    raw["shared_model_config"] = "configs/smoke/deterministic_heuristic.yaml"
    path = ROOT / "outputs/diagnostics/tmp_bad_manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(FormalConfigError):
        validate_experiment_manifest(path, allow_placeholders=True)
