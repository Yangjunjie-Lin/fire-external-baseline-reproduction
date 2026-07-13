"""Formal config validation tests (controlled + paper-fidelity + placeholders + .example)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

from external_baselines.common.formal_config_validator import (  # noqa: E402
    FormalConfigError,
    _is_placeholder,
    _validate_positive_dimension,
    validate_experiment_manifest,
    validate_method_config,
)
from external_baselines.common.io import read_yaml

FORMAL_MANIFEST = ROOT / "configs/experiments/controlled_main_table_v1.yaml.example"
PAPER_FIDELITY_MANIFEST = ROOT / "configs/experiments/ekell_paper_fidelity_v1.yaml.example"
SMOKE = ROOT / "configs/smoke/deterministic_heuristic.yaml"
EKELL_FROZEN = ROOT / "configs/frozen/ekell_controlled_shared_llm_v1.yaml"


# --- existing controlled tests ---


def test_only_one_formal_main_manifest():
    active = [
        p.name
        for p in (ROOT / "configs/experiments").glob("*.yaml*")
        if "controlled_main_table_v1" in p.name and "deprecated" not in p.read_text(encoding="utf-8")[:80]
    ]
    assert "controlled_main_table_v1.yaml.example" in active


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
    assert result["mode"] == "template"


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
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(FormalConfigError):
        validate_experiment_manifest(path, allow_placeholders=True)


# --- paper-fidelity formal validation ---


def test_paper_fidelity_manifest_runs_full_validation():
    result = validate_experiment_manifest(PAPER_FIDELITY_MANIFEST, allow_placeholders=True)
    assert result["valid"] is True


def test_paper_fidelity_rejects_smoke_llm():
    cfg = read_yaml(ROOT / "configs/ekell_paper_fidelity_chatglm6b.yaml.example")
    cfg["llm"] = {"provider": "heuristic", "model": "x", "model_version": "y"}
    with pytest.raises(FormalConfigError):
        validate_method_config(cfg, method_id="ekell_style_paper_fidelity", allow_placeholders=False)


def test_paper_fidelity_rejects_smoke_vector():
    cfg = read_yaml(ROOT / "configs/ekell_paper_fidelity_chatglm6b.yaml.example")
    cfg["llm"] = {"provider": "chatglm_local", "model": "THUDM/chatglm3-6b", "model_version": "rev1"}
    cfg["ekell_vector"]["backend"] = "smoke_hash"
    with pytest.raises(FormalConfigError):
        validate_method_config(cfg, method_id="ekell_style_paper_fidelity", allow_placeholders=False)


def test_paper_fidelity_requires_explicit_vector():
    cfg = read_yaml(ROOT / "configs/ekell_paper_fidelity_chatglm6b.yaml.example")
    cfg["llm"] = {"provider": "chatglm_local", "model": "THUDM/chatglm3-6b", "model_version": "rev1"}
    del cfg["ekell_vector"]
    with pytest.raises(FormalConfigError):
        validate_method_config(cfg, method_id="ekell_style_paper_fidelity", allow_placeholders=False)


def test_paper_fidelity_requires_model_version():
    cfg = read_yaml(ROOT / "configs/ekell_paper_fidelity_chatglm6b.yaml.example")
    cfg["llm"] = {"provider": "chatglm_local", "model": "THUDM/chatglm3-6b", "model_version": "REPLACE"}
    with pytest.raises(FormalConfigError):
        validate_method_config(cfg, method_id="ekell_style_paper_fidelity", allow_placeholders=False)


def test_paper_fidelity_requires_embedding_model():
    cfg = read_yaml(ROOT / "configs/ekell_paper_fidelity_chatglm6b.yaml.example")
    cfg["llm"] = {"provider": "chatglm_local", "model": "THUDM/chatglm3-6b", "model_version": "rev1"}
    cfg["ekell_vector"]["model_name"] = "REPLACE_WITH_TEXT2VEC_MODEL"
    with pytest.raises(FormalConfigError):
        validate_method_config(cfg, method_id="ekell_style_paper_fidelity", allow_placeholders=False)


def test_paper_fidelity_not_in_main_table():
    raw = read_yaml(PAPER_FIDELITY_MANIFEST)
    assert "ekell_style_paper_fidelity" not in (raw.get("main_table_methods") or [])
    assert "ekell_style_paper_fidelity" not in (raw.get("supplemental_methods") or [])


def test_paper_fidelity_model_run_false_allowed():
    cfg = read_yaml(ROOT / "configs/ekell_paper_fidelity_chatglm6b.yaml.example")
    cfg["paper_fidelity_model_run"] = False
    cfg["llm"] = {
        "provider": "chatglm_local",
        "model": "THUDM/chatglm3-6b",
        "model_version": "REPLACE_WITH_REVISION",
    }
    validate_method_config(cfg, method_id="ekell_style_paper_fidelity", allow_placeholders=True)


# --- placeholder detection ---


@pytest.mark.parametrize(
    "value,expected",
    [
        ("REPLACE", True),
        ("REPLACE_WITH_REVISION", True),
        ("REPLACE_WITH_TEXT2VEC_MODEL", True),
        ("REQUIRED_BEFORE_FORMAL_RUN", True),
        ("REQUIRED_FOO", True),
        ("path/to/runner_bundle", True),
        ("outputs/indexes/ekell/<model-hash>/<corpus-hash>/", True),
        ("768", False),
        ("text2vec-base", False),
        (768, False),
    ],
)
def test_placeholder_detection(value, expected):
    assert _is_placeholder(value) is expected


def test_rejects_zero_dimension():
    with pytest.raises(FormalConfigError):
        _validate_positive_dimension(0, allow_placeholders=False)


def test_rejects_negative_dimension():
    with pytest.raises(FormalConfigError):
        _validate_positive_dimension(-1, allow_placeholders=False)


def test_accepts_real_dimension():
    assert _validate_positive_dimension(768, allow_placeholders=False) == 768


def test_rejects_string_dimension():
    with pytest.raises(FormalConfigError, match="exact YAML integer"):
        _validate_positive_dimension("768", allow_placeholders=False)


def test_formal_config_rejects_dimension_string():
    with pytest.raises(FormalConfigError, match="dense_rag.dimension"):
        _validate_positive_dimension("8", allow_placeholders=False, field="dense_rag.dimension")


def test_formal_config_rejects_dimension_float():
    with pytest.raises(FormalConfigError, match="dense_rag.dimension"):
        _validate_positive_dimension(8.0, allow_placeholders=False, field="dense_rag.dimension")


def test_formal_config_rejects_dimension_bool():
    with pytest.raises(FormalConfigError, match="dense_rag.dimension"):
        _validate_positive_dimension(True, allow_placeholders=False, field="dense_rag.dimension")


def test_formal_config_rejects_reject_smoke_string_false():
    from external_baselines.common.formal_config_validator import _validate_exact_bool

    with pytest.raises(FormalConfigError, match="dense_rag.reject_smoke"):
        _validate_exact_bool("false", field="dense_rag.reject_smoke", required=True)


def test_formal_config_rejects_reject_smoke_integer_one():
    from external_baselines.common.formal_config_validator import _validate_exact_bool

    with pytest.raises(FormalConfigError, match="dense_rag.reject_smoke"):
        _validate_exact_bool(1, field="dense_rag.reject_smoke", required=True)


def test_formal_config_rejects_paper_final_string_true():
    from external_baselines.common.formal_config_validator import _validate_exact_bool

    with pytest.raises(FormalConfigError, match="paper_final"):
        _validate_exact_bool("true", field="paper_final", required=True)


def test_formal_config_accepts_exact_bool_and_int():
    from external_baselines.common.formal_config_validator import _validate_exact_bool

    assert _validate_exact_bool(True, field="dense_rag.reject_smoke", required=True) is True
    assert _validate_exact_bool(False, field="dense_rag.reject_smoke", required=False) is False
    assert _validate_positive_dimension(8, allow_placeholders=False, field="dense_rag.dimension") == 8


# --- .example protection ---


def test_template_validation_allows_example_with_flag():
    validate_experiment_manifest(FORMAL_MANIFEST, allow_placeholders=True)


def test_formal_validation_rejects_example_manifest():
    with pytest.raises(FormalConfigError, match="\\.example"):
        validate_experiment_manifest(FORMAL_MANIFEST, allow_placeholders=False)


def test_formal_validation_rejects_example_shared_model(tmp_path):
    manifest = tmp_path / "formal_manifest.yaml"
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
        encoding="utf-8",
    )
    method = tmp_path / "method.yaml"
    method.write_text(
        yaml.safe_dump(
            {
                "method_id": "direct_llm",
                "ekell_vector": {
                    "backend": "text2vec",
                    "model_name": "m",
                    "model_version": "v",
                    "dimension": 768,
                    "reject_smoke": True,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "t",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "paper_final": True,
                "freeze_status": "provisional",
                "bundle": "bundle",
                "base_config": str(ROOT / "configs/default.yaml"),
                "shared_model_config": str(shared) + ".example",
                "require_bundle_checksum": True,
                "require_external_schema": True,
                "require_complete_case_match": True,
                "fail_on_schema_error": True,
                "fail_on_duplicate_case_id": True,
                "fail_on_missing_case": True,
                "fail_on_extra_case": True,
                "main_table_methods": ["direct_llm"],
                "methods": [
                    {
                        "method_id": "direct_llm",
                        "config": str(method),
                        "enabled": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="shared_model_config"):
        validate_experiment_manifest(manifest, allow_placeholders=False)


def test_formal_validation_requires_existing_config_files(tmp_path):
    manifest = tmp_path / "formal_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "t",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "paper_final": True,
                "freeze_status": "provisional",
                "bundle": "bundle",
                "base_config": str(ROOT / "configs/default.yaml"),
                "shared_model_config": str(tmp_path / "missing_shared.yaml"),
                "require_bundle_checksum": True,
                "require_external_schema": True,
                "require_complete_case_match": True,
                "fail_on_schema_error": True,
                "fail_on_duplicate_case_id": True,
                "fail_on_missing_case": True,
                "fail_on_extra_case": True,
                "main_table_methods": ["direct_llm"],
                "methods": [{"method_id": "direct_llm", "config": "missing.yaml", "enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FormalConfigError, match="not found"):
        validate_experiment_manifest(manifest, allow_placeholders=False)


def test_formal_validation_accepts_complete_real_config(tmp_path):
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: deepseek\n  model_version: v1\n",
        encoding="utf-8",
    )
    method = tmp_path / "direct.yaml"
    method.write_text(
        "method_id: direct_llm\nllm:\n  max_tokens: 1200\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "formal_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "t",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "paper_final": True,
                "freeze_status": "provisional",
                "bundle": "bundle",
                "base_config": str(ROOT / "configs/default.yaml"),
                "shared_model_config": str(shared),
                "require_bundle_checksum": True,
                "require_external_schema": True,
                "require_complete_case_match": True,
                "fail_on_schema_error": True,
                "fail_on_duplicate_case_id": True,
                "fail_on_missing_case": True,
                "fail_on_extra_case": True,
                "main_table_methods": ["direct_llm"],
                "methods": [{"method_id": "direct_llm", "config": str(method), "enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    result = validate_experiment_manifest(manifest, allow_placeholders=False, validation_stage="dry_run")
    assert result["valid"] is True
    assert result["validation_stage"] == "dry_run"
