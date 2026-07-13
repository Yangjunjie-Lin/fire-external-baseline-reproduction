"""Strict Formal config type validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from external_baselines.common.formal_config_validator import (
    FormalConfigError,
    _validate_exact_bool,
    _validate_positive_dimension,
    validate_dense_config_for_real_run,
    validate_llm_for_formal,
    validate_method_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _formal_llm_config(**llm_overrides):
    llm = {
        "provider": "openai_compatible",
        "model": "glm-4",
        "model_version": "v1",
        "api_key_env": "API_KEY",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": 42,
        "allow_model_env_override": False,
    }
    llm.update(llm_overrides)
    return {"paper_final": True, "llm": llm}


def _controlled_ekell_config(**ekell_overrides):
    config = {
        "method_id": "ekell_style_controlled_shared_llm",
        "paper_final": True,
        "llm": _formal_llm_config()["llm"],
        "ekell_style": {},
        "ekell_vector": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v1",
            "dimension": 8,
            "reject_smoke": True,
            "index_path": "REQUIRED_BEFORE_FORMAL_RUN",
        },
    }
    config["ekell_style"].update(ekell_overrides)
    return config


def _dense_config(**dense_overrides):
    base = {
        "paper_final": True,
        "dense_rag": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v1",
            "dimension": 8,
            "reject_smoke": True,
            "normalize_embeddings": True,
            "allow_index_rebuild": False,
            "index_path": "indexes/dense",
        },
    }
    base["dense_rag"].update(dense_overrides)
    return base


def test_formal_dense_dimension_rejects_bool():
    with pytest.raises(FormalConfigError, match="exact YAML integer"):
        _validate_positive_dimension(True, field="dense_rag.dimension")


def test_formal_dense_dimension_rejects_string():
    with pytest.raises(FormalConfigError, match="exact YAML integer"):
        _validate_positive_dimension("8", field="dense_rag.dimension")


def test_formal_dense_dimension_rejects_float():
    with pytest.raises(FormalConfigError, match="exact YAML integer"):
        _validate_positive_dimension(8.0, field="dense_rag.dimension")


def test_formal_reject_smoke_rejects_string_false():
    with pytest.raises(FormalConfigError, match="exact boolean"):
        _validate_exact_bool("false", field="dense_rag.reject_smoke", required=True)


def test_formal_allow_index_rebuild_rejects_string_false():
    with pytest.raises(FormalConfigError, match="exact boolean"):
        _validate_exact_bool("false", field="dense_rag.allow_index_rebuild", required=False)


def test_formal_enable_thinking_requires_bool():
    config = {
        "paper_final": True,
        "llm": {
            "provider": "openai_compatible",
            "model": "glm-4",
            "model_version": "v1",
            "api_key_env": "API_KEY",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1024,
            "seed": 42,
            "enable_thinking": "false",
        },
    }
    with pytest.raises(FormalConfigError, match="exact boolean"):
        validate_llm_for_formal(config, validation_stage="formal")


def test_template_mode_allows_explicit_placeholders():
    config = _dense_config(dimension="REQUIRED_BEFORE_FORMAL_RUN", index_path="path/to/index")
    validate_dense_config_for_real_run(
        config,
        allow_placeholders=True,
        validation_stage="template",
    )


def test_jsonschema_is_core_project_dependency():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "jsonschema>=" in text
    assert text.index("jsonschema") < text.index("[project.optional-dependencies]")


@pytest.mark.parametrize(
    ("value", "pattern"),
    [
        ("false", "llm.allow_model_env_override"),
        ("true", "llm.allow_model_env_override"),
        (0, "llm.allow_model_env_override"),
        (1, "llm.allow_model_env_override"),
    ],
)
def test_formal_allow_model_env_override_rejects_non_bool(value, pattern):
    with pytest.raises(FormalConfigError, match=pattern):
        validate_llm_for_formal(_formal_llm_config(allow_model_env_override=value))


def test_formal_allow_model_env_override_accepts_exact_false():
    validate_llm_for_formal(_formal_llm_config(allow_model_env_override=False))


def test_method_config_rejects_string_paper_final():
    config = _formal_llm_config()
    config["paper_final"] = "false"
    with pytest.raises(FormalConfigError, match="paper_final must be an exact boolean"):
        validate_method_config(config, method_id="direct_llm", allow_placeholders=True)


def test_paper_fidelity_model_run_rejects_string_false():
    config = {
        "track": "paper_fidelity",
        "paper_original_output_format": True,
        "controlled_output_format": False,
        "official_reproduction": False,
        "paper_fidelity_model_run": "false",
    }
    with pytest.raises(FormalConfigError, match="paper_fidelity_model_run"):
        from external_baselines.common.formal_config_validator import validate_paper_fidelity_method_config

        validate_paper_fidelity_method_config(config)


def test_paper_fidelity_model_run_rejects_integer():
    config = {
        "track": "paper_fidelity",
        "paper_original_output_format": True,
        "controlled_output_format": False,
        "official_reproduction": False,
        "paper_fidelity_model_run": 1,
    }
    with pytest.raises(FormalConfigError, match="paper_fidelity_model_run"):
        from external_baselines.common.formal_config_validator import validate_paper_fidelity_method_config

        validate_paper_fidelity_method_config(config)


def test_controlled_ekell_hook_rejects_string_false():
    with pytest.raises(FormalConfigError, match="ekell_style.dense_entity_retrieval"):
        validate_method_config(
            _controlled_ekell_config(dense_entity_retrieval="false"),
            method_id="ekell_style_controlled_shared_llm",
            allow_placeholders=True,
        )


def test_controlled_ekell_hook_rejects_integer_zero():
    with pytest.raises(FormalConfigError, match="ekell_style.self_consistency"):
        validate_method_config(
            _controlled_ekell_config(self_consistency=0),
            method_id="ekell_style_controlled_shared_llm",
            allow_placeholders=True,
        )


def test_controlled_ekell_hook_accepts_exact_false():
    validate_method_config(
        _controlled_ekell_config(dense_entity_retrieval=False),
        method_id="ekell_style_controlled_shared_llm",
        allow_placeholders=True,
    )
