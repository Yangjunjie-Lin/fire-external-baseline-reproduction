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
)

ROOT = Path(__file__).resolve().parents[1]


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
