"""Strict Formal config type validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from external_baselines.common.formal_config_validator import (
    FormalConfigError,
    _validate_exact_bool,
    _validate_positive_dimension,
    validate_dense_config_for_real_run,
    validate_ekell_vector_for_formal,
    validate_experiment_manifest,
    validate_hybrid_config_for_real_run,
    validate_llm_for_formal,
    validate_method_config,
)
from external_baselines.common.strict_config_types import exact_number

ROOT = Path(__file__).resolve().parents[1]


def _write_strict_manifest(tmp_path: Path, monkeypatch, *, mutate=None) -> Path:
    import yaml

    import external_baselines.common.freeze_manifest as freeze_manifest

    monkeypatch.setattr(freeze_manifest, "validate_freeze_manifest", lambda *_args, **_kwargs: None)
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n"
        "  provider: openai_compatible\n"
        "  model: contract-generation-v1\n"
        "  model_version: v1\n"
        "  api_key_env: OFFLINE_TEST_API_KEY\n"
        "  temperature: 0.0\n"
        "  top_p: 1.0\n"
        "  max_tokens: 1024\n"
        "  seed: 20260710\n",
        encoding="utf-8",
    )
    method = tmp_path / "direct.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    payload = {
        "experiment_id": "strict_manifest",
        "schema_version": "firebench-interop-v1",
        "track": "A_shared_outcome",
        "run_mode": "formal",
        "base_config": str(ROOT / "configs/default.yaml"),
        "shared_model_config": str(shared),
        "freeze_manifest": str(freeze),
        "freeze_status": "frozen",
        "paper_final": True,
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
    if mutate is not None:
        mutate(payload)
    manifest = tmp_path / "strict_manifest.yaml"
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return manifest


def _validate_strict_manifest(path: Path) -> dict:
    return validate_experiment_manifest(path, validation_stage="formal")


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


def test_formal_manifest_requires_explicit_experiment_id(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.pop("experiment_id"))
    with pytest.raises(FormalConfigError, match="experiment_id"):
        _validate_strict_manifest(path)


def test_formal_manifest_rejects_null_experiment_id(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.__setitem__("experiment_id", None))
    with pytest.raises(FormalConfigError, match="experiment_id.*exact YAML string"):
        _validate_strict_manifest(path)


def test_formal_manifest_requires_explicit_schema_version(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.pop("schema_version"))
    with pytest.raises(FormalConfigError, match="schema_version"):
        _validate_strict_manifest(path)


def test_formal_manifest_rejects_null_schema_version(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.__setitem__("schema_version", None))
    with pytest.raises(FormalConfigError, match="schema_version.*exact YAML string"):
        _validate_strict_manifest(path)


def test_formal_manifest_rejects_wrong_schema_version(tmp_path, monkeypatch):
    path = _write_strict_manifest(
        tmp_path,
        monkeypatch,
        mutate=lambda payload: payload.__setitem__("schema_version", "firebench-interop-v2"),
    )
    with pytest.raises(FormalConfigError, match="schema_version must be exactly"):
        _validate_strict_manifest(path)


def test_formal_manifest_requires_explicit_track(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.pop("track"))
    with pytest.raises(FormalConfigError, match="track"):
        _validate_strict_manifest(path)


def test_formal_manifest_rejects_null_track(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.__setitem__("track", None))
    with pytest.raises(FormalConfigError, match="track.*exact YAML string"):
        _validate_strict_manifest(path)


def test_formal_manifest_requires_explicit_base_config(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.pop("base_config"))
    with pytest.raises(FormalConfigError, match="base_config"):
        _validate_strict_manifest(path)


def test_formal_manifest_rejects_null_base_config(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.__setitem__("base_config", None))
    with pytest.raises(FormalConfigError, match="base_config.*exact YAML string"):
        _validate_strict_manifest(path)


def test_formal_manifest_rejects_null_output_when_present(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.__setitem__("output", None))
    with pytest.raises(FormalConfigError, match="output.*exact YAML string"):
        _validate_strict_manifest(path)


def test_formal_manifest_rejects_null_run_manifest_when_present(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.__setitem__("run_manifest", None))
    with pytest.raises(FormalConfigError, match="run_manifest.*exact YAML string"):
        _validate_strict_manifest(path)


def test_formal_manifest_rejects_identity_wrong_type(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch, mutate=lambda payload: payload.__setitem__("track", ["A"]))
    with pytest.raises(FormalConfigError, match="track.*exact YAML string"):
        _validate_strict_manifest(path)


def test_formal_manifest_accepts_explicit_identity_strings(tmp_path, monkeypatch):
    path = _write_strict_manifest(tmp_path, monkeypatch)
    result = _validate_strict_manifest(path)
    assert result["valid"] is True


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


def _hybrid_config(**hybrid_overrides):
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
        "hybrid_rag": {
            "reject_smoke": True,
            "top_k": 3,
            "candidate_pool": 5,
            "rrf_k": 60.0,
            "lexical_weight": 1.0,
            "dense_weight": 1.0,
        },
    }
    base["hybrid_rag"].update(hybrid_overrides)
    return base


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "0.0"),
        ("temperature", True),
        ("top_p", "1.0"),
        ("top_p", False),
        ("max_tokens", 1024.0),
        ("max_tokens", True),
        ("seed", "42"),
        ("max_retries", True),
    ],
)
def test_formal_llm_numeric_rejects_invalid_types(field, value):
    config = _formal_llm_config(**{field: value})
    with pytest.raises(FormalConfigError):
        validate_llm_for_formal(config, validation_stage="formal")


def test_formal_llm_top_p_rejects_out_of_range():
    with pytest.raises(FormalConfigError, match="llm.top_p"):
        validate_llm_for_formal(_formal_llm_config(top_p=1.5), validation_stage="formal")


def test_formal_hybrid_top_k_rejects_bool():
    with pytest.raises(FormalConfigError, match="hybrid_rag.top_k"):
        validate_hybrid_config_for_real_run(
            _hybrid_config(top_k=True),
            allow_placeholders=True,
            validation_stage="template",
        )


def test_formal_hybrid_top_k_rejects_string():
    with pytest.raises(FormalConfigError, match="hybrid_rag.top_k"):
        validate_hybrid_config_for_real_run(
            _hybrid_config(top_k="5"),
            allow_placeholders=True,
            validation_stage="template",
        )


def test_formal_hybrid_candidate_pool_rejects_float():
    with pytest.raises(FormalConfigError, match="hybrid_rag.candidate_pool"):
        validate_hybrid_config_for_real_run(
            _hybrid_config(candidate_pool=10.0),
            allow_placeholders=True,
            validation_stage="template",
        )


def test_formal_hybrid_rrf_k_rejects_string():
    with pytest.raises(FormalConfigError, match="hybrid_rag.rrf_k"):
        validate_hybrid_config_for_real_run(
            _hybrid_config(rrf_k="60"),
            allow_placeholders=True,
            validation_stage="template",
        )


def test_formal_hybrid_weight_rejects_bool():
    with pytest.raises(FormalConfigError, match="hybrid_rag.lexical_weight"):
        validate_hybrid_config_for_real_run(
            _hybrid_config(lexical_weight=False),
            allow_placeholders=True,
            validation_stage="template",
        )


def test_formal_hybrid_valid_numeric_types_pass():
    validate_hybrid_config_for_real_run(
        _hybrid_config(top_k=3, candidate_pool=5, rrf_k=60, lexical_weight=1, dense_weight=1),
        allow_placeholders=True,
        validation_stage="template",
    )


def test_null_allow_model_env_override_is_rejected():
    with pytest.raises(FormalConfigError, match="allow_model_env_override"):
        validate_llm_for_formal(
            _formal_llm_config(allow_model_env_override=None),
            validation_stage="formal",
        )


def test_missing_allow_model_env_override_defaults_false():
    config = _formal_llm_config()
    del config["llm"]["allow_model_env_override"]
    validate_llm_for_formal(config, validation_stage="formal")


def test_exact_number_rejects_nan():
    with pytest.raises(ValueError, match="field.*finite number"):
        exact_number(float("nan"), field="field")


def test_exact_number_rejects_positive_infinity():
    with pytest.raises(ValueError, match="field.*finite number"):
        exact_number(float("inf"), field="field")


def test_exact_number_rejects_negative_infinity():
    with pytest.raises(ValueError, match="field.*finite number"):
        exact_number(float("-inf"), field="field")


def test_exact_number_accepts_finite_int():
    assert exact_number(1, field="field") == 1.0


def test_exact_number_accepts_finite_float():
    assert exact_number(0.25, field="field") == 0.25


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", float("nan")),
        ("top_p", float("nan")),
        ("timeout_sec", float("inf")),
        ("timeout_sec", float("-inf")),
    ],
)
def test_formal_llm_rejects_nonfinite_numbers(field, value):
    config = _formal_llm_config(**{field: value})
    with pytest.raises(FormalConfigError, match=fr"llm\.{field}.*finite number"):
        validate_llm_for_formal(config, validation_stage="formal")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rrf_k", float("nan")),
        ("lexical_weight", float("inf")),
        ("dense_weight", float("-inf")),
    ],
)
def test_formal_hybrid_rejects_nonfinite_numbers(field, value):
    with pytest.raises(FormalConfigError, match=fr"hybrid_rag\.{field}.*finite number"):
        validate_hybrid_config_for_real_run(
            _hybrid_config(**{field: value}),
            allow_placeholders=True,
            validation_stage="template",
        )


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("provider", 123, "llm.provider.*exact YAML string"),
        ("model", True, "llm.model.*exact YAML string"),
        ("model_version", 2026, "llm.model_version.*exact YAML string"),
        ("api_key_env", False, "llm.api_key_env.*exact YAML string"),
        ("model", "   ", "llm.model.*non-empty string"),
    ],
)
def test_formal_llm_rejects_non_string_identity_fields(field, value, pattern):
    config = _formal_llm_config(**{field: value})
    with pytest.raises(FormalConfigError, match=pattern):
        validate_llm_for_formal(config, validation_stage="formal")


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("backend", 123, "dense_rag.backend.*exact YAML string"),
        ("model_name", False, "dense_rag.model_name.*exact YAML string"),
        ("model_version", 2026, "dense_rag.model_version.*exact YAML string"),
        ("index_path", {"path": "idx"}, "dense_rag.index_path.*exact YAML string"),
    ],
)
def test_formal_dense_rejects_non_string_identity_fields(field, value, pattern):
    with pytest.raises(FormalConfigError, match=pattern):
        validate_dense_config_for_real_run(
            _dense_config(**{field: value}),
            allow_placeholders=True,
            validation_stage="template",
        )


def test_formal_rejects_non_string_ekell_prompt_dir():
    config = _controlled_ekell_config()
    config["ekell_style"]["prompt_dir"] = 123
    with pytest.raises(FormalConfigError, match="ekell_style.prompt_dir.*exact YAML string"):
        validate_ekell_vector_for_formal(config, allow_placeholders=True)


def test_formal_accepts_exact_nonempty_identity_strings():
    validate_llm_for_formal(_formal_llm_config(), validation_stage="formal")
    validate_dense_config_for_real_run(
        _dense_config(),
        allow_placeholders=True,
        validation_stage="template",
    )
    validate_hybrid_config_for_real_run(
        _hybrid_config(),
        allow_placeholders=True,
        validation_stage="template",
    )
