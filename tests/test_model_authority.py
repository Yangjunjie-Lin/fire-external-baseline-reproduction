"""YAML model authority and SILICONFLOW_MODEL override tests."""

from __future__ import annotations

import pytest

from external_baselines.common.formal_config_validator import FormalConfigError, validate_llm_for_formal
from external_baselines.common.llm_client import (
    llm_config_summary,
    resolve_siliconflow_model,
)


def test_yaml_model_wins_over_siliconflow_model_env(monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_MODEL", "env/should-not-win")
    model, source = resolve_siliconflow_model(
        {
            "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "allow_model_env_override": False,
        },
        paper_final=True,
    )
    assert model == "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    assert source == "yaml_config"


def test_formal_rejects_model_env_override() -> None:
    with pytest.raises(FormalConfigError):
        validate_llm_for_formal(
            {
                "paper_final": True,
                "llm": {
                    "provider": "siliconflow",
                    "model": "m",
                    "model_version": "v",
                    "allow_model_env_override": True,
                },
            }
        )


def test_dev_may_allow_explicit_model_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_MODEL", "env/dev-model")
    model, source = resolve_siliconflow_model(
        {
            "model": "yaml/model",
            "allow_model_env_override": True,
        },
        paper_final=False,
    )
    assert model == "env/dev-model"
    assert source == "env_override"


def test_model_source_recorded_as_yaml_config() -> None:
    summary = llm_config_summary(
        {
            "paper_final": True,
            "llm": {
                "provider": "siliconflow",
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                "model_version": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            },
        }
    )
    assert summary["model_source"] == "yaml_config"
    assert summary["model"] == "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def test_llm_summary_records_resolved_model(monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_MODEL", "env/ignored")
    summary = llm_config_summary(
        {
            "paper_final": False,
            "llm": {
                "provider": "siliconflow",
                "model": "yaml/canonical",
                "model_version": "yaml/canonical",
                "allow_model_env_override": False,
            },
        }
    )
    assert summary["model"] == "yaml/canonical"
    assert summary["model_source"] == "yaml_config"
