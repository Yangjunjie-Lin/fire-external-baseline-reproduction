"""Method-set and freeze-stage validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from external_baselines.common.experiment_manifest import resolve_method_set
from external_baselines.common.formal_config_validator import FormalConfigError, validate_experiment_manifest
from external_baselines.method_registry import comparison_suite_methods, main_table_methods

ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = ROOT / "configs/experiments/controlled_main_table_v1.yaml.example"


def test_main_table_method_set_has_three_methods() -> None:
    assert list(main_table_methods()) == ["direct_llm", "bm25_rag", "ekell_style_controlled_shared_llm"]


def test_comparison_suite_has_exactly_five_methods() -> None:
    assert list(comparison_suite_methods()) == [
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ]


def test_comparison_suite_excludes_enhanced() -> None:
    assert "ekell_style_enhanced" not in comparison_suite_methods()


def test_comparison_suite_excludes_paper_fidelity() -> None:
    assert "ekell_style_paper_fidelity" not in comparison_suite_methods()


def test_comparison_suite_prediction_count_is_case_count_times_five() -> None:
    case_count = 3
    assert case_count * len(comparison_suite_methods()) == 15


def test_resolve_method_set_from_manifest() -> None:
    from external_baselines.common.experiment_manifest import load_experiment_manifest

    manifest = load_experiment_manifest(FORMAL_MANIFEST)
    assert resolve_method_set(manifest, method_set="main_table") == list(main_table_methods())
    assert resolve_method_set(manifest, method_set="comparison_suite") == list(comparison_suite_methods())


def test_template_requires_provisional() -> None:
    result = validate_experiment_manifest(FORMAL_MANIFEST, validation_stage="template")
    assert result["validation_stage"] == "template"


def test_dry_run_accepts_provisional(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
        encoding="utf-8",
    )
    method = tmp_path / "direct.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    manifest = tmp_path / "m.yaml"
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
    result = validate_experiment_manifest(manifest, validation_stage="dry_run")
    assert result["valid"] is True


def test_formal_rejects_provisional(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
        encoding="utf-8",
    )
    method = tmp_path / "direct.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    manifest = tmp_path / "m.yaml"
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
    with pytest.raises(FormalConfigError, match="freeze_status=frozen"):
        validate_experiment_manifest(manifest, validation_stage="formal")


def test_formal_requires_freeze_manifest(tmp_path: Path) -> None:
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: m\n  model_version: v\n",
        encoding="utf-8",
    )
    method = tmp_path / "direct.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "firebench-interop-v1",
                "experiment_id": "t",
                "track": "A_shared_outcome",
                "run_mode": "formal",
                "paper_final": True,
                "freeze_status": "frozen",
                "freeze_manifest": None,
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
    with pytest.raises(FormalConfigError, match="freeze_manifest"):
        validate_experiment_manifest(manifest, validation_stage="formal")
