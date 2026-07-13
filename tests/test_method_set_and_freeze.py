"""Method-set and freeze-stage validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from external_baselines.common.experiment_manifest import enabled_methods, load_experiment_manifest, resolve_method_set
from external_baselines.common.formal_config_validator import FormalConfigError, validate_experiment_manifest
from external_baselines.method_registry import comparison_suite_methods, main_table_methods

ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = ROOT / "configs/experiments/controlled_main_table_v1.yaml.example"


def _patch_formal_resource_checks(monkeypatch) -> None:
    import external_baselines.common.formal_config_validator as validator
    import external_baselines.common.freeze_manifest as freeze_manifest
    import external_baselines.interop.bundle as bundle

    monkeypatch.setattr(bundle, "load_runner_bundle", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(validator, "validate_method_config", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(validator, "validate_hybrid_config_for_real_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(freeze_manifest, "validate_freeze_manifest", lambda *_args, **_kwargs: None)


def _write_comparison_manifest(
    tmp_path: Path,
    *,
    methods: list[dict] | None = None,
    freeze_status: str = "frozen",
    comparison_methods: list[str] | None = None,
) -> Path:
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
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"freeze_status":"frozen"}\n', encoding="utf-8")
    method_entries = methods
    if method_entries is None:
        method_entries = []
        for mid in [
            "direct_llm",
            "bm25_rag",
            "ekell_style_controlled_shared_llm",
            "dense_rag",
            "hybrid_rag",
            "ekell_style_enhanced",
        ]:
            cfg = tmp_path / f"{mid}.yaml"
            cfg.write_text(f"method_id: {mid}\n", encoding="utf-8")
            method_entries.append(
                {
                    "method_id": mid,
                    "config": str(cfg),
                    "enabled": mid != "ekell_style_enhanced",
                }
            )
    payload = {
        "schema_version": "firebench-interop-v1",
        "experiment_id": "comparison_test",
        "track": "A_shared_outcome",
        "run_mode": "formal",
        "paper_final": True,
        "freeze_status": freeze_status,
        "freeze_manifest": str(freeze),
        "bundle": str(tmp_path / "runner_bundle"),
        "base_config": str(ROOT / "configs/default.yaml"),
        "shared_model_config": str(shared),
        "require_bundle_checksum": True,
        "require_external_schema": True,
        "require_complete_case_match": True,
        "fail_on_schema_error": True,
        "fail_on_duplicate_case_id": True,
        "fail_on_missing_case": True,
        "fail_on_extra_case": True,
        "main_table_methods": ["direct_llm", "bm25_rag", "ekell_style_controlled_shared_llm"],
        "comparison_suite_methods": comparison_methods or list(comparison_suite_methods()),
        "methods": method_entries,
    }
    manifest = tmp_path / "comparison.yaml"
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return manifest


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


def test_official_manifest_template_method_entries_are_formal_compatible(tmp_path: Path, monkeypatch) -> None:
    _patch_formal_resource_checks(monkeypatch)
    raw = yaml.safe_load(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    raw["freeze_status"] = "frozen"
    raw["bundle"] = str(tmp_path / "runner_bundle")
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"freeze_status":"frozen"}\n', encoding="utf-8")
    raw["freeze_manifest"] = str(freeze)
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: openai_compatible\n  model: contract-generation-v1\n"
        "  model_version: v1\n  api_key_env: OFFLINE_TEST_API_KEY\n"
        "  temperature: 0.0\n  top_p: 1.0\n  max_tokens: 1024\n  seed: 20260710\n",
        encoding="utf-8",
    )
    raw["shared_model_config"] = str(shared)
    for entry in raw["methods"]:
        cfg = tmp_path / f"{entry['method_id']}.yaml"
        cfg.write_text(f"method_id: {entry['method_id']}\n", encoding="utf-8")
        entry["config"] = str(cfg)
    manifest = tmp_path / "controlled_main_table_v1.yaml"
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = validate_experiment_manifest(
        manifest,
        validation_stage="formal",
        method_set="comparison_suite",
    )

    assert result["valid"] is True


def test_method_entry_order_does_not_control_comparison_order(tmp_path: Path, monkeypatch) -> None:
    _patch_formal_resource_checks(monkeypatch)
    methods = []
    for mid in [
        "ekell_style_controlled_shared_llm",
        "hybrid_rag",
        "dense_rag",
        "bm25_rag",
        "direct_llm",
        "ekell_style_enhanced",
    ]:
        cfg = tmp_path / f"{mid}.yaml"
        cfg.write_text(f"method_id: {mid}\n", encoding="utf-8")
        methods.append({"method_id": mid, "config": str(cfg), "enabled": mid != "ekell_style_enhanced"})
    manifest = _write_comparison_manifest(tmp_path, methods=methods)

    result = validate_experiment_manifest(
        manifest,
        validation_stage="formal",
        method_set="comparison_suite",
    )
    resolved = enabled_methods(load_experiment_manifest(manifest), method_set="comparison_suite")

    assert result["valid"] is True
    assert [entry["method_id"] for entry in resolved] == list(comparison_suite_methods())


def test_disabled_enhanced_entry_is_allowed(tmp_path: Path, monkeypatch) -> None:
    _patch_formal_resource_checks(monkeypatch)
    manifest = _write_comparison_manifest(tmp_path)

    result = validate_experiment_manifest(
        manifest,
        validation_stage="formal",
        method_set="comparison_suite",
    )

    assert result["valid"] is True


def test_enabled_enhanced_entry_is_rejected(tmp_path: Path, monkeypatch) -> None:
    _patch_formal_resource_checks(monkeypatch)
    manifest = _write_comparison_manifest(tmp_path)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    for entry in raw["methods"]:
        if entry["method_id"] == "ekell_style_enhanced":
            entry["enabled"] = True
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(FormalConfigError, match="Non-comparison method entry must be disabled"):
        validate_experiment_manifest(
            manifest,
            validation_stage="formal",
            method_set="comparison_suite",
        )


def test_disabled_required_entry_is_rejected(tmp_path: Path, monkeypatch) -> None:
    _patch_formal_resource_checks(monkeypatch)
    manifest = _write_comparison_manifest(tmp_path)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    for entry in raw["methods"]:
        if entry["method_id"] == "dense_rag":
            entry["enabled"] = False
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(FormalConfigError, match="Required comparison suite method entry disabled: dense_rag"):
        validate_experiment_manifest(
            manifest,
            validation_stage="formal",
            method_set="comparison_suite",
        )


def test_missing_required_entry_is_rejected(tmp_path: Path, monkeypatch) -> None:
    _patch_formal_resource_checks(monkeypatch)
    manifest = _write_comparison_manifest(tmp_path)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["methods"] = [entry for entry in raw["methods"] if entry["method_id"] != "hybrid_rag"]
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(FormalConfigError, match="missing method entries"):
        validate_experiment_manifest(
            manifest,
            validation_stage="formal",
            method_set="comparison_suite",
        )


def test_duplicate_method_entry_is_rejected(tmp_path: Path, monkeypatch) -> None:
    _patch_formal_resource_checks(monkeypatch)
    manifest = _write_comparison_manifest(tmp_path)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["methods"].append(dict(raw["methods"][0]))
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(FormalConfigError, match="Duplicate method_id"):
        validate_experiment_manifest(
            manifest,
            validation_stage="formal",
            method_set="comparison_suite",
        )


def test_freeze_candidate_accepts_provisional_without_existing_freeze(tmp_path: Path, monkeypatch) -> None:
    _patch_formal_resource_checks(monkeypatch)
    manifest = _write_comparison_manifest(tmp_path, freeze_status="provisional")
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw.pop("freeze_manifest")
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = validate_experiment_manifest(
        manifest,
        validation_stage="freeze_candidate",
        method_set="comparison_suite",
    )

    assert result["valid"] is True


def test_freeze_candidate_requires_formal_bundle_authority(tmp_path: Path, monkeypatch) -> None:
    import external_baselines.common.formal_config_validator as validator
    import external_baselines.interop.bundle as bundle

    calls = []
    _patch_formal_resource_checks(monkeypatch)

    def fake_load_runner_bundle(_path, *, formal=False):
        calls.append(formal)
        return {"ok": True}

    monkeypatch.setattr(bundle, "load_runner_bundle", fake_load_runner_bundle)
    monkeypatch.setattr(validator, "validate_method_config", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(validator, "validate_hybrid_config_for_real_run", lambda *_args, **_kwargs: None)
    manifest = _write_comparison_manifest(tmp_path, freeze_status="provisional")

    validate_experiment_manifest(
        manifest,
        validation_stage="freeze_candidate",
        method_set="comparison_suite",
    )

    assert calls == [True]


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
