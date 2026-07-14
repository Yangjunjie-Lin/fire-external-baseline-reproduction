"""Offline official build -> validate -> freeze -> Formal preflight lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from external_baselines.common.decision_suite_preflight import preflight_decision_suite
from external_baselines.common.experiment_manifest import (
    build_method_config,
    enabled_methods,
    load_experiment_manifest,
)
from external_baselines.common.formal_config_validator import (
    FormalConfigError,
    validate_experiment_manifest,
)
from external_baselines.interop.bundle import load_runner_bundle
from scripts import build_comparison_indexes as builder
from scripts import create_freeze_manifest
from tests.test_comparison_index_builder import _invoke, _mutate_yaml, _write_experiment
from tests.test_decision_comparison_suite import _make_runner_bundle
from tests.test_dense_real_index import FakeEmbeddingModel


def _run_builder(argv: list[str], output: Path) -> tuple[int, dict[str, Any]]:
    try:
        builder.main([*argv, "--output", str(output)])
    except SystemExit as exc:
        code = int(exc.code)
    else:
        code = 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is (code == 0)
    return code, report


def _install_fake_embedding_backend(monkeypatch) -> None:
    import external_baselines.dense_rag.pipeline as dense_pipeline
    import external_baselines.retrieval.embedding_backends as embedding_backends

    original = embedding_backends.create_embedding_backend

    def offline_backend(
        backend: str,
        *,
        model_name: str,
        model_version: str,
        dimension: int,
        paper_final: bool = False,
        reject_smoke: bool = False,
        **_kwargs,
    ):
        return original(
            backend,
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            paper_final=paper_final,
            reject_smoke=reject_smoke,
            model=FakeEmbeddingModel(dimension),
        )

    monkeypatch.setattr(embedding_backends, "create_embedding_backend", offline_backend)
    monkeypatch.setattr(dense_pipeline, "create_embedding_backend", offline_backend)


def _complete_lifecycle(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    import external_baselines.common.decision_suite_preflight as preflight_module
    import external_baselines.common.experiment_manifest as experiment_manifest_module
    import external_baselines.common.formal_config_validator as formal_validator

    bundle_path = _make_runner_bundle(tmp_path)
    manifest_path, paths = _write_experiment(
        tmp_path,
        bundle_path,
        build_indexes=False,
    )
    dense_relative = paths["dense_index"].relative_to(tmp_path).as_posix()
    ekell_relative = paths["ekell_index"].relative_to(tmp_path).as_posix()
    prompt_relative = (tmp_path / "prompts").relative_to(tmp_path).as_posix()
    _mutate_yaml(paths["dense_rag"], "dense_rag", "index_path", dense_relative)
    _mutate_yaml(paths["hybrid_rag"], "dense_rag", "index_path", dense_relative)
    _mutate_yaml(
        paths["ekell_style_controlled_shared_llm"],
        "ekell_vector",
        "index_path",
        ekell_relative,
    )
    _mutate_yaml(
        paths["ekell_style_controlled_shared_llm"],
        "ekell_style",
        "prompt_dir",
        prompt_relative,
    )
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(create_freeze_manifest, "ROOT", tmp_path)
    monkeypatch.setattr(experiment_manifest_module, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(formal_validator, "ROOT_REL", tmp_path)
    monkeypatch.setattr(preflight_module, "ROOT_REL", tmp_path)
    _install_fake_embedding_backend(monkeypatch)

    build_code, build_report = _run_builder(
        ["--experiment-manifest", str(manifest_path)],
        tmp_path / "build_report.json",
    )
    assert build_code == 0
    validate_code, validate_report = _invoke(manifest_path, tmp_path)
    assert validate_code == 0

    selected_dev = tmp_path / "selected_dev.json"
    selected_dev.write_text(
        '{"run_id":"offline-dev","selection":"safety-gated"}\n',
        encoding="utf-8",
    )
    freeze_path = tmp_path / "freeze.json"
    create_freeze_manifest.main(
        [
            "--experiment-manifest",
            str(manifest_path),
            "--selected-dev-run",
            selected_dev.name,
            "--bundle",
            str(bundle_path),
            "--output",
            str(freeze_path),
        ]
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze["ekell_prompt_bundle"]["required_prompt_files"]
    provenance = freeze["path_provenance"]
    expected_provenance = {
        "experiment_manifest",
        "base_config",
        "shared_model_config",
        "method_configs",
        "runner_bundle",
        "selected_dev_evidence",
        "prompt_dir",
        "dense_index",
        "ekell_index",
    }
    assert expected_provenance <= provenance.keys()
    for label in expected_provenance - {"method_configs"}:
        entry = provenance[label]
        assert type(entry.get("declared_path")) is str
        assert type(entry.get("canonical_path")) is str
        assert type(entry.get("path_policy")) is str
        assert type(entry.get("resolved_path_at_freeze")) is str
        assert entry["resolved_path_authoritative"] is False
        assert entry["path_policy"] in {"experiment_relative", "repository_relative"}
        assert not Path(entry["canonical_path"]).is_absolute()
    for entry in provenance["method_configs"].values():
        assert type(entry.get("declared_path")) is str
        assert type(entry.get("canonical_path")) is str
        assert type(entry.get("path_policy")) is str
        assert type(entry.get("resolved_path_at_freeze")) is str
        assert entry["resolved_path_authoritative"] is False
        assert entry["path_policy"] in {"experiment_relative", "repository_relative"}
        assert not Path(entry["canonical_path"]).is_absolute()

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["freeze_status"] = "frozen"
    raw["freeze_manifest"] = freeze_path.name
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    formal_result = validate_experiment_manifest(
        manifest_path,
        validation_stage="formal",
        method_set="comparison_suite",
        runtime_bundle_path=bundle_path,
    )
    assert formal_result["valid"] is True

    experiment = load_experiment_manifest(manifest_path)
    bundle = load_runner_bundle(bundle_path, formal=True)
    method_configs: dict[str, dict[str, Any]] = {}
    for entry in enabled_methods(experiment, method_set="comparison_suite"):
        config = build_method_config(experiment, entry)
        config.setdefault("paths", {})["corpus_dir"] = bundle["corpus_dir"]
        method_configs[entry["method_id"]] = config
    preflight = preflight_decision_suite(
        method_ids=list(method_configs),
        method_configs=method_configs,
        runner_bundle=bundle_path,
        execution_stage="formal",
        experiment_manifest=manifest_path,
    )
    assert preflight["ok"] is True, {
        "methods": {
            method_id: {
                "ok": report.get("ok"),
                "config_valid": report.get("config_valid"),
                "resources_valid": report.get("resources_valid"),
                "errors": report.get("errors"),
            }
            for method_id, report in (preflight.get("methods") or {}).items()
            if not report.get("ok")
        },
        "shared_errors": preflight.get("shared_errors"),
        "generation_ok": (preflight.get("shared_generation_identity") or {}).get("ok"),
        "generation_mismatches": (
            preflight.get("shared_generation_identity") or {}
        ).get("mismatches"),
        "formal_runtime_errors": (
            preflight.get("formal_runtime_validation") or {}
        ).get("errors"),
    }
    return {
        "repository_root": tmp_path,
        "bundle": bundle_path,
        "manifest": manifest_path,
        "freeze": freeze_path,
        "paths": paths,
        "build_report": build_report,
        "validate_report": validate_report,
        "preflight": preflight,
        "method_configs": method_configs,
    }


def test_official_manifest_relative_lifecycle_build_validate_freeze_finalize_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _complete_lifecycle(tmp_path, monkeypatch)
    assert result["build_report"]["ok"] is True
    assert result["validate_report"]["ok"] is True
    assert result["preflight"]["formal_runtime_validation"]["ok"] is True


def test_official_lifecycle_stops_when_validate_only_fails(
    tmp_path: Path,
) -> None:
    bundle = _make_runner_bundle(tmp_path)
    manifest, _paths = _write_experiment(tmp_path, bundle, build_indexes=False)
    code, report = _invoke(manifest, tmp_path)
    assert code == 1
    assert report["ok"] is False
    assert "dense_index_path_missing" in report["errors"]
    assert not (tmp_path / "freeze.json").exists()


def test_official_lifecycle_rejects_modified_prompt_after_freeze(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _complete_lifecycle(tmp_path, monkeypatch)
    prompt_dir = Path(
        result["method_configs"]["ekell_style_controlled_shared_llm"]["ekell_style"][
            "prompt_dir"
        ]
    )
    if not prompt_dir.is_absolute():
        prompt_dir = result["repository_root"] / prompt_dir
    (prompt_dir / "stepwise_projection.txt").write_text("modified\n", encoding="utf-8")
    with pytest.raises(FormalConfigError, match="prompt"):
        validate_experiment_manifest(
            result["manifest"],
            validation_stage="formal",
            method_set="comparison_suite",
            runtime_bundle_path=result["bundle"],
        )


def test_official_lifecycle_rejects_modified_kg_after_freeze(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _complete_lifecycle(tmp_path, monkeypatch)
    triple_path = result["bundle"] / "corpus" / "triples.jsonl"
    triple_path.write_text(
        triple_path.read_text(encoding="utf-8")
        + '{"head":"new","relation":"r","tail":"new"}\n',
        encoding="utf-8",
    )
    report = preflight_decision_suite(
        method_ids=list(result["method_configs"]),
        method_configs=result["method_configs"],
        runner_bundle=result["bundle"],
        execution_stage="formal",
        experiment_manifest=result["manifest"],
    )
    assert report["ok"] is False
    assert report["shared_errors"]


def test_official_lifecycle_rejects_modified_base_config_after_freeze(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _complete_lifecycle(tmp_path, monkeypatch)
    base = Path(load_experiment_manifest(result["manifest"])["base_config_resolved"])
    base.write_text("retrieval: {top_k: 99}\n", encoding="utf-8")
    with pytest.raises(FormalConfigError, match="base_config_sha256"):
        validate_experiment_manifest(
            result["manifest"],
            validation_stage="formal",
            method_set="comparison_suite",
            runtime_bundle_path=result["bundle"],
        )


def test_official_lifecycle_is_cwd_independent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _complete_lifecycle(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    validated = validate_experiment_manifest(
        result["manifest"],
        validation_stage="formal",
        method_set="comparison_suite",
        runtime_bundle_path=result["bundle"],
    )
    assert validated["valid"] is True
    report = preflight_decision_suite(
        method_ids=list(result["method_configs"]),
        method_configs=result["method_configs"],
        runner_bundle=result["bundle"],
        execution_stage="formal",
        experiment_manifest=result["manifest"],
    )
    assert report["ok"] is True
