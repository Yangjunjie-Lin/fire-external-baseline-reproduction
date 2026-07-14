from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from external_baselines.common.experiment_manifest import load_experiment_manifest
from external_baselines.common.path_resolution import PathContext, resolve_declared_path


def test_relative_path_does_not_depend_on_cwd(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repo"
    resource = repository / "configs" / "resource.txt"
    resource.parent.mkdir(parents=True)
    resource.write_text("identity", encoding="utf-8")
    context = PathContext(repository_root=repository)
    first = resolve_declared_path(
        "configs/resource.txt",
        context=context,
        policy="repository_relative",
        expected_kind="file",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    second = resolve_declared_path(
        "configs/resource.txt",
        context=context,
        policy="repository_relative",
        expected_kind="file",
    )
    assert first == second == resource.resolve()


def test_experiment_relative_path_resolves_from_manifest_parent(tmp_path: Path) -> None:
    manifest = tmp_path / "experiments" / "manifest.yaml"
    resource = manifest.parent / "shared.yaml"
    resource.parent.mkdir(parents=True)
    resource.write_text("llm: {}\n", encoding="utf-8")
    resolved = resolve_declared_path(
        "shared.yaml",
        context=PathContext(
            repository_root=tmp_path,
            experiment_manifest_path=manifest,
        ),
        policy="experiment_relative",
        expected_kind="file",
    )
    assert resolved == resource.resolve()


def test_repository_relative_path_resolves_from_repo_root(tmp_path: Path) -> None:
    resource = tmp_path / "configs" / "base.yaml"
    resource.parent.mkdir()
    resource.write_text("{}\n", encoding="utf-8")
    resolved = resolve_declared_path(
        "configs/base.yaml",
        context=PathContext(repository_root=tmp_path),
        policy="repository_relative",
        expected_kind="file",
    )
    assert resolved == resource.resolve()


def test_same_declared_path_resolves_identically_from_different_cwds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repo"
    target = repository / "data" / "corpus"
    target.mkdir(parents=True)
    context = PathContext(repository_root=repository)
    results = []
    for cwd in (tmp_path, repository / "data"):
        monkeypatch.chdir(cwd)
        results.append(
            resolve_declared_path(
                "data/corpus",
                context=context,
                policy="repository_relative",
                expected_kind="directory",
            )
        )
    assert results == [target.resolve(), target.resolve()]


def test_path_resolution_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes_repository_relative_root"):
        resolve_declared_path(
            "../outside.txt",
            context=PathContext(repository_root=tmp_path / "repo"),
            policy="repository_relative",
            must_exist=False,
        )


def test_path_resolution_rejects_symlink_escape(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = repository / "linked"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="escapes_repository_relative_root"):
        resolve_declared_path(
            "linked",
            context=PathContext(repository_root=repository),
            policy="repository_relative",
            expected_kind="directory",
        )


def test_bundle_relative_path_cannot_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(ValueError, match="escapes_bundle_relative_root"):
        resolve_declared_path(
            "../secret.json",
            context=PathContext(repository_root=tmp_path, bundle_root=bundle),
            policy="bundle_relative",
            must_exist=False,
        )
    with pytest.raises(ValueError, match="bundle_relative_path_must_be_relative"):
        resolve_declared_path(
            tmp_path / "outside.json",
            context=PathContext(repository_root=tmp_path, bundle_root=bundle),
            policy="bundle_relative",
            must_exist=False,
        )


def test_base_config_resolution_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    (experiment_root / "base.yaml").write_text("retrieval: {top_k: 1}\n", encoding="utf-8")
    (experiment_root / "shared.yaml").write_text("llm: {}\n", encoding="utf-8")
    (experiment_root / "method.yaml").write_text("method_id: direct_llm\n", encoding="utf-8")
    manifest = experiment_root / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "shared_model_config": "shared.yaml",
                "base_config": "base.yaml",
                "methods": [
                    {"method_id": "direct_llm", "config": "method.yaml"}
                ],
            }
        ),
        encoding="utf-8",
    )
    first = load_experiment_manifest(manifest)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    second = load_experiment_manifest(manifest)
    assert first["base_config_resolved"] == second["base_config_resolved"] == str(
        (experiment_root / "base.yaml").resolve()
    )
    assert first["methods"][0]["config"] == second["methods"][0]["config"]
