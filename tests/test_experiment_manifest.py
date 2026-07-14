"""Experiment manifest merge order and formal command surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from external_baselines.common.experiment_manifest import (
    MAIN_TABLE_METHODS,
    MethodEntryError,
    build_method_config,
    enabled_methods,
    get_method_entry,
    load_experiment_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_controlled_main_table_manifest_example(tmp_path):
    src = ROOT / "configs" / "experiments" / "controlled_main_table_v1.yaml.example"
    text = src.read_text(encoding="utf-8")
    text = text.replace("paper_final: true", "paper_final: false")
    text = text.replace(
        "shared_model_config: configs/models/shared_real_model.yaml.example",
        "shared_model_config: configs/deterministic_heuristic_smoke.yaml",
    )
    text = text.replace("bundle: path/to/runner_bundle", "bundle: unused")
    path = tmp_path / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    manifest = load_experiment_manifest(path)
    assert manifest["freeze_status"] == "provisional"
    main = enabled_methods(manifest, include_supplemental=False)
    assert [m["method_id"] for m in main] == list(MAIN_TABLE_METHODS)
    supp = enabled_methods(manifest, include_supplemental=True)
    assert len(supp) == 5
    assert [m["method_id"] for m in supp] == [
        "direct_llm",
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    ]


def test_method_config_merge_order(tmp_path):
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "llm:\n  provider: siliconflow\n  model: shared-model\n  model_version: v1\n  temperature: 0.2\n",
        encoding="utf-8",
    )
    method = tmp_path / "method.yaml"
    method.write_text("retrieval:\n  top_k: 5\nllm:\n  max_tokens: 900\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("llm:\n  provider: heuristic\n  model: base\n  max_tokens: 1200\n", encoding="utf-8")
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        f"""
experiment_id: t
shared_model_config: {shared.as_posix()}
base_config: {base.as_posix()}
methods:
  - method_id: bm25_rag
    config: {method.as_posix()}
    paper_table_role: main_table
""".strip(),
        encoding="utf-8",
    )
    manifest = load_experiment_manifest(exp)
    cfg = build_method_config(manifest, manifest["methods"][0])
    assert cfg["llm"]["provider"] == "siliconflow"
    assert cfg["llm"]["model"] == "shared-model"
    assert cfg["llm"]["max_tokens"] == 900
    assert cfg["retrieval"]["top_k"] == 5


def test_get_method_entry_returns_matching_entry(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        """
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
methods:
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
    enabled: true
""".strip(),
        encoding="utf-8",
    )
    manifest = load_experiment_manifest(exp)
    entry = get_method_entry(manifest, "direct_llm")
    assert entry["method_id"] == "direct_llm"


def test_get_method_entry_rejects_missing_method(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        """
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
methods:
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
""".strip(),
        encoding="utf-8",
    )
    manifest = load_experiment_manifest(exp)
    with pytest.raises(MethodEntryError, match="does not define method entry"):
        get_method_entry(manifest, "dense_rag")


def test_get_method_entry_rejects_disabled_method_in_formal(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        """
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
methods:
  - method_id: bm25_rag
    config: configs/methods/bm25_rag.yaml
    enabled: false
""".strip(),
        encoding="utf-8",
    )
    manifest = load_experiment_manifest(exp)
    with pytest.raises(MethodEntryError, match="disabled"):
        get_method_entry(manifest, "bm25_rag", require_enabled=True)


def test_build_method_config_rejects_string_method_id(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        """
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
methods:
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
""".strip(),
        encoding="utf-8",
    )
    manifest = load_experiment_manifest(exp)
    with pytest.raises(TypeError, match="method entry mapping"):
        build_method_config(manifest, "direct_llm")  # type: ignore[arg-type]


def test_build_method_config_requires_mapping():
    with pytest.raises(TypeError, match="method entry mapping"):
        build_method_config({}, "not-a-mapping")  # type: ignore[arg-type]


def _minimal_manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(
        f"""
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
methods:
{body}
""".strip(),
        encoding="utf-8",
    )
    return path


def test_manifest_enabled_rejects_string_false(tmp_path):
    path = _minimal_manifest(
        tmp_path,
        """
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
    enabled: "false"
""",
    )
    with pytest.raises(ValueError, match="enabled must be an exact boolean"):
        load_experiment_manifest(path)


def test_manifest_enabled_rejects_zero(tmp_path):
    path = _minimal_manifest(
        tmp_path,
        """
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
    enabled: 0
""",
    )
    with pytest.raises(ValueError, match="enabled must be an exact boolean"):
        load_experiment_manifest(path)


def test_manifest_enabled_accepts_exact_false(tmp_path):
    path = _minimal_manifest(
        tmp_path,
        """
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
    enabled: false
""",
    )
    manifest = load_experiment_manifest(path)
    assert manifest["methods"][0]["enabled"] is False


def test_manifest_enabled_defaults_true_when_missing(tmp_path):
    path = _minimal_manifest(
        tmp_path,
        """
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
""",
    )
    manifest = load_experiment_manifest(path)
    assert manifest["methods"][0]["enabled"] is True


def test_template_manifest_can_use_documented_defaults(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        """
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
methods:
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
""".strip(),
        encoding="utf-8",
    )
    manifest = load_experiment_manifest(path)
    assert manifest["schema_version"] == "firebench-interop-v1"
    assert manifest["track"] == "A_shared_outcome"
    assert manifest["base_config"] == "configs/default.yaml"
    assert manifest["output"] == "outputs/firebench_interop_v1_predictions.jsonl"
    assert manifest["run_manifest"] == "outputs/interop_run_manifest.json"


def test_manifest_paper_final_rejects_string_true(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        """
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
paper_final: "true"
methods:
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="paper_final must be an exact boolean"):
        load_experiment_manifest(path)


def test_manifest_rejects_numeric_shared_model_config(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        """
experiment_id: t
shared_model_config: 123
methods:
  - method_id: direct_llm
    config: configs/methods/direct_llm.yaml
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shared_model_config.*exact YAML string"):
        load_experiment_manifest(path)


def test_manifest_rejects_boolean_method_id(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        """
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
methods:
  - method_id: true
    config: configs/methods/direct_llm.yaml
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="methods\\[0\\]\\.method_id.*exact YAML string"):
        load_experiment_manifest(path)


def test_manifest_rejects_non_string_method_config(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        """
experiment_id: t
shared_model_config: configs/deterministic_heuristic_smoke.yaml
methods:
  - method_id: direct_llm
    config: false
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="methods\\[0\\]\\.config.*exact YAML string"):
        load_experiment_manifest(path)


def test_run_interop_rejects_multi_config():
    import runpy
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_interop_baselines.py"
    argv = [
        str(script),
        "--experiment-manifest",
        "configs/experiments/controlled_main_table_v1.yaml.example",
        "--config",
        "a.yaml",
        "--config",
        "b.yaml",
    ]
    old = sys.argv[:]
    try:
        sys.argv = argv
        with pytest.raises(SystemExit) as ei:
            runpy.run_path(str(script), run_name="__main__")
        assert "Multiple --config" in str(ei.value)
    finally:
        sys.argv = old


def test_manifest_relative_resources_publish_complete_resolved_path_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import external_baselines.common.experiment_manifest as experiment_manifest_module

    monkeypatch.setattr(experiment_manifest_module, "REPOSITORY_ROOT", tmp_path)
    base = tmp_path / "base.yaml"
    shared = tmp_path / "shared.yaml"
    method = tmp_path / "method.yaml"
    bundle = tmp_path / "bundle"
    freeze = tmp_path / "freeze.json"
    base.write_text("{}\n", encoding="utf-8")
    shared.write_text("llm: {}\n", encoding="utf-8")
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    bundle.mkdir()
    freeze.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "experiment.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "experiment_id: resolved-path-contract",
                "base_config: base.yaml",
                "shared_model_config: shared.yaml",
                "bundle: bundle",
                "freeze_manifest: freeze.json",
                "methods:",
                "  - method_id: direct_llm",
                "    config: method.yaml",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_experiment_manifest(manifest_path)

    manifest_provenance = manifest["path_provenance"]["experiment_manifest"]
    assert manifest_provenance["path_policy"] == "repository_relative"
    assert manifest_provenance["canonical_path"] == "experiment.yaml"
    assert manifest_provenance["resolved_path_authoritative"] is False
    for prefix, expected in (
        ("base_config", base),
        ("shared_model_config", shared),
        ("bundle", bundle),
        ("freeze_manifest", freeze),
    ):
        assert manifest[f"{prefix}_resolved"] == str(expected.resolve())
        assert manifest[f"{prefix}_path_policy"] == "experiment_relative"
        assert manifest[f"{prefix}_canonical_path"] == expected.name
    entry = manifest["methods"][0]
    assert entry["config"] == str(method.resolve())
    assert entry["config_path_policy"] == "experiment_relative"
    assert entry["config_canonical_path"] == "method.yaml"


def test_manifest_relative_path_wins_over_repository_candidate(tmp_path: Path) -> None:
    relative = "configs/default.yaml"
    local = tmp_path / relative
    local.parent.mkdir()
    local.write_text("project: {name: manifest-local}\n", encoding="utf-8")
    shared = tmp_path / "shared.yaml"
    shared.write_text("llm: {}\n", encoding="utf-8")
    method = tmp_path / "method.yaml"
    method.write_text("method_id: direct_llm\n", encoding="utf-8")
    manifest_path = tmp_path / "experiment.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                f"base_config: {relative}",
                "shared_model_config: shared.yaml",
                "methods:",
                "  - method_id: direct_llm",
                "    config: method.yaml",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_experiment_manifest(manifest_path)

    assert manifest["base_config_resolved"] == str(local.resolve())
    assert manifest["base_config_path_policy"] == "experiment_relative"
    provenance = manifest["path_provenance"]["base_config"]
    assert provenance["alternate_repository_candidate_exists"] is True
