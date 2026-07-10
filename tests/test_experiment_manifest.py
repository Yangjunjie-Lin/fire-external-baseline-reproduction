"""Experiment manifest merge order and formal command surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from external_baselines.common.experiment_manifest import (
    MAIN_TABLE_METHODS,
    build_method_config,
    enabled_methods,
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
