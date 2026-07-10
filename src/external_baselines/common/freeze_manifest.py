"""Freeze manifest create/validate helpers for formal comparison runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file
from external_baselines.common.formal_config_validator import FormalConfigError, _is_placeholder
from external_baselines.common.io import read_json


def prompt_tree_checksum(prompt_dir: str | Path) -> str | None:
    root = Path(prompt_dir)
    if not root.is_dir():
        from external_baselines.common.formal_config_validator import ROOT_REL

        root = ROOT_REL / str(prompt_dir)
    if not root.is_dir():
        return None
    digests: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = sha256_file(path)
            if digest:
                digests.append(f"{path.relative_to(root).as_posix()}:{digest}")
    import hashlib

    return hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()


def build_freeze_manifest_payload(
    *,
    experiment_manifest_path: str | Path,
    experiment_raw: dict[str, Any],
    selected_dev_run: str | Path,
    bundle_checksum: str | None = None,
    corpus_checksum: str | None = None,
    schema_checksum: str | None = None,
    method_config_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    experiment_manifest_path = Path(experiment_manifest_path)
    selected = Path(selected_dev_run)
    shared = experiment_raw.get("shared_model_config")
    method_hashes: dict[str, str | None] = {}
    for mid, rel in (method_config_paths or {}).items():
        method_hashes[mid] = sha256_file(rel) if rel and Path(rel).is_file() else None

    llm = {}
    shared_path = Path(str(shared)) if shared else None
    if shared_path and shared_path.is_file():
        from external_baselines.common.io import read_yaml

        shared_cfg = read_yaml(shared_path)
        llm_block = shared_cfg.get("llm") or {}
        llm = {
            "provider": llm_block.get("provider"),
            "model": llm_block.get("model"),
            "model_version": llm_block.get("model_version") or llm_block.get("version"),
        }

    return {
        "freeze_id": "controlled_comparison_v1",
        "freeze_status": "frozen",
        "selected_dev_run_evidence": str(selected).replace("\\", "/"),
        "selection_criterion": (
            "Safety-Gated + Critical Failure Rate + Risk/Action F1 + evidence support + latency"
        ),
        "experiment_manifest_sha256": sha256_file(experiment_manifest_path),
        "shared_model_config_sha256": sha256_file(shared) if shared else None,
        "method_config_sha256": method_hashes,
        "prompt_tree_sha256": prompt_tree_checksum("configs/prompts/controlled"),
        "runner_bundle_checksum": bundle_checksum,
        "corpus_checksum": corpus_checksum,
        "prediction_schema_checksum": schema_checksum,
        "llm": llm,
        "embedding": {
            "backend": "text2vec",
            "model_name": "BAAI/bge-m3",
            "model_version": None,
            "dimension": 1024,
        },
    }


def validate_freeze_manifest(
    freeze_path: str | Path,
    *,
    experiment_manifest_path: str | Path,
    experiment_raw: dict[str, Any],
) -> dict[str, Any]:
    freeze = read_json(freeze_path)
    if not isinstance(freeze, dict):
        raise FormalConfigError("freeze_manifest must be a JSON object.")
    if str(freeze.get("freeze_status") or "").lower() != "frozen":
        raise FormalConfigError("freeze_manifest.freeze_status must be frozen.")
    evidence = freeze.get("selected_dev_run_evidence")
    if not evidence or _is_placeholder(evidence):
        raise FormalConfigError("freeze_manifest requires selected_dev_run_evidence.")
    evidence_path = Path(str(evidence))
    if not evidence_path.is_file():
        # Allow relative to repo root
        from external_baselines.common.formal_config_validator import ROOT_REL

        evidence_path = ROOT_REL / str(evidence)
    if not evidence_path.is_file():
        raise FormalConfigError(f"selected_dev_run_evidence not found: {evidence}")

    expected_exp = sha256_file(experiment_manifest_path)
    if freeze.get("experiment_manifest_sha256") and freeze.get("experiment_manifest_sha256") != expected_exp:
        raise FormalConfigError("freeze_manifest experiment_manifest_sha256 mismatch.")

    shared = experiment_raw.get("shared_model_config")
    if shared and freeze.get("shared_model_config_sha256"):
        actual = sha256_file(shared)
        if actual != freeze.get("shared_model_config_sha256"):
            raise FormalConfigError("freeze_manifest shared_model_config_sha256 mismatch.")

    prompt_hash = freeze.get("prompt_tree_sha256")
    if prompt_hash:
        actual_prompt = prompt_tree_checksum("configs/prompts/controlled")
        if not actual_prompt:
            raise FormalConfigError("freeze_manifest prompt_tree_sha256 set but prompt tree missing.")
        if actual_prompt != prompt_hash:
            raise FormalConfigError("freeze_manifest prompt_tree_sha256 mismatch.")

    emb = freeze.get("embedding") or {}
    if emb.get("model_version") is None or _is_placeholder(emb.get("model_version")):
        raise FormalConfigError("freeze_manifest embedding.model_version must be set.")

    return {"ok": True, "freeze_id": freeze.get("freeze_id")}
