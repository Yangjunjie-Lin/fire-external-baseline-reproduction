"""Comparison-suite resource readiness (per-method diagnostics; no secrets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_baselines.common.environment import environment_variable_presence, load_local_environment
from external_baselines.common.experiment_manifest import (
    build_method_config,
    load_experiment_manifest,
    resolve_method_set,
)
from external_baselines.common.formal_config_validator import _is_placeholder
from external_baselines.common.main_project_readiness import assess_main_project_readiness

ROOT = Path(__file__).resolve().parents[3]


def _corpus_dir(config: dict[str, Any], bundle_corpus: str | None = None) -> Path:
    if bundle_corpus:
        return Path(bundle_corpus)
    return Path((config.get("paths") or {}).get("corpus_dir") or "data/corpus")


def _check_direct(config: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    llm = config.get("llm") or {}
    provider = str(llm.get("provider") or "").lower()
    if provider in {"", "heuristic", "local", "smoke"}:
        reasons.append("llm_provider_not_real")
    if not llm.get("model") or _is_placeholder(llm.get("model")):
        reasons.append("llm_model_missing")
    if not (llm.get("model_version") or llm.get("version")) or _is_placeholder(
        llm.get("model_version") or llm.get("version")
    ):
        reasons.append("llm_model_version_missing")
    presence = environment_variable_presence([str(llm.get("api_key_env") or "SILICONFLOW_API_KEY")])
    if "missing" in presence.values():
        reasons.append("api_env_missing")
    return {"ready": not reasons, "reasons": reasons}


def _check_bm25(config: dict[str, Any], corpus: Path) -> dict[str, Any]:
    reasons: list[str] = []
    evidence = corpus / "evidence_chunks.jsonl"
    if not evidence.is_file():
        reasons.append("evidence_chunks_missing")
    else:
        lines = [ln for ln in evidence.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            reasons.append("evidence_chunks_empty")
    return {"ready": not reasons, "reasons": reasons}


def _check_dense(config: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    dense = config.get("dense_rag") or {}
    if str(dense.get("backend") or "").lower() in {"", "smoke_hash_embedding", "smoke"}:
        reasons.append("dense_backend_not_real")
    for field in ("model_name", "model_version"):
        if not dense.get(field) or _is_placeholder(dense.get(field)):
            reasons.append(f"dense_{field}_placeholder")
    dim = dense.get("dimension", dense.get("dim"))
    try:
        if int(dim) <= 0:
            reasons.append("dense_dimension_invalid")
    except Exception:
        reasons.append("dense_dimension_invalid")
    index_path = dense.get("index_path")
    if not index_path or _is_placeholder(index_path):
        reasons.append("dense_index_missing")
    else:
        path = Path(str(index_path))
        if path.is_dir():
            if not (path / "index_manifest.json").is_file():
                reasons.append("dense_index_missing")
        elif not path.is_file():
            reasons.append("dense_index_missing")
    return {"ready": not reasons, "reasons": reasons}


def _check_hybrid(config: dict[str, Any], dense_report: dict[str, Any], bm25_report: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not bm25_report.get("ready"):
        reasons.append("bm25_dependency_not_ready")
    if not dense_report.get("ready"):
        reasons.append("dense_dependency_not_ready")
    hybrid = config.get("hybrid_rag") or {}
    for key in ("rrf_k", "lexical_weight", "dense_weight", "top_k", "candidate_pool"):
        if key in hybrid:
            try:
                float(hybrid[key])
            except Exception:
                reasons.append(f"hybrid_{key}_invalid")
    return {"ready": not reasons, "reasons": reasons}


def _check_ekell(config: dict[str, Any], corpus: Path) -> dict[str, Any]:
    reasons: list[str] = []
    for name in ("entities.jsonl", "relations.jsonl", "triples.jsonl"):
        if not (corpus / name).is_file():
            reasons.append(f"ekell_{name}_missing")
    vector = config.get("ekell_vector") or {}
    if str(vector.get("backend") or "").lower() in {"", "smoke", "hash", "smoke_hash_embedding"}:
        reasons.append("ekell_backend_not_real")
    for field in ("model_name", "model_version"):
        if not vector.get(field) or _is_placeholder(vector.get(field)):
            reasons.append(f"ekell_{field}_placeholder")
    index_path = vector.get("index_path")
    if not index_path or _is_placeholder(index_path):
        reasons.append("ekell_index_missing")
    else:
        path = Path(str(index_path))
        if not (path.is_dir() or path.is_file()):
            reasons.append("ekell_index_missing")
    # FOL / prompt chain presence (controlled or paper_fidelity trees)
    prompt_dir = Path(str((config.get("ekell_style") or {}).get("prompt_dir") or "configs/prompts/controlled"))
    candidates = [prompt_dir, ROOT / prompt_dir, ROOT / "configs/prompts/controlled", ROOT / "configs/prompts/paper_fidelity"]
    if not any(p.is_dir() for p in candidates):
        reasons.append("ekell_prompt_dir_missing")
    return {"ready": not reasons, "reasons": reasons}


def assess_comparison_readiness(
    *,
    experiment_manifest: str | Path,
    resources_path: str | Path | None = None,
    method_set: str = "comparison_suite",
) -> dict[str, Any]:
    load_local_environment()
    manifest = load_experiment_manifest(experiment_manifest)
    method_ids = resolve_method_set(manifest, method_set=method_set)
    resources_file = Path(resources_path or "configs/local/experiment_resources.yaml")
    readiness = assess_main_project_readiness(resources_file) if resources_file.is_file() else {}

    methods_out: dict[str, Any] = {}
    by_entry = {e["method_id"]: e for e in manifest["methods"]}
    dense_report: dict[str, Any] = {"ready": False, "reasons": ["not_evaluated"]}
    bm25_report: dict[str, Any] = {"ready": False, "reasons": ["not_evaluated"]}

    for mid in method_ids:
        entry = by_entry.get(mid)
        if entry is None:
            methods_out[mid] = {"ready": False, "reasons": ["method_entry_missing"]}
            continue
        cfg = build_method_config(manifest, entry)
        corpus = _corpus_dir(cfg)
        if mid == "direct_llm":
            methods_out[mid] = _check_direct(cfg)
        elif mid == "bm25_rag":
            bm25_report = _check_bm25(cfg, corpus)
            methods_out[mid] = bm25_report
        elif mid == "dense_rag":
            dense_report = _check_dense(cfg)
            methods_out[mid] = dense_report
        elif mid == "hybrid_rag":
            if "bm25_rag" not in methods_out:
                bm25_report = _check_bm25(cfg, corpus)
            if "dense_rag" not in methods_out:
                dense_report = _check_dense(cfg)
            methods_out[mid] = _check_hybrid(cfg, dense_report, bm25_report)
        elif mid == "ekell_style_controlled_shared_llm":
            methods_out[mid] = _check_ekell(cfg, corpus)
        else:
            methods_out[mid] = {"ready": False, "reasons": ["unsupported_in_comparison_suite"]}

    comparison_ready = all(m.get("ready") for m in methods_out.values()) and bool(
        readiness.get("main_project_v1_ready")
    )
    return {
        "comparison_ready": comparison_ready,
        "method_set": method_set,
        "methods": methods_out,
        "main_project_readiness": {
            "main_project_v1_ready": readiness.get("main_project_v1_ready"),
            "reasons": readiness.get("reasons"),
        },
        "api_env_presence": environment_variable_presence(),
    }
