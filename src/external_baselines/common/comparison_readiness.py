"""Comparison-suite resource readiness (per-method diagnostics; no secrets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file
from external_baselines.common.environment import environment_variable_presence, load_local_environment
from external_baselines.common.experiment_manifest import (
    build_method_config,
    load_experiment_manifest,
    resolve_method_set,
)
from external_baselines.common.formal_config_validator import _is_placeholder
from external_baselines.common.io import read_yaml
from external_baselines.common.main_project_readiness import assess_main_project_readiness
from external_baselines.common.strict_config_types import read_exact_int, read_exact_number
from external_baselines.interop.bundle import load_runner_bundle, validate_bundle_checksum

ROOT = Path(__file__).resolve().parents[3]

CONTROLLED_PROMPT_FILES = (
    "query_decomposition.txt",
    "logical_expression_validation.txt",
    "stepwise_projection.txt",
    "stepwise_intersection.txt",
    "stepwise_union.txt",
    "stepwise_negation.txt",
    "final_kg_grounded_response.txt",
)


def _corpus_dir(config: dict[str, Any], bundle_corpus: str | None = None) -> Path:
    if bundle_corpus:
        return Path(bundle_corpus)
    return Path((config.get("paths") or {}).get("corpus_dir") or "data/corpus")


def _resolve_bundle_path(
    *,
    bundle_path: str | Path | None,
    experiment: dict[str, Any],
    resources_path: Path,
) -> Path | None:
    if bundle_path and not _is_placeholder(bundle_path):
        return Path(str(bundle_path))
    exp_bundle = experiment.get("bundle")
    if exp_bundle and not _is_placeholder(exp_bundle):
        return Path(str(exp_bundle))
    if resources_path.is_file():
        try:
            resources = read_yaml(resources_path)
            main = (resources or {}).get("main_project") or {}
            rb = main.get("runner_bundle_path")
            if rb and not _is_placeholder(rb):
                return Path(str(rb))
        except Exception:  # noqa: BLE001
            pass
    return None


def _check_bundle(bundle: dict[str, Any] | None, *, bundle_path: Path | None) -> dict[str, Any]:
    reasons: list[str] = []
    if bundle is None or bundle_path is None:
        return {"ready": False, "reasons": ["runner_bundle_missing"], "path": str(bundle_path) if bundle_path else None}
    checksum = validate_bundle_checksum(bundle)
    if not checksum.get("ok"):
        reasons.append("runner_bundle_checksum_failed")
    scenarios = bundle.get("scenarios_path")
    if not scenarios or not Path(str(scenarios)).is_file():
        reasons.append("input_cases_missing")
    else:
        lines = [ln for ln in Path(str(scenarios)).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            reasons.append("input_cases_empty")
    schema = bundle.get("prediction_schema_path")
    if not schema or not Path(str(schema)).is_file():
        reasons.append("prediction_schema_missing")
    corpus = bundle.get("corpus_dir")
    if not corpus or not Path(str(corpus)).is_dir():
        reasons.append("corpus_dir_missing")
    else:
        evidence = Path(str(corpus)) / "evidence_chunks.jsonl"
        if not evidence.is_file():
            reasons.append("evidence_chunks_missing")
        else:
            lines = [ln for ln in evidence.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                reasons.append("evidence_chunks_empty")
    corpus_manifest = bundle.get("corpus_manifest")
    if isinstance(corpus_manifest, dict) and not corpus_manifest.get("aggregate_sha256"):
        reasons.append("corpus_manifest_checksum_missing")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "path": str(bundle_path),
        "checksum": bundle.get("producer_declared_checksum") or bundle.get("consumer_computed_bundle_hash"),
        "corpus_dir": bundle.get("corpus_dir"),
        "scenarios_path": bundle.get("scenarios_path"),
    }


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


def _check_dense(config: dict[str, Any], *, corpus: Path) -> dict[str, Any]:
    reasons: list[str] = []
    dense = config.get("dense_rag") or {}
    if str(dense.get("backend") or "").lower() in {"", "smoke_hash_embedding", "smoke", "hash"}:
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
        reasons.append("dense_index_path_placeholder")
        return {"ready": False, "reasons": reasons}

    path = Path(str(index_path))
    if not path.is_dir():
        reasons.append("dense_index_dir_missing")
        return {"ready": False, "reasons": reasons}
    if not (path / "index_manifest.json").is_file():
        reasons.append("dense_index_manifest_missing")
        return {"ready": False, "reasons": reasons}

    try:
        from external_baselines.retrieval.dense_index import DenseIndexError, load_dense_index

        corpus_checksum = None
        evidence = corpus / "evidence_chunks.jsonl"
        if evidence.is_file():
            corpus_checksum = sha256_file(evidence)
        payload = load_dense_index(
            path,
            expected_model_name=str(dense.get("model_name")) if dense.get("model_name") else None,
            expected_model_version=str(dense.get("model_version")) if dense.get("model_version") else None,
            expected_backend=str(dense.get("backend")) if dense.get("backend") else None,
            expected_dimension=int(dim) if dim is not None and not _is_placeholder(dim) else None,
            expected_corpus_checksum=corpus_checksum,
        )
        manifest = payload.get("manifest") or {}
        if not bool(manifest.get("actual_embedding_used", True)):
            reasons.append("dense_actual_embedding_used_false")
        if bool(manifest.get("smoke_fallback_used")):
            reasons.append("dense_smoke_fallback_used")
    except DenseIndexError as exc:
        msg = str(exc).lower()
        if "embeddings_checksum" in msg:
            reasons.append("dense_index_embeddings_checksum_mismatch")
        elif "documents_checksum" in msg or "documents_file_checksum" in msg:
            reasons.append("dense_index_documents_checksum_mismatch")
        elif "index_checksum" in msg:
            reasons.append("dense_index_checksum_mismatch")
        elif "model_version" in msg:
            reasons.append("dense_index_model_version_mismatch")
        elif "model_name" in msg:
            reasons.append("dense_index_model_name_mismatch")
        elif "backend" in msg:
            reasons.append("dense_index_backend_mismatch")
        elif "dimension" in msg:
            reasons.append("dense_index_dimension_mismatch")
        elif "corpus_checksum" in msg:
            reasons.append("dense_index_corpus_checksum_mismatch")
        elif "missing required file" in msg:
            reasons.append("dense_index_incomplete")
        else:
            reasons.append(f"dense_index_load_failed:{exc}")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"dense_index_load_failed:{exc}")
    return {"ready": not reasons, "reasons": reasons, "index_path": str(path)}


def _check_hybrid(
    config: dict[str, Any],
    dense_report: dict[str, Any],
    bm25_report: dict[str, Any],
    *,
    dense_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not bm25_report.get("ready"):
        reasons.append("bm25_dependency_not_ready")
    if not dense_report.get("ready"):
        reasons.append("dense_dependency_not_ready")
    hybrid = config.get("hybrid_rag") or {}
    dense = config.get("dense_rag") or {}
    dense_ref = (dense_config or {}).get("dense_rag") or {}

    hybrid_index = dense.get("index_path") or hybrid.get("index_path")
    dense_index = dense_ref.get("index_path")
    if dense_index and hybrid_index and str(Path(str(hybrid_index))) != str(Path(str(dense_index))):
        reasons.append("hybrid_dense_index_path_mismatch")

    try:
        retrieval = config.get("retrieval") or {}
        default_top_k = read_exact_int(
            retrieval,
            "top_k",
            field="retrieval.top_k",
            default=5,
            minimum=1,
        )
        top_k = read_exact_int(
            hybrid,
            "top_k",
            field="hybrid_rag.top_k",
            default=default_top_k,
            minimum=1,
        )
        candidate_pool = read_exact_int(
            hybrid,
            "candidate_pool",
            field="hybrid_rag.candidate_pool",
            default=top_k,
            minimum=0,
        )
        rrf_k = read_exact_number(
            hybrid,
            "rrf_k",
            field="hybrid_rag.rrf_k",
            default=60.0,
            minimum=0,
            minimum_inclusive=False,
        )
        lexical_weight = read_exact_number(
            hybrid,
            "lexical_weight",
            field="hybrid_rag.lexical_weight",
            default=1.0,
            minimum=0,
            minimum_inclusive=False,
        )
        dense_weight = read_exact_number(
            hybrid,
            "dense_weight",
            field="hybrid_rag.dense_weight",
            default=1.0,
            minimum=0,
            minimum_inclusive=False,
        )
        if candidate_pool < top_k:
            reasons.append("hybrid_candidate_pool_lt_top_k")
        if rrf_k <= 0:
            reasons.append("hybrid_rrf_k_invalid")
        if lexical_weight <= 0:
            reasons.append("hybrid_lexical_weight_invalid")
        if dense_weight <= 0:
            reasons.append("hybrid_dense_weight_invalid")
    except Exception:
        reasons.append("hybrid_rrf_params_invalid")
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

    prompt_dir = Path(str((config.get("ekell_style") or {}).get("prompt_dir") or "configs/prompts/controlled"))
    candidates = [prompt_dir, ROOT / prompt_dir, ROOT / "configs/prompts/controlled"]
    prompt_root = next((p for p in candidates if p.is_dir()), None)
    if prompt_root is None:
        reasons.append("ekell_prompt_dir_missing")
    else:
        for name in CONTROLLED_PROMPT_FILES:
            if not (prompt_root / name).is_file():
                reasons.append(f"ekell_prompt_missing:{name}")

    try:
        from external_baselines.ekell_style.logical_query import execute_query  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"ekell_fol_executor_import_failed:{exc}")

    index_path = vector.get("index_path")
    if not index_path or _is_placeholder(index_path):
        reasons.append("ekell_index_path_placeholder")
        return {"ready": False, "reasons": reasons}

    path = Path(str(index_path))
    if not path.is_dir():
        reasons.append("ekell_index_dir_missing")
        return {"ready": False, "reasons": reasons}

    try:
        from external_baselines.ekell_style.vector_index import VectorIndex, VectorIndexError

        dim = vector.get("dimension")
        VectorIndex.load_directory(
            path,
            expected_backend=str(vector.get("backend")) if vector.get("backend") else None,
            expected_model_name=str(vector.get("model_name")) if vector.get("model_name") else None,
            expected_model_version=str(vector.get("model_version")) if vector.get("model_version") else None,
            expected_dimension=int(dim) if dim is not None and not _is_placeholder(dim) else None,
            require_real_embedding=True,
        )
    except VectorIndexError as exc:
        msg = str(exc).lower()
        if "embeddings_checksum" in msg:
            reasons.append("ekell_index_embeddings_checksum_mismatch")
        elif "documents" in msg and "checksum" in msg:
            reasons.append("ekell_index_documents_checksum_mismatch")
        elif "index_checksum" in msg:
            reasons.append("ekell_index_checksum_mismatch")
        elif "model_version" in msg:
            reasons.append("ekell_index_model_version_mismatch")
        elif "kg_checksum" in msg:
            reasons.append("ekell_index_kg_checksum_mismatch")
        elif "corpus_checksum" in msg:
            reasons.append("ekell_index_corpus_checksum_mismatch")
        elif "missing required file" in msg:
            reasons.append("ekell_index_incomplete")
        else:
            reasons.append(f"ekell_index_load_failed:{exc}")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"ekell_index_load_failed:{exc}")
    return {"ready": not reasons, "reasons": reasons, "index_path": str(path)}


def assess_comparison_readiness(
    *,
    experiment_manifest: str | Path,
    resources_path: str | Path | None = None,
    method_set: str = "comparison_suite",
    bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    load_local_environment()
    manifest = load_experiment_manifest(experiment_manifest)
    method_ids = resolve_method_set(manifest, method_set=method_set)
    resources_file = Path(resources_path or "configs/local/experiment_resources.yaml")
    readiness = assess_main_project_readiness(resources_file) if resources_file.is_file() else {}

    resolved_bundle = _resolve_bundle_path(
        bundle_path=bundle_path,
        experiment=manifest,
        resources_path=resources_file,
    )
    bundle: dict[str, Any] | None = None
    if resolved_bundle is not None and resolved_bundle.exists():
        try:
            bundle = load_runner_bundle(resolved_bundle)
        except Exception as exc:  # noqa: BLE001
            bundle = None
            bundle_report = {
                "ready": False,
                "reasons": [f"runner_bundle_load_failed:{exc}"],
                "path": str(resolved_bundle),
            }
        else:
            bundle_report = _check_bundle(bundle, bundle_path=resolved_bundle)
    else:
        bundle_report = _check_bundle(None, bundle_path=resolved_bundle)

    bundle_corpus = bundle.get("corpus_dir") if isinstance(bundle, dict) else None

    methods_out: dict[str, Any] = {}
    by_entry = {e["method_id"]: e for e in manifest["methods"]}
    dense_report: dict[str, Any] = {"ready": False, "reasons": ["not_evaluated"]}
    bm25_report: dict[str, Any] = {"ready": False, "reasons": ["not_evaluated"]}
    dense_cfg: dict[str, Any] | None = None

    for mid in method_ids:
        entry = by_entry.get(mid)
        if entry is None:
            methods_out[mid] = {"ready": False, "reasons": ["method_entry_missing"]}
            continue
        cfg = build_method_config(manifest, entry)
        if bundle_corpus:
            cfg.setdefault("paths", {})["corpus_dir"] = bundle_corpus
        corpus = _corpus_dir(cfg, bundle_corpus)
        if mid == "direct_llm":
            methods_out[mid] = _check_direct(cfg)
        elif mid == "bm25_rag":
            bm25_report = _check_bm25(cfg, corpus)
            methods_out[mid] = bm25_report
        elif mid == "dense_rag":
            dense_cfg = cfg
            dense_report = _check_dense(cfg, corpus=corpus)
            methods_out[mid] = dense_report
        elif mid == "hybrid_rag":
            if "bm25_rag" not in methods_out:
                bm25_report = _check_bm25(cfg, corpus)
            if "dense_rag" not in methods_out:
                dense_report = _check_dense(cfg, corpus=corpus)
                dense_cfg = cfg
            methods_out[mid] = _check_hybrid(cfg, dense_report, bm25_report, dense_config=dense_cfg)
        elif mid == "ekell_style_controlled_shared_llm":
            methods_out[mid] = _check_ekell(cfg, corpus)
        else:
            methods_out[mid] = {"ready": False, "reasons": ["unsupported_in_comparison_suite"]}

    comparison_ready = (
        all(m.get("ready") for m in methods_out.values())
        and bool(bundle_report.get("ready"))
        and bool(readiness.get("main_project_v1_ready"))
    )
    return {
        "comparison_ready": comparison_ready,
        "method_set": method_set,
        "bundle": bundle_report,
        "methods": methods_out,
        "main_project_readiness": {
            "main_project_v1_ready": readiness.get("main_project_v1_ready"),
            "reasons": readiness.get("reasons"),
        },
        "api_env_presence": environment_variable_presence(),
    }
