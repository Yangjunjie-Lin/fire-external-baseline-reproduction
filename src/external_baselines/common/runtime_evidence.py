"""Runtime evidence collection for formal compliance reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SMOKE_LLM_PROVIDERS = frozenset({"heuristic", "local", "smoke", "mock", "fake", ""})
SMOKE_MODEL_TOKENS = ("heuristic", "smoke", "fixture", "mock", "fake")


@dataclass
class RuntimeEvidence:
    method_id: str

    llm_provider: str | None = None
    llm_model: str | None = None
    llm_model_version: str | None = None
    llm_is_smoke: bool | None = None
    llm_initialized: bool = False

    embedding_backend: str | None = None
    embedding_model: str | None = None
    embedding_model_version: str | None = None
    actual_embedding_used: bool | None = None
    smoke_fallback_used: bool | None = None

    index_type: str | None = None
    index_path: str | None = None
    index_checksum: str | None = None
    index_manifest_sha256: str | None = None
    index_loaded: bool = False
    index_built_during_run: bool = False
    index_document_count: int | None = None

    lexical_ready: bool | None = None
    dense_dependency_index_checksum: str | None = None
    dense_dependency_backend: str | None = None
    dense_dependency_smoke_fallback_used: bool | None = None
    rrf_k: float | None = None
    candidate_pool: int | None = None

    runtime_prepared: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_smoke_llm(provider: str | None, model: str | None) -> bool:
    prov = str(provider or "").lower().strip()
    if prov in SMOKE_LLM_PROVIDERS:
        return True
    model_lower = str(model or "").lower()
    return any(token in model_lower for token in SMOKE_MODEL_TOKENS)


def collect_llm_evidence(
    *,
    method_id: str,
    config: dict[str, Any],
    llm: Any | None = None,
) -> RuntimeEvidence:
    from external_baselines.common.llm_client import llm_config_summary

    summary = llm_config_summary(config, llm)
    provider = str(summary.get("provider") or "")
    model = str(summary.get("model") or "")
    evidence = RuntimeEvidence(
        method_id=method_id,
        llm_provider=provider,
        llm_model=model,
        llm_model_version=str(summary.get("model_version") or ""),
        llm_is_smoke=_is_smoke_llm(provider, model),
        llm_initialized=llm is not None,
    )
    return evidence


def collect_dense_runtime_evidence(
    *,
    method_id: str,
    config: dict[str, Any],
    runtime: Any | None,
) -> RuntimeEvidence:
    dense_cfg = config.get("dense_rag") or {}
    evidence = collect_llm_evidence(method_id=method_id, config=config, llm=None)
    evidence.runtime_prepared = runtime is not None
    if runtime is None:
        return evidence
    manifest = dict(getattr(runtime, "index_manifest", None) or {})
    emb = getattr(runtime, "embedding_backend", None)
    index = getattr(runtime, "dense_index", None)
    audit = getattr(runtime, "audit", None)
    evidence.embedding_backend = str(
        manifest.get("backend") or getattr(emb, "backend", None) or dense_cfg.get("backend") or ""
    )
    evidence.embedding_model = str(
        manifest.get("model_name") or getattr(emb, "model_name", None) or dense_cfg.get("model_name") or ""
    )
    evidence.embedding_model_version = str(
        manifest.get("model_version")
        or getattr(emb, "model_version", None)
        or dense_cfg.get("model_version")
        or ""
    )
    evidence.actual_embedding_used = bool(
        manifest.get("actual_embedding_used", getattr(emb, "actual_embedding_used", None))
    )
    evidence.smoke_fallback_used = bool(
        manifest.get("smoke_fallback_used", getattr(emb, "smoke_fallback_used", None))
    )
    evidence.index_type = str(manifest.get("index_type") or "dense_evidence_index")
    evidence.index_path = str(
        manifest.get("index_dir") or dense_cfg.get("index_path") or getattr(index, "index_dir", "") or ""
    )
    evidence.index_checksum = str(
        manifest.get("index_checksum") or manifest.get("checksum") or getattr(index, "checksum", "") or ""
    )
    evidence.index_manifest_sha256 = evidence.index_checksum
    evidence.index_loaded = bool(getattr(audit, "index_load_count", 0) or index is not None)
    evidence.index_built_during_run = bool(getattr(runtime, "index_built_during_run", False))
    evidence.index_document_count = int(manifest.get("document_count") or 0) or None
    return evidence


def collect_hybrid_runtime_evidence(
    *,
    method_id: str,
    config: dict[str, Any],
    runtime: Any | None,
) -> RuntimeEvidence:
    hybrid_cfg = config.get("hybrid_rag") or {}
    if runtime is None:
        evidence = collect_llm_evidence(method_id=method_id, config=config, llm=None)
        evidence.runtime_prepared = False
        return evidence
    dense_rt = getattr(runtime, "dense_runtime", None)
    evidence = collect_dense_runtime_evidence(method_id=method_id, config=config, runtime=dense_rt)
    evidence.method_id = method_id
    evidence.runtime_prepared = True
    evidence.lexical_ready = getattr(runtime, "lexical_retriever", None) is not None
    evidence.dense_dependency_index_checksum = evidence.index_checksum
    evidence.dense_dependency_backend = evidence.embedding_backend
    evidence.dense_dependency_smoke_fallback_used = evidence.smoke_fallback_used
    evidence.rrf_k = float(hybrid_cfg.get("rrf_k")) if hybrid_cfg.get("rrf_k") is not None else None
    evidence.candidate_pool = int(hybrid_cfg.get("candidate_pool")) if hybrid_cfg.get("candidate_pool") is not None else None
    return evidence


def collect_ekell_runtime_evidence(
    *,
    method_id: str,
    config: dict[str, Any],
    runtime: Any | None,
) -> RuntimeEvidence:
    vector_cfg = config.get("ekell_vector") or {}
    evidence = collect_llm_evidence(method_id=method_id, config=config, llm=None)
    evidence.runtime_prepared = runtime is not None
    if runtime is None:
        return evidence
    manifest = dict(getattr(runtime, "index_manifest", None) or {})
    emb = getattr(runtime, "embedding_backend", None)
    index = getattr(runtime, "vector_index", None)
    audit = getattr(runtime, "audit", None)
    evidence.embedding_backend = str(manifest.get("backend") or getattr(emb, "backend", None) or "")
    evidence.embedding_model = str(
        manifest.get("model_name") or manifest.get("embedding_model") or getattr(emb, "model_name", None) or ""
    )
    evidence.embedding_model_version = str(
        manifest.get("model_version") or getattr(emb, "model_version", None) or ""
    )
    evidence.actual_embedding_used = bool(manifest.get("actual_embedding_used", getattr(emb, "actual_embedding_used", None)))
    evidence.smoke_fallback_used = bool(manifest.get("smoke_fallback_used", getattr(emb, "smoke_fallback_used", None)))
    evidence.index_type = str(manifest.get("index_type") or "ekell_kg_vector_index")
    evidence.index_path = str(manifest.get("index_dir") or vector_cfg.get("index_path") or "")
    evidence.index_checksum = str(manifest.get("index_checksum") or manifest.get("checksum") or "")
    evidence.index_manifest_sha256 = evidence.index_checksum
    evidence.index_loaded = bool(getattr(audit, "index_load_count", 0) or index is not None)
    evidence.index_built_during_run = bool(getattr(runtime, "index_built_during_run", False))
    evidence.index_document_count = int(manifest.get("document_count") or 0) or None
    return evidence


def collect_method_runtime_evidence(
    *,
    method_id: str,
    config: dict[str, Any],
    llm: Any | None = None,
    runtime: Any | None = None,
) -> RuntimeEvidence:

    if method_id == "dense_rag":
        evidence = collect_dense_runtime_evidence(method_id=method_id, config=config, runtime=runtime)
    elif method_id == "hybrid_rag":
        evidence = collect_hybrid_runtime_evidence(method_id=method_id, config=config, runtime=runtime)
    elif method_id in {"ekell_style_controlled_shared_llm", "ekell_style_paper_fidelity"}:
        evidence = collect_ekell_runtime_evidence(method_id=method_id, config=config, runtime=runtime)
    else:
        evidence = collect_llm_evidence(method_id=method_id, config=config, llm=llm)
        evidence.runtime_prepared = runtime is not None
    if llm is not None:
        llm_part = collect_llm_evidence(method_id=method_id, config=config, llm=llm)
        evidence.llm_provider = llm_part.llm_provider
        evidence.llm_model = llm_part.llm_model
        evidence.llm_model_version = llm_part.llm_model_version
        evidence.llm_is_smoke = llm_part.llm_is_smoke
        evidence.llm_initialized = True
    return evidence


def method_formal_compliance(
    evidence: RuntimeEvidence,
    *,
    method_id: str,
    coverage_ok: bool,
    parsing_failures: int,
    schema_failures: int,
    taxonomy_valid: bool,
) -> dict[str, Any]:
    needs_index = method_id in {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"}
    real_index: bool | None
    if not needs_index:
        real_index = None
    else:
        real_index = bool(
            evidence.index_loaded
            and not evidence.index_built_during_run
            and evidence.actual_embedding_used is not False
            and evidence.smoke_fallback_used is not True
        )
    no_runtime_build = not evidence.index_built_during_run if needs_index else None
    formal_result = bool(
        coverage_ok
        and parsing_failures == 0
        and schema_failures == 0
        and taxonomy_valid
        and evidence.llm_is_smoke is False
        and (real_index is not False if needs_index else True)
        and (no_runtime_build is not False if needs_index else True)
    )
    return {
        "real_llm": evidence.llm_is_smoke is False,
        "real_index": real_index,
        "no_runtime_index_build": no_runtime_build,
        "complete_case_coverage": coverage_ok,
        "formal_result": formal_result,
    }


def compute_suite_formal_compliance(
    *,
    formal: bool,
    experiment_manifest_provided: bool,
    limit_used: bool,
    preflight_ok: bool,
    coverage_ok: bool,
    method_evidences: dict[str, RuntimeEvidence],
    method_compliance: dict[str, dict[str, Any]],
    dev_aliases_enabled: bool,
) -> dict[str, Any]:
    llm_methods = [e for e in method_evidences.values() if e.llm_is_smoke is not None]
    dense_ev = method_evidences.get("dense_rag")
    ekell_ev = method_evidences.get("ekell_style_controlled_shared_llm")
    real_llm = all(e.llm_is_smoke is False for e in llm_methods) if llm_methods else False
    real_dense = bool(
        dense_ev
        and dense_ev.index_loaded
        and not dense_ev.index_built_during_run
        and dense_ev.actual_embedding_used is not False
        and dense_ev.smoke_fallback_used is not True
    ) if formal else False
    real_ekell = bool(
        ekell_ev
        and ekell_ev.index_loaded
        and not ekell_ev.index_built_during_run
        and ekell_ev.actual_embedding_used is not False
        and ekell_ev.smoke_fallback_used is not True
    ) if formal else False
    index_methods = {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"}
    no_runtime_build = all(
        not e.index_built_during_run
        for e in method_evidences.values()
        if e.method_id in index_methods
    ) if formal else False
    all_method_formal = all(v.get("formal_result") for v in method_compliance.values()) if method_compliance else False
    formal_result = bool(
        formal
        and preflight_ok
        and not limit_used
        and coverage_ok
        and all_method_formal
        and real_llm
        and (not formal or (real_dense and real_ekell and no_runtime_build))
    )
    if not formal:
        formal_result = False
    return {
        "real_manifest": bool(formal and experiment_manifest_provided),
        "real_llm": real_llm if formal else False,
        "real_dense_index": real_dense if formal else False,
        "real_ekell_index": real_ekell if formal else False,
        "formal_aliases_only": bool(formal and not dev_aliases_enabled),
        "canonical_ids_only": True,
        "explicit_required_fields": formal,
        "complete_case_coverage": coverage_ok if formal else False,
        "no_runtime_index_build": no_runtime_build if formal else False,
        "preflight_ok": preflight_ok,
        "limit_used": limit_used,
        "formal_result": formal_result,
    }


def evidence_to_summary_sections(evidence: RuntimeEvidence) -> dict[str, Any]:
    llm = {
        "provider": evidence.llm_provider,
        "model": evidence.llm_model,
        "model_version": evidence.llm_model_version,
        "is_smoke": evidence.llm_is_smoke,
        "initialized": evidence.llm_initialized,
    }
    embedding: dict[str, Any] | None = None
    if evidence.embedding_backend or evidence.embedding_model:
        embedding = {
            "backend": evidence.embedding_backend,
            "model": evidence.embedding_model,
            "model_version": evidence.embedding_model_version,
            "actual_embedding_used": evidence.actual_embedding_used,
            "smoke_fallback_used": evidence.smoke_fallback_used,
        }
    index: dict[str, Any] | None = None
    if evidence.index_type or evidence.index_path:
        index = {
            "index_type": evidence.index_type,
            "index_path": evidence.index_path,
            "index_checksum": evidence.index_checksum,
            "index_manifest_sha256": evidence.index_manifest_sha256,
            "index_loaded": evidence.index_loaded,
            "index_built_during_run": evidence.index_built_during_run,
            "document_count": evidence.index_document_count,
        }
    runtime = {
        "runtime_prepared": evidence.runtime_prepared,
        "llm_initialized": evidence.llm_initialized,
        "lexical_ready": evidence.lexical_ready,
        "dense_dependency_index_checksum": evidence.dense_dependency_index_checksum,
        "dense_dependency_backend": evidence.dense_dependency_backend,
        "dense_dependency_smoke_fallback_used": evidence.dense_dependency_smoke_fallback_used,
        "rrf_k": evidence.rrf_k,
        "candidate_pool": evidence.candidate_pool,
    }
    return {"llm": llm, "embedding": embedding, "index": index, "runtime": runtime}
