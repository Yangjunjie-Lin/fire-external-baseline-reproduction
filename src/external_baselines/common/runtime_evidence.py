"""Runtime evidence collection for formal compliance reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from external_baselines.common.checksums import sha256_file
from external_baselines.common.generation_identity import (
    GenerationIdentity,
    extract_generation_identity,
    validate_runtime_generation_identity,
)

SMOKE_LLM_PROVIDERS = frozenset({"heuristic", "local", "smoke", "mock", "fake", ""})
SMOKE_MODEL_TOKENS = ("heuristic", "smoke", "fixture", "mock", "fake")


def _strict_real_index_evidence(
    evidence: "RuntimeEvidence",
) -> bool:
    return (
        evidence.index_loaded is True
        and evidence.index_built_during_run is False
        and evidence.actual_embedding_used is True
        and evidence.smoke_fallback_used is False
    )


def _manifest_file_sha(index_dir: str | Path | None) -> str | None:
    if not index_dir:
        return None
    manifest_path = Path(index_dir) / "index_manifest.json"
    if not manifest_path.is_file():
        return None
    return sha256_file(manifest_path)


@dataclass
class RuntimeEvidence:
    method_id: str

    llm_provider: str | None = None
    llm_model: str | None = None
    llm_model_version: str | None = None
    llm_temperature: float | None = None
    llm_top_p: float | None = None
    llm_max_tokens: int | None = None
    llm_seed: int | None = None
    llm_enable_thinking: bool | None = None
    llm_is_smoke: bool | None = None
    llm_initialized: bool = False

    embedding_backend: str | None = None
    embedding_model: str | None = None
    embedding_model_version: str | None = None
    actual_embedding_used: bool | None = None
    smoke_fallback_used: bool | None = None

    index_type: str | None = None
    configured_index_path: str | None = None
    resolved_index_path: str | None = None
    index_path: str | None = None
    index_checksum: str | None = None
    index_manifest_sha256: str | None = None
    index_loaded: bool = False
    index_built_during_run: bool = False
    index_document_count: int | None = None

    lexical_ready: bool | None = None
    dense_dependency_index_checksum: str | None = None
    dense_dependency_backend: str | None = None
    dense_dependency_actual_embedding_used: bool | None = None
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


def _resolve_index_paths(
    *,
    configured: str | Path | None,
    resolved: str | Path | None,
) -> tuple[str | None, str | None]:
    configured_text = str(configured or "").strip() or None
    resolved_text = str(resolved or configured or "").strip() or None
    return configured_text, resolved_text


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
    seed_raw = summary.get("seed")
    seed = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None
    enable_thinking = summary.get("enable_thinking")
    evidence = RuntimeEvidence(
        method_id=method_id,
        llm_provider=provider,
        llm_model=model,
        llm_model_version=str(summary.get("model_version") or ""),
        llm_temperature=float(summary.get("temperature") if summary.get("temperature") is not None else 0.0),
        llm_top_p=float(summary.get("top_p") if summary.get("top_p") is not None else 1.0),
        llm_max_tokens=int(summary.get("max_tokens") if summary.get("max_tokens") is not None else 0),
        llm_seed=seed,
        llm_enable_thinking=bool(enable_thinking) if enable_thinking is not None else False,
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
    index = getattr(runtime, "dense_index", None)
    audit = getattr(runtime, "audit", None)
    configured_path = dense_cfg.get("index_path")
    resolved_path = (
        getattr(index, "index_dir", None)
        or manifest.get("index_dir")
        or configured_path
    )
    configured_text, resolved_text = _resolve_index_paths(
        configured=configured_path,
        resolved=resolved_path,
    )
    evidence.embedding_backend = str(manifest.get("backend") or dense_cfg.get("backend") or "")
    evidence.embedding_model = str(manifest.get("model_name") or dense_cfg.get("model_name") or "")
    evidence.embedding_model_version = str(
        manifest.get("model_version") or dense_cfg.get("model_version") or ""
    )
    if "actual_embedding_used" in manifest:
        evidence.actual_embedding_used = manifest["actual_embedding_used"] is True
    if "smoke_fallback_used" in manifest:
        evidence.smoke_fallback_used = manifest["smoke_fallback_used"] is True
    evidence.index_type = str(manifest.get("index_type") or "dense_evidence_index")
    evidence.configured_index_path = configured_text
    evidence.resolved_index_path = resolved_text
    evidence.index_path = resolved_text
    evidence.index_checksum = str(manifest.get("index_checksum") or manifest.get("checksum") or "")
    evidence.index_manifest_sha256 = _manifest_file_sha(resolved_text)
    evidence.index_loaded = bool(getattr(audit, "index_load_count", 0) or index is not None)
    evidence.index_built_during_run = bool(getattr(runtime, "index_built_during_run", False))
    evidence.index_document_count = int(manifest.get("document_count") or 0) or None
    if configured_text and resolved_text and Path(configured_text).resolve() != Path(resolved_text).resolve():
        evidence.errors.append("configured_resolved_index_path_mismatch")
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
    evidence.dense_dependency_actual_embedding_used = evidence.actual_embedding_used
    evidence.dense_dependency_smoke_fallback_used = evidence.smoke_fallback_used
    evidence.rrf_k = float(hybrid_cfg.get("rrf_k")) if hybrid_cfg.get("rrf_k") is not None else None
    evidence.candidate_pool = (
        int(hybrid_cfg.get("candidate_pool")) if hybrid_cfg.get("candidate_pool") is not None else None
    )
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
    index = getattr(runtime, "vector_index", None)
    audit = getattr(runtime, "audit", None)
    configured_path = vector_cfg.get("index_path")
    resolved_path = manifest.get("index_dir") or getattr(index, "index_dir", None) or configured_path
    configured_text, resolved_text = _resolve_index_paths(
        configured=configured_path,
        resolved=resolved_path,
    )
    evidence.embedding_backend = str(manifest.get("backend") or "")
    evidence.embedding_model = str(manifest.get("model_name") or manifest.get("embedding_model") or "")
    evidence.embedding_model_version = str(manifest.get("model_version") or "")
    if "actual_embedding_used" in manifest:
        evidence.actual_embedding_used = manifest["actual_embedding_used"] is True
    if "smoke_fallback_used" in manifest:
        evidence.smoke_fallback_used = manifest["smoke_fallback_used"] is True
    evidence.index_type = str(manifest.get("index_type") or "ekell_kg_vector_index")
    evidence.configured_index_path = configured_text
    evidence.resolved_index_path = resolved_text
    evidence.index_path = resolved_text
    evidence.index_checksum = str(manifest.get("index_checksum") or manifest.get("checksum") or "")
    evidence.index_manifest_sha256 = _manifest_file_sha(resolved_text)
    evidence.index_loaded = bool(getattr(audit, "index_load_count", 0) or index is not None)
    evidence.index_built_during_run = bool(getattr(runtime, "index_built_during_run", False))
    evidence.index_document_count = int(manifest.get("document_count") or 0) or None
    if configured_text and resolved_text and Path(configured_text).resolve() != Path(resolved_text).resolve():
        evidence.errors.append("configured_resolved_index_path_mismatch")
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
        evidence.llm_temperature = llm_part.llm_temperature
        evidence.llm_top_p = llm_part.llm_top_p
        evidence.llm_max_tokens = llm_part.llm_max_tokens
        evidence.llm_seed = llm_part.llm_seed
        evidence.llm_enable_thinking = llm_part.llm_enable_thinking
        evidence.llm_is_smoke = llm_part.llm_is_smoke
        evidence.llm_initialized = True
    return evidence


def method_formal_compliance(
    evidence: RuntimeEvidence,
    *,
    formal: bool,
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
        real_index = _strict_real_index_evidence(evidence)
    no_runtime_build = (evidence.index_built_during_run is False) if needs_index else None
    technical_checks_passed = bool(
        coverage_ok
        and parsing_failures == 0
        and schema_failures == 0
        and taxonomy_valid
        and evidence.llm_is_smoke is False
        and (real_index is True if needs_index else True)
        and (no_runtime_build is True if needs_index else True)
        and not evidence.errors
    )
    formal_result = bool(formal and technical_checks_passed)
    return {
        "real_llm": evidence.llm_is_smoke is False,
        "real_index": real_index,
        "no_runtime_index_build": no_runtime_build,
        "complete_case_coverage": coverage_ok,
        "technical_checks_passed": technical_checks_passed,
        "formal_result": formal_result,
        "reason": None if formal else "execution_stage_not_formal",
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
    runner_bundle_integrity_ok: bool = False,
    input_cases_integrity_ok: bool = False,
    prediction_schema_integrity_ok: bool = False,
    corpus_integrity_ok: bool = False,
    shared_generation_identity_match: bool = False,
    runtime_generation_identity_match: bool = False,
    hybrid_dense_identity_match: bool = False,
    ekell_prompt_bundle_valid: bool = False,
    transactional_publish_complete: bool = False,
    transactional_publish_committed: bool | None = None,
    transactional_cleanup_complete: bool = True,
    method_ids: list[str] | None = None,
    phase: Literal["pre_publish", "final"] = "pre_publish",
) -> dict[str, Any]:
    llm_methods = [e for e in method_evidences.values() if e.llm_is_smoke is not None]
    dense_ev = method_evidences.get("dense_rag")
    hybrid_ev = method_evidences.get("hybrid_rag")
    ekell_ev = method_evidences.get("ekell_style_controlled_shared_llm")
    real_llm = all(e.llm_is_smoke is False for e in llm_methods) if llm_methods else False
    real_dense = _strict_real_index_evidence(dense_ev) if formal and dense_ev else False
    real_ekell = _strict_real_index_evidence(ekell_ev) if formal and ekell_ev else False
    index_methods = {"dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"}
    no_runtime_build = all(
        e.index_built_during_run is False for e in method_evidences.values() if e.method_id in index_methods
    ) if formal else False
    all_method_formal = all(v.get("formal_result") for v in method_compliance.values()) if method_compliance else False

    if formal and method_ids:
        runtime_identity = validate_runtime_generation_identity(
            method_ids=method_ids,
            method_evidences=method_evidences,
        )
        runtime_generation_identity_match = bool(runtime_identity.get("ok"))

    if formal and dense_ev and hybrid_ev:
        hybrid_dense_identity_match = (
            dense_ev.index_checksum == hybrid_ev.dense_dependency_index_checksum
            and hybrid_ev.dense_dependency_actual_embedding_used is True
            and hybrid_ev.dense_dependency_smoke_fallback_used is False
        )

    pre_publish_compliance_passed = bool(
        formal
        and preflight_ok
        and runner_bundle_integrity_ok
        and input_cases_integrity_ok
        and prediction_schema_integrity_ok
        and corpus_integrity_ok
        and not limit_used
        and coverage_ok
        and all_method_formal
        and real_llm
        and shared_generation_identity_match
        and runtime_generation_identity_match
        and real_dense
        and real_ekell
        and hybrid_dense_identity_match
        and ekell_prompt_bundle_valid
        and no_runtime_build
    )
    if not formal:
        pre_publish_compliance_passed = False

    publish_committed = (
        transactional_publish_committed
        if transactional_publish_committed is not None
        else transactional_publish_complete
    )
    cleanup_complete = transactional_cleanup_complete if formal and phase == "final" else True

    formal_result = bool(
        pre_publish_compliance_passed and publish_committed
    ) if phase == "final" else False

    publish_complete = publish_committed if formal and phase == "final" else False

    return {
        "real_manifest": bool(formal and experiment_manifest_provided),
        "runner_bundle_integrity": runner_bundle_integrity_ok if formal else False,
        "input_cases_integrity": input_cases_integrity_ok if formal else False,
        "prediction_schema_integrity": prediction_schema_integrity_ok if formal else False,
        "corpus_integrity": corpus_integrity_ok if formal else False,
        "shared_generation_identity_match": shared_generation_identity_match if formal else False,
        "runtime_generation_identity_match": runtime_generation_identity_match if formal else False,
        "real_llm": real_llm if formal else False,
        "real_dense_index": real_dense if formal else False,
        "real_ekell_index": real_ekell if formal else False,
        "hybrid_dense_identity_match": hybrid_dense_identity_match if formal else False,
        "ekell_prompt_bundle_valid": ekell_prompt_bundle_valid if formal else False,
        "formal_aliases_only": bool(formal and not dev_aliases_enabled),
        "canonical_ids_only": True,
        "explicit_required_fields": formal,
        "complete_case_coverage": coverage_ok if formal else False,
        "no_runtime_index_build": no_runtime_build if formal else False,
        "pre_publish_compliance_passed": pre_publish_compliance_passed if formal else False,
        "transactional_publish_complete": publish_complete,
        "transactional_publish_committed": publish_committed if formal and phase == "final" else False,
        "transactional_cleanup_complete": cleanup_complete if formal and phase == "final" else True,
        "preflight_ok": preflight_ok,
        "limit_used": limit_used,
        "formal_result": formal_result,
    }


def evidence_to_summary_sections(evidence: RuntimeEvidence) -> dict[str, Any]:
    llm = {
        "provider": evidence.llm_provider,
        "model": evidence.llm_model,
        "model_version": evidence.llm_model_version,
        "temperature": evidence.llm_temperature,
        "top_p": evidence.llm_top_p,
        "max_tokens": evidence.llm_max_tokens,
        "seed": evidence.llm_seed,
        "enable_thinking": evidence.llm_enable_thinking,
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
            "configured_index_path": evidence.configured_index_path,
            "resolved_index_path": evidence.resolved_index_path,
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
        "dense_dependency_actual_embedding_used": evidence.dense_dependency_actual_embedding_used,
        "dense_dependency_smoke_fallback_used": evidence.dense_dependency_smoke_fallback_used,
        "rrf_k": evidence.rrf_k,
        "candidate_pool": evidence.candidate_pool,
    }
    return {"llm": llm, "embedding": embedding, "index": index, "runtime": runtime}


def extract_configured_generation_identity(config: dict[str, Any]) -> GenerationIdentity:
    return extract_generation_identity(config)
