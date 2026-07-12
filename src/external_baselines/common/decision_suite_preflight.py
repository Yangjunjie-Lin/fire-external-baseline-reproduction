"""Unified resource preflight for the five-method decision comparison suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from external_baselines.common.formal_config_validator import (
    FormalConfigError,
    validate_method_config,
)
from external_baselines.interop.bundle import load_runner_bundle
from external_baselines.method_registry import canonicalize_method_id


@dataclass
class MethodPreflightResult:
    method_id: str
    config_valid: bool = False
    resources_valid: bool = False
    llm_identity: dict[str, Any] = field(default_factory=dict)
    embedding_identity: dict[str, Any] | None = None
    index_identity: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.config_valid and self.resources_valid and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "ok": self.ok,
            "config_valid": self.config_valid,
            "resources_valid": self.resources_valid,
            "llm_identity": dict(self.llm_identity),
            "embedding_identity": dict(self.embedding_identity) if self.embedding_identity else None,
            "index_identity": dict(self.index_identity) if self.index_identity else None,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _resolve_path(config: dict[str, Any], rel: str | Path) -> Path:
    candidate = Path(rel)
    if candidate.is_file() or candidate.is_dir():
        return candidate
    root = Path(__file__).resolve().parents[3]
    return root / rel


def _preflight_llm_identity(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm") or {}
    return {
        "provider": llm.get("provider"),
        "model": llm.get("model"),
        "model_version": llm.get("model_version") or llm.get("version"),
        "api_key_env": llm.get("api_key_env"),
        "temperature": llm.get("temperature"),
        "top_p": llm.get("top_p"),
        "max_tokens": llm.get("max_tokens"),
        "seed": llm.get("seed"),
    }


def _preflight_method_resources(
    method_id: str,
    config: dict[str, Any],
    *,
    execution_stage: str,
    dense_config: dict[str, Any] | None = None,
) -> MethodPreflightResult:
    result = MethodPreflightResult(method_id=method_id)
    result.llm_identity = _preflight_llm_identity(config)
    formal = execution_stage == "formal"
    try:
        validate_method_config(
            config,
            method_id=method_id,
            allow_placeholders=not formal,
            require_formal=formal,
            validation_stage=execution_stage,
            dense_config=dense_config,
        )
        result.config_valid = True
    except FormalConfigError as exc:
        result.errors.append(str(exc))
        return result

    corpus_dir = Path((config.get("paths") or {}).get("corpus_dir") or "data/corpus")
    if method_id == "bm25_rag":
        evidence = corpus_dir / "evidence_chunks.jsonl"
        if not evidence.is_file() or evidence.stat().st_size == 0:
            result.errors.append("bm25_evidence_chunks_missing_or_empty")
            return result
    elif method_id == "dense_rag":
        dense = config.get("dense_rag") or {}
        index_path = _resolve_path(config, str(dense.get("index_path") or ""))
        from external_baselines.retrieval.dense_index import DenseIndexError, validate_dense_index_directory

        try:
            payload = validate_dense_index_directory(
                index_path,
                load_embeddings=formal,
                expected_model_name=str(dense.get("model_name") or ""),
                expected_model_version=str(dense.get("model_version") or ""),
                expected_backend=str(dense.get("backend") or ""),
                expected_dimension=int(dense.get("dimension") or dense.get("dim") or 0) or None,
            )
            result.index_identity = dict(payload.get("manifest") or payload)
            result.embedding_identity = {
                "backend": payload.get("backend"),
                "model_name": payload.get("model_name"),
                "model_version": payload.get("model_version"),
            }
        except DenseIndexError as exc:
            result.errors.append(str(exc))
            return result
    elif method_id == "hybrid_rag":
        evidence = corpus_dir / "evidence_chunks.jsonl"
        if not evidence.is_file() or evidence.stat().st_size == 0:
            result.errors.append("hybrid_bm25_evidence_missing_or_empty")
            return result
        dense_cfg = dict(config.get("dense_rag") or {})
        hybrid = config.get("hybrid_rag") or {}
        index_path = _resolve_path(
            config,
            str(dense_cfg.get("index_path") or hybrid.get("dense_index_path") or ""),
        )
        from external_baselines.retrieval.dense_index import DenseIndexError, validate_dense_index_directory

        try:
            payload = validate_dense_index_directory(
                index_path,
                load_embeddings=formal,
                expected_model_name=str(dense_cfg.get("model_name") or hybrid.get("dense_model_name") or ""),
                expected_model_version=str(
                    dense_cfg.get("model_version") or hybrid.get("dense_model_version") or ""
                ),
                expected_backend=str(dense_cfg.get("backend") or hybrid.get("dense_method") or ""),
                expected_dimension=int(dense_cfg.get("dimension") or hybrid.get("dimension") or 0) or None,
            )
            result.index_identity = dict(payload.get("manifest") or payload)
            result.embedding_identity = {
                "backend": payload.get("backend"),
                "model_name": payload.get("model_name"),
                "model_version": payload.get("model_version"),
            }
        except DenseIndexError as exc:
            result.errors.append(str(exc))
            return result
    elif method_id == "ekell_style_controlled_shared_llm":
        for name in ("entities.jsonl", "relations.jsonl", "triples.jsonl"):
            path = corpus_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                result.errors.append(f"ekell_{name}_missing_or_empty")
                return result
        vector = config.get("ekell_vector") or {}
        index_path = _resolve_path(config, str(vector.get("index_path") or ""))
        from external_baselines.ekell_style.vector_index import VectorIndex, VectorIndexError

        try:
            manifest = VectorIndex.validate_directory(
                index_path,
                load_embeddings=formal,
                expected_backend=str(vector.get("backend") or ""),
                expected_model_name=str(vector.get("model_name") or ""),
                expected_model_version=str(vector.get("model_version") or ""),
                expected_dimension=int(vector.get("dimension") or vector.get("dim") or 0) or None,
                require_real_embedding=formal,
            )
            result.index_identity = dict(manifest)
            result.embedding_identity = {
                "backend": manifest.get("backend"),
                "model_name": manifest.get("model_name"),
                "model_version": manifest.get("model_version"),
            }
        except VectorIndexError as exc:
            result.errors.append(str(exc))
            return result
        prompt_dir = Path((config.get("ekell_style") or {}).get("prompt_dir") or "configs/prompts/controlled")
        if not prompt_dir.is_dir():
            result.errors.append("ekell_prompt_dir_missing")
            return result

    result.resources_valid = True
    return result


def preflight_decision_suite(
    *,
    method_ids: list[str],
    method_configs: dict[str, dict[str, Any]],
    runner_bundle: Path,
    execution_stage: str,
) -> dict[str, Any]:
    """Validate all method resources before any LLM client initialization."""
    formal = execution_stage == "formal"
    bundle = load_runner_bundle(runner_bundle)
    scenarios_path = bundle.get("scenarios_path")
    schema_path = bundle.get("prediction_schema_path")
    corpus_dir = bundle.get("corpus_dir")
    shared_errors: list[str] = []
    if not scenarios_path or not Path(scenarios_path).is_file():
        shared_errors.append("runner_bundle_input_cases_missing")
    if not schema_path or not Path(schema_path).is_file():
        shared_errors.append("runner_bundle_prediction_schema_missing")
    if not corpus_dir or not Path(corpus_dir).is_dir():
        shared_errors.append("runner_bundle_corpus_dir_missing")

    if formal:
        from external_baselines.common.firebench_taxonomy import validate_formal_alias_table

        try:
            validate_formal_alias_table()
        except Exception as exc:  # noqa: BLE001
            shared_errors.append(f"formal_alias_validation_failed:{exc}")

    dense_cfg = method_configs.get("dense_rag")
    method_reports: dict[str, Any] = {}
    all_ok = not shared_errors
    for method_id in method_ids:
        mid = canonicalize_method_id(method_id)
        cfg = method_configs.get(mid) or method_configs.get(method_id) or {}
        report = _preflight_method_resources(
            mid,
            cfg,
            execution_stage=execution_stage,
            dense_config=dense_cfg if mid == "hybrid_rag" else None,
        )
        if shared_errors:
            report.errors.extend(shared_errors)
            report.resources_valid = False
        method_reports[mid] = report.to_dict()
        if not report.ok:
            all_ok = False

    return {
        "ok": all_ok,
        "execution_stage": execution_stage,
        "runner_bundle": str(runner_bundle),
        "shared_errors": shared_errors,
        "methods": method_reports,
    }
