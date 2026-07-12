"""Unified resource preflight for the five-method decision comparison suite."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from external_baselines.common.bundle_integrity import (
    extract_frozen_runner_bundle_identity,
    validate_formal_runner_bundle_integrity,
)
from external_baselines.common.checksums import sha256_file
from external_baselines.common.formal_config_validator import (
    FormalConfigError,
    validate_method_config,
)
from external_baselines.common.generation_identity import (
    detect_method_llm_overrides,
    validate_shared_generation_identity,
)
from external_baselines.common.io import read_json, read_jsonl, read_yaml
from external_baselines.interop.bundle import load_runner_bundle, validate_bundle_checksum
from external_baselines.method_registry import canonicalize_method_id

EKELL_REQUIRED_PROMPTS = (
    "stepwise_projection.txt",
    "stepwise_intersection.txt",
    "stepwise_union.txt",
    "stepwise_negation.txt",
    "final_kg_grounded_response.txt",
)

EKELL_LOGICAL_COMPONENTS = (
    ("parse_query", "external_baselines.ekell_style.logical_query.parser", "parse_query"),
    ("validate_query", "external_baselines.ekell_style.logical_query.validator", "validate_query"),
    ("execute_query", "external_baselines.ekell_style.logical_query.fol_executor", "execute_query"),
    ("run_stepwise_prompt_chain", "external_baselines.ekell_style.stepwise_prompt_chain", "run_stepwise_prompt_chain"),
    ("expand_neighborhood", "external_baselines.ekell_style.neighborhood_expander", "expand_neighborhood"),
    ("load_kg", "external_baselines.ekell_style.kg_loader", "load_kg"),
)


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


def _load_freeze_manifest(experiment_manifest: Path | None) -> dict[str, Any] | None:
    if experiment_manifest is None or not experiment_manifest.is_file():
        return None
    from external_baselines.common.experiment_manifest import load_experiment_manifest

    manifest = load_experiment_manifest(experiment_manifest)
    freeze_path = manifest.get("freeze_manifest")
    if not freeze_path:
        return None
    freeze_file = Path(str(freeze_path))
    if not freeze_file.is_file():
        root = Path(__file__).resolve().parents[3]
        freeze_file = root / str(freeze_path)
    if not freeze_file.is_file():
        return None
    return read_json(freeze_file)


def _load_shared_model_config(experiment_manifest: Path | None) -> dict[str, Any]:
    if experiment_manifest is None or not experiment_manifest.is_file():
        return {}
    from external_baselines.common.experiment_manifest import load_experiment_manifest

    manifest = load_experiment_manifest(experiment_manifest)
    shared_path = manifest.get("shared_model_config")
    if not shared_path:
        return {}
    shared_file = Path(str(shared_path))
    if not shared_file.is_file():
        shared_file = Path(__file__).resolve().parents[3] / str(shared_path)
    if not shared_file.is_file():
        return {}
    return read_yaml(shared_file)


def _validate_runner_bundle_integrity(
    bundle: dict[str, Any],
    *,
    formal: bool,
    freeze: dict[str, Any] | None,
) -> dict[str, Any]:
    if not formal:
        validation = validate_bundle_checksum(bundle)
        file_report = bundle.get("file_checksum_report") or {}
        file_ok = file_report.get("ok") if file_report.get("checked") else None
        return {
            "ok": validation.get("ok") is True,
            "bundle_checksum_ok": validation.get("ok") is True,
            "input_cases_integrity": True,
            "prediction_schema_integrity": True,
            "corpus_integrity": True,
            "per_file_checksums_checked": bool(file_report.get("checked")),
            "per_file_checksums_ok": file_ok,
            "producer_declared_checksum": validation.get("producer_declared_checksum"),
            "consumer_computed_hash": validation.get("consumer_computed_bundle_hash"),
            "expected_frozen_checksum": None,
            "input_cases_sha256": sha256_file(bundle.get("scenarios_path"))
            if bundle.get("scenarios_path")
            else None,
            "expected_input_cases_sha256": None,
            "prediction_schema_sha256": bundle.get("prediction_schema_sha256"),
            "expected_prediction_schema_sha256": None,
            "corpus_aggregate_sha256": (
                (bundle.get("corpus_manifest") or {}).get("aggregate_sha256")
                if isinstance(bundle.get("corpus_manifest"), dict)
                else None
            ),
            "expected_corpus_aggregate_sha256": None,
            "file_checksum_report_ok": file_ok,
            "mismatches": [],
            "errors": [],
        }

    frozen_identity = extract_frozen_runner_bundle_identity(freeze or {}, formal=formal)
    live = validate_formal_runner_bundle_integrity(bundle, frozen_identity=frozen_identity)
    if not live.get("input_cases_sha256") and bundle.get("scenarios_path"):
        live["input_cases_sha256"] = sha256_file(bundle.get("scenarios_path"))
    return live


def _validate_ekell_prompts(
    prompt_dir: Path,
    *,
    freeze: dict[str, Any] | None,
    formal: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    prompt_hashes: dict[str, str] = {}
    for name in EKELL_REQUIRED_PROMPTS:
        path = prompt_dir / name
        if not path.is_file():
            errors.append(f"ekell_prompt_missing:{name}")
            continue
        if path.stat().st_size == 0:
            errors.append(f"ekell_prompt_empty:{name}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"ekell_prompt_not_utf8:{name}")
            continue
        if not text.strip():
            errors.append(f"ekell_prompt_empty:{name}")
            continue
        prompt_hashes[name] = sha256_file(path)

    frozen_prompt_hashes = {}
    if isinstance(freeze, dict):
        frozen_prompt_hashes = dict(freeze.get("ekell_prompt_hashes") or {})
        prompt_tree = freeze.get("prompt_tree_sha256")
        if formal and prompt_tree and prompt_hashes:
            from external_baselines.common.freeze_manifest import prompt_tree_checksum

            actual_tree = prompt_tree_checksum(prompt_dir)
            if actual_tree and str(actual_tree) != str(prompt_tree):
                errors.append("ekell_prompt_tree_hash_mismatch")
    if formal and frozen_prompt_hashes:
        for name, expected in frozen_prompt_hashes.items():
            actual = prompt_hashes.get(name)
            if actual and str(actual) != str(expected):
                errors.append(f"ekell_prompt_hash_mismatch:{name}")

    return {"ok": not errors, "prompt_dir": str(prompt_dir), "prompt_hashes": prompt_hashes, "errors": errors}


def _validate_ekell_logical_components() -> list[str]:
    errors: list[str] = []
    for name, module_path, attr in EKELL_LOGICAL_COMPONENTS:
        try:
            module = importlib.import_module(module_path)
            component = getattr(module, attr)
            if not callable(component):
                errors.append(f"ekell_component_not_callable:{name}")
        except ImportError:
            errors.append(f"ekell_component_import_failed:{name}")
    return errors


def _validate_kg_jsonl(corpus_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in ("entities.jsonl", "relations.jsonl", "triples.jsonl"):
        path = corpus_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"ekell_{name}_missing_or_empty")
            continue
        try:
            rows = read_jsonl(path)
        except ValueError:
            errors.append(f"ekell_kg_jsonl_parse_error:{name}")
            continue
        if not rows:
            errors.append(f"ekell_{name}_missing_or_empty")
    triples_path = corpus_dir / "triples.jsonl"
    if triples_path.is_file():
        try:
            triples = read_jsonl(triples_path)
            for row in triples:
                if not isinstance(row, dict):
                    errors.append("ekell_triples_invalid_record")
                    break
                if not any(row.get(k) for k in ("head", "relation", "tail", "triple_id")):
                    errors.append("ekell_triples_missing_fields")
                    break
        except ValueError:
            errors.append("ekell_kg_jsonl_parse_error:triples.jsonl")
    return errors


def _preflight_llm_identity(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm") or {}
    return {
        "provider": llm.get("provider"),
        "model": llm.get("model"),
        "model_version": llm.get("model_version") or llm.get("version"),
        "temperature": llm.get("temperature"),
        "top_p": llm.get("top_p"),
        "max_tokens": llm.get("max_tokens"),
        "seed": llm.get("seed"),
        "enable_thinking": llm.get("enable_thinking"),
    }


def _preflight_method_resources(
    method_id: str,
    config: dict[str, Any],
    *,
    execution_stage: str,
    dense_config: dict[str, Any] | None = None,
    freeze: dict[str, Any] | None = None,
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
                require_explicit_embedding_evidence=formal,
            )
            result.index_identity = dict(payload.get("manifest") or payload)
            result.embedding_identity = {
                "backend": payload.get("backend"),
                "model_name": payload.get("model_name"),
                "model_version": payload.get("model_version"),
                "actual_embedding_used": (payload.get("manifest") or {}).get("actual_embedding_used"),
                "smoke_fallback_used": (payload.get("manifest") or {}).get("smoke_fallback_used"),
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
                require_explicit_embedding_evidence=formal,
            )
            result.index_identity = dict(payload.get("manifest") or payload)
            result.embedding_identity = {
                "backend": payload.get("backend"),
                "model_name": payload.get("model_name"),
                "model_version": payload.get("model_version"),
                "actual_embedding_used": (payload.get("manifest") or {}).get("actual_embedding_used"),
                "smoke_fallback_used": (payload.get("manifest") or {}).get("smoke_fallback_used"),
            }
        except DenseIndexError as exc:
            result.errors.append(str(exc))
            return result
    elif method_id == "ekell_style_controlled_shared_llm":
        result.errors.extend(_validate_kg_jsonl(corpus_dir))
        if result.errors:
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
                "actual_embedding_used": manifest.get("actual_embedding_used"),
                "smoke_fallback_used": manifest.get("smoke_fallback_used"),
            }
        except VectorIndexError as exc:
            result.errors.append(str(exc))
            return result
        prompt_dir_raw = (config.get("ekell_style") or {}).get("prompt_dir") or "configs/prompts/controlled"
        prompt_dir = Path(prompt_dir_raw)
        if not prompt_dir.is_dir():
            prompt_dir = Path(__file__).resolve().parents[3] / str(prompt_dir_raw)
        if not prompt_dir.is_dir():
            result.errors.append("ekell_prompt_dir_missing")
            return result
        prompt_report = _validate_ekell_prompts(prompt_dir, freeze=freeze, formal=formal)
        result.errors.extend(prompt_report.get("errors") or [])
        result.warnings.append(json.dumps({"ekell_prompt_hashes": prompt_report.get("prompt_hashes") or {}}))
        result.errors.extend(_validate_ekell_logical_components())

    result.resources_valid = not result.errors
    return result


def preflight_decision_suite(
    *,
    method_ids: list[str],
    method_configs: dict[str, dict[str, Any]],
    runner_bundle: Path,
    execution_stage: str,
    experiment_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate all method resources before any LLM client initialization."""
    formal = execution_stage == "formal"
    bundle = load_runner_bundle(runner_bundle)
    freeze = _load_freeze_manifest(experiment_manifest) if formal else None
    scenarios_path = bundle.get("scenarios_path")
    schema_path = bundle.get("prediction_schema_path")
    corpus_dir = bundle.get("corpus_dir")
    shared_errors: list[str] = []

    bundle_integrity = _validate_runner_bundle_integrity(bundle, formal=formal, freeze=freeze)
    if formal and not bundle_integrity.get("ok"):
        shared_errors.extend(bundle_integrity.get("errors") or ["runner_bundle_integrity_failed"])

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

    shared_model_cfg = _load_shared_model_config(experiment_manifest)
    generation_identity = validate_shared_generation_identity(
        method_ids=method_ids,
        method_configs=method_configs,
    )
    if formal:
        for method_id in method_ids:
            overrides = detect_method_llm_overrides(
                shared_config=shared_model_cfg,
                method_config=method_configs.get(method_id) or {},
            )
            for item in overrides:
                generation_identity["mismatches"].append({**item, "method_id": method_id})
        generation_identity["ok"] = not generation_identity.get("mismatches")

    dense_cfg = method_configs.get("dense_rag")
    method_reports: dict[str, Any] = {}
    all_ok = not shared_errors and bool(bundle_integrity.get("ok", True))
    ekell_prompt_bundle_valid = True

    for method_id in method_ids:
        mid = canonicalize_method_id(method_id)
        cfg = method_configs.get(mid) or method_configs.get(method_id) or {}
        report = _preflight_method_resources(
            mid,
            cfg,
            execution_stage=execution_stage,
            dense_config=dense_cfg if mid == "hybrid_rag" else None,
            freeze=freeze,
        )
        if shared_errors:
            report.errors.extend(shared_errors)
            report.resources_valid = False
        if mid == "ekell_style_controlled_shared_llm":
            prompt_errors = [e for e in report.errors if e.startswith("ekell_prompt")]
            if prompt_errors:
                ekell_prompt_bundle_valid = False
        method_reports[mid] = report.to_dict()
        if not report.ok:
            all_ok = False

    if formal:
        ekell_prompt_bundle_valid = ekell_prompt_bundle_valid and not any(
            e.startswith("ekell_prompt") for e in shared_errors
        )

    return {
        "ok": all_ok and generation_identity.get("ok", True),
        "execution_stage": execution_stage,
        "runner_bundle": str(runner_bundle),
        "shared_errors": shared_errors,
        "runner_bundle_integrity": bundle_integrity,
        "shared_generation_identity": generation_identity,
        "ekell_prompt_bundle_valid": ekell_prompt_bundle_valid if formal else False,
        "methods": method_reports,
    }
