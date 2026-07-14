"""Freeze manifest create/validate helpers for formal comparison runs."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from external_baselines.common.checksums import sha256_file
from external_baselines.common.experiment_manifest import (
    build_method_config,
    experiment_core_sha256,
    get_method_entry,
    load_experiment_manifest,
    merged_method_config_sha256,
)
from external_baselines.common.formal_config_validator import FormalConfigError, _is_placeholder
from external_baselines.common.io import read_json
from external_baselines.common.path_resolution import (
    PathContext,
    ResolvedPathReference,
    resolve_declared_path,
    resolve_path_reference,
)
from external_baselines.common.strict_config_types import require_exact_nonempty_string
from external_baselines.ekell_style.prompt_identity import (
    EKELL_REQUIRED_PROMPTS,
    validate_and_hash_prompt_bundle,
)
from external_baselines.method_registry import comparison_suite_methods

COMPARISON_METHOD_IDS = comparison_suite_methods()
SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")

REQUIRED_COMPLETE_FIELDS = (
    "selected_dev_run_evidence",
    "experiment_core_sha256",
    "base_config_sha256",
    "shared_model_config_sha256",
    "method_config_sha256",
    "merged_method_config_sha256",
    "prompt_tree_sha256",
    "ekell_prompt_bundle",
    "runner_bundle",
    "llm",
    "embedding",
    "indexes",
    "path_provenance",
)

CRITICAL_PORTABLE_PROVENANCE = (
    "experiment_manifest",
    "base_config",
    "shared_model_config",
    "method_configs",
    "runner_bundle",
    "selected_dev_evidence",
    "prompt_dir",
    "dense_index",
    "ekell_index",
    "freeze_manifest",
)

RUNNER_BUNDLE_IDENTITY_FIELDS = (
    "consumer_computed_hash",
    "input_cases_sha256",
    "prediction_schema_sha256",
    "corpus_aggregate_sha256",
)

DENSE_COMPLETE_INDEX_FIELDS = (
    "index_type",
    "backend",
    "model_name",
    "model_version",
    "dimension",
    "normalize_embeddings",
    "document_count",
    "corpus_checksum",
    "documents_checksum",
    "documents_file_checksum",
    "embeddings_checksum",
    "evidence_source_checksum",
    "index_checksum",
    "index_manifest_sha256",
    "actual_embedding_used",
    "smoke_fallback_used",
)

HYBRID_COMPLETE_INDEX_FIELDS = (
    "index_checksum",
    "index_manifest_sha256",
)

EKELL_COMPLETE_INDEX_FIELDS = (
    "index_type",
    "backend",
    "model_name",
    "model_version",
    "dimension",
    "normalize_embeddings",
    "document_count",
    "kg_checksum",
    "corpus_checksum",
    "documents_checksum",
    "documents_file_checksum",
    "embeddings_checksum",
    "index_checksum",
    "index_manifest_sha256",
    "actual_embedding_used",
    "smoke_fallback_used",
)

INDEX_SHA256_FIELDS = frozenset(
    {
        "corpus_checksum",
        "documents_checksum",
        "documents_file_checksum",
        "embeddings_checksum",
        "evidence_source_checksum",
        "kg_checksum",
        "index_checksum",
        "index_manifest_sha256",
    }
)
INDEX_STRING_FIELDS = frozenset({"index_type", "backend", "model_name", "model_version"})
INDEX_POSITIVE_INT_FIELDS = frozenset({"dimension", "document_count"})
INDEX_BOOL_FIELDS = frozenset(
    {"normalize_embeddings", "actual_embedding_used", "smoke_fallback_used"}
)


def runner_bundle_block_from_freeze(freeze: dict[str, Any]) -> dict[str, Any]:
    block = freeze.get("runner_bundle")
    if isinstance(block, dict):
        return dict(block)
    return {
        "producer_declared_checksum": freeze.get("producer_declared_checksum"),
        "consumer_computed_hash": freeze.get("runner_bundle_checksum"),
        "input_cases_sha256": freeze.get("input_cases_sha256"),
        "prediction_schema_sha256": freeze.get("prediction_schema_checksum"),
        "corpus_aggregate_sha256": freeze.get("corpus_checksum"),
    }


def prompt_tree_checksum(prompt_dir: str | Path) -> str | None:
    from external_baselines.common.formal_config_validator import ROOT_REL

    try:
        root = resolve_declared_path(
            prompt_dir,
            context=PathContext(repository_root=ROOT_REL),
            policy="repository_relative",
            must_exist=False,
        )
    except (TypeError, ValueError):
        return None
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


def _resolve_path(path: str | Path) -> Path:
    from external_baselines.common.formal_config_validator import ROOT_REL

    return resolve_declared_path(
        path,
        context=PathContext(repository_root=ROOT_REL),
        policy="repository_relative",
        must_exist=False,
    )


def _raw_experiment_mapping(experiment: dict[str, Any]) -> dict[str, Any]:
    raw = experiment.get("raw")
    return raw if isinstance(raw, dict) else experiment


def _freeze_config_identities(
    *,
    experiment_manifest_path: str | Path,
    experiment: dict[str, Any],
    method_config_paths: dict[str, str] | None,
    repository_root: Path,
) -> tuple[str | None, str | None, dict[str, str | None], dict[str, str | None]]:
    """Compute file-level and canonical merged-config identities from one merge path."""
    try:
        resolved = load_experiment_manifest(
            experiment_manifest_path,
            repository_root=repository_root,
        )
    except (FileNotFoundError, ValueError):
        resolved = experiment if "base_config" in experiment else {}

    base = (
        resolved.get("base_config_resolved")
        or resolved.get("base_config")
        or _raw_experiment_mapping(experiment).get("base_config")
    )
    shared = resolved.get("shared_model_config_resolved") or resolved.get(
        "shared_model_config"
    ) or _raw_experiment_mapping(experiment).get("shared_model_config")
    base_sha = sha256_file(Path(str(base))) if base else None
    shared_sha = sha256_file(Path(str(shared))) if shared else None

    paths = dict(method_config_paths or {})
    if not paths:
        for entry in resolved.get("methods") or experiment.get("methods") or []:
            if isinstance(entry, dict) and entry.get("method_id") and entry.get("config"):
                paths[str(entry["method_id"])] = str(entry["config"])
    method_hashes: dict[str, str | None] = {mid: None for mid in COMPARISON_METHOD_IDS}
    for mid in COMPARISON_METHOD_IDS:
        rel = paths.get(mid)
        if rel:
            method_hashes[mid] = sha256_file(Path(rel))

    merged_hashes: dict[str, str | None] = {mid: None for mid in COMPARISON_METHOD_IDS}
    if resolved:
        for mid in COMPARISON_METHOD_IDS:
            try:
                entry = dict(get_method_entry(resolved, mid, require_enabled=False))
                if paths.get(mid):
                    entry["config"] = str(Path(paths[mid]))
                merged_hashes[mid] = merged_method_config_sha256(
                    build_method_config(resolved, entry)
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                merged_hashes[mid] = None
    return base_sha, shared_sha, method_hashes, merged_hashes


def _ekell_prompt_bundle_from_experiment(
    experiment: dict[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any] | None:
    try:
        entry = get_method_entry(
            experiment,
            "ekell_style_controlled_shared_llm",
            require_enabled=False,
        )
        merged = build_method_config(experiment, entry)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    ekell_style = merged.get("ekell_style") or {}
    if not isinstance(ekell_style, dict):
        raise FormalConfigError("ekell_style must be an object")
    prompt_dir = require_exact_nonempty_string(
        ekell_style.get("prompt_dir"),
        field="ekell_style.prompt_dir",
    )
    if _is_placeholder(prompt_dir):
        raise FormalConfigError("ekell_style.prompt_dir must not be a placeholder")
    return validate_and_hash_prompt_bundle(
        prompt_dir,
        path_context=PathContext(repository_root=repository_root),
    )


def _index_block(
    *,
    index_checksum: str | None = None,
    index_manifest_sha256: str | None = None,
    corpus_checksum: str | None = None,
    kg_checksum: str | None = None,
    model_version: str | None = None,
    include_kg: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "index_checksum": index_checksum,
        "index_manifest_sha256": index_manifest_sha256,
        "corpus_checksum": corpus_checksum,
        "model_version": model_version,
    }
    if include_kg:
        block["kg_checksum"] = kg_checksum
    block.update(extra)
    return block


def _provenance_entry_is_external(entry: dict[str, Any]) -> bool:
    return (
        entry.get("external") is True
        or entry.get("path_policy") in {"absolute_external", "absolute"}
    )


def _provenance_entry_is_portable(entry: dict[str, Any]) -> bool:
    if _provenance_entry_is_external(entry) or entry.get("portable") is False:
        return False
    if entry.get("path_policy") not in {
        "repository_relative",
        "experiment_relative",
        "bundle_relative",
    }:
        return False
    canonical = entry.get("canonical_path") or entry.get(
        "canonical_relative_path"
    )
    if type(canonical) is not str or not canonical:
        return False
    posix = PurePosixPath(canonical.replace("\\", "/"))
    windows = PureWindowsPath(canonical)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        return False
    return True


def _freeze_path_provenance_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Add machine-local freeze diagnostics without making them authoritative."""
    frozen = dict(entry)
    canonical = frozen.get("canonical_path") or frozen.get(
        "canonical_relative_path"
    )
    if canonical is not None:
        frozen["canonical_path"] = canonical
    resolved = frozen.get("resolved_path") or frozen.get("resolved_prompt_dir")
    if resolved is not None:
        frozen["resolved_path_at_freeze"] = str(resolved)
        frozen["resolved_path_authoritative"] = False
    frozen["portable"] = _provenance_entry_is_portable(frozen)
    return frozen


def assert_freeze_provenance_portable(
    provenance: dict[str, Any],
    *,
    require_complete_entries: bool = False,
) -> None:
    """Fail closed when any complete-freeze critical resource is external."""
    def _check(label: str, entry: Any) -> None:
        if not isinstance(entry, dict):
            return
        if not _provenance_entry_is_portable(entry):
            raise FormalConfigError(
                f"complete_freeze_external_resource_not_portable:{label}"
            )

    provenance = provenance or {}
    if require_complete_entries:
        for label in CRITICAL_PORTABLE_PROVENANCE:
            if label not in provenance or provenance[label] in (None, ""):
                raise FormalConfigError(
                    f"complete_freeze_path_provenance_missing:{label}"
                )
        method_entries = provenance.get("method_configs")
        if not isinstance(method_entries, dict):
            raise FormalConfigError(
                "complete_freeze_path_provenance_missing:method_configs"
            )
        for method_id in COMPARISON_METHOD_IDS:
            if not isinstance(method_entries.get(method_id), dict):
                raise FormalConfigError(
                    "complete_freeze_path_provenance_missing:"
                    f"method_configs.{method_id}"
                )

    for label, entry in provenance.items():
        if label == "method_configs" and isinstance(entry, dict):
            for method_id, method_entry in entry.items():
                _check(f"method_configs.{method_id}", method_entry)
        else:
            _check(label, entry)


def _freeze_experiment_path_provenance(
    experiment: dict[str, Any],
) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for label, entry in dict(experiment.get("path_provenance") or {}).items():
        if label == "method_configs" and isinstance(entry, dict):
            provenance[label] = {
                method_id: _freeze_path_provenance_entry(method_entry)
                for method_id, method_entry in entry.items()
                if isinstance(method_entry, dict)
            }
        elif isinstance(entry, dict):
            provenance[label] = _freeze_path_provenance_entry(entry)
        else:
            provenance[label] = entry
    return provenance


def build_freeze_manifest_payload(
    *,
    experiment_manifest_path: str | Path,
    experiment_raw: dict[str, Any],
    selected_dev_run: str | Path | ResolvedPathReference,
    producer_declared_checksum: str | None = None,
    consumer_computed_hash: str | None = None,
    input_cases_sha256: str | None = None,
    corpus_checksum: str | None = None,
    schema_checksum: str | None = None,
    method_config_paths: dict[str, str] | None = None,
    indexes: dict[str, Any] | None = None,
    embedding: dict[str, Any] | None = None,
    llm: dict[str, Any] | None = None,
    producer_checksum_available: bool | None = None,
    include_legacy_compat_fields: bool = False,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    experiment_manifest_path = Path(experiment_manifest_path)
    repository_root = Path(
        repository_root or Path(__file__).resolve().parents[3]
    ).resolve()
    if isinstance(selected_dev_run, ResolvedPathReference):
        selected_ref = selected_dev_run
    else:
        selected_ref = resolve_path_reference(
            selected_dev_run,
            context=PathContext(
                repository_root=repository_root,
                experiment_manifest_path=experiment_manifest_path,
            ),
            policy="repository_relative",
            must_exist=False,
            expected_kind="file",
        )
    try:
        resolved_experiment = load_experiment_manifest(
            experiment_manifest_path,
            repository_root=repository_root,
        )
    except (FileNotFoundError, ValueError):
        resolved_experiment = experiment_raw
    raw_experiment = _raw_experiment_mapping(resolved_experiment)
    shared = resolved_experiment.get("shared_model_config_resolved") or resolved_experiment.get(
        "shared_model_config"
    ) or raw_experiment.get("shared_model_config")
    base_sha, shared_sha, method_hashes, merged_hashes = _freeze_config_identities(
        experiment_manifest_path=experiment_manifest_path,
        experiment=experiment_raw,
        method_config_paths=method_config_paths,
        repository_root=repository_root,
    )
    prompt_bundle = _ekell_prompt_bundle_from_experiment(
        resolved_experiment,
        repository_root=repository_root,
    )

    llm_out = dict(llm or {})
    shared_path = Path(str(shared)) if shared else None
    if not llm_out and shared_path and shared_path.is_file():
        from external_baselines.common.io import read_yaml

        shared_cfg = read_yaml(shared_path)
        llm_block = shared_cfg.get("llm") or {}
        llm_out = {
            "provider": llm_block.get("provider"),
            "model": llm_block.get("model"),
            "model_version": llm_block.get("model_version") or llm_block.get("version"),
        }

    emb_out = {
        "backend": "text2vec",
        "model_name": "BAAI/bge-m3",
        "model_version": None,
        "dimension": 1024,
        "normalize_embeddings": True,
    }
    if embedding:
        emb_out.update({k: v for k, v in embedding.items() if v is not None})

    index_payload = {
        "dense": _index_block(),
        "hybrid_dense_dependency": {"index_checksum": None},
        "ekell": _index_block(include_kg=True),
    }
    if indexes:
        for key in ("dense", "hybrid_dense_dependency", "ekell"):
            if key in indexes and isinstance(indexes[key], dict):
                index_payload[key].update(indexes[key])

    producer_available = (
        producer_checksum_available
        if producer_checksum_available is not None
        else bool(producer_declared_checksum)
    )
    runner_bundle_block = {
        "producer_declared_checksum": producer_declared_checksum,
        "consumer_computed_hash": consumer_computed_hash,
        "producer_checksum_available": producer_available,
        "input_cases_sha256": input_cases_sha256,
        "prediction_schema_sha256": schema_checksum,
        "corpus_aggregate_sha256": corpus_checksum,
    }

    payload: dict[str, Any] = {
        "freeze_id": "controlled_comparison_v1",
        "freeze_status": "frozen",
        "selected_dev_run_evidence": {
            "declared_path": selected_ref.declared_path.replace("\\", "/"),
            "canonical_path": selected_ref.canonical_path,
            "path_policy": selected_ref.path_policy,
            "sha256": sha256_file(selected_ref.resolved_path),
        },
        "selected_dev_run_evidence_resolved_at_freeze": str(
            selected_ref.resolved_path
        ),
        "selected_dev_run_evidence_resolved_authoritative": False,
        "selection_criterion": (
            "Safety-Gated + Critical Failure Rate + Risk/Action F1 + evidence support + latency"
        ),
        "experiment_manifest_sha256_at_freeze_candidate": sha256_file(
            experiment_manifest_path
        ),
        "experiment_manifest_sha256_authoritative": False,
        "experiment_core_sha256": experiment_core_sha256(raw_experiment),
        "base_config_sha256": base_sha,
        "shared_model_config_sha256": shared_sha,
        "method_config_sha256": method_hashes,
        "merged_method_config_sha256": merged_hashes,
        "prompt_tree_sha256": (
            prompt_bundle.get("prompt_tree_sha256") if prompt_bundle else None
        ),
        "ekell_prompt_bundle": prompt_bundle,
        "path_provenance": {
            **_freeze_experiment_path_provenance(resolved_experiment),
            "selected_dev_evidence": {
                "declared_path": selected_ref.declared_path.replace("\\", "/"),
                "canonical_path": selected_ref.canonical_path,
                "path_policy": selected_ref.path_policy,
                "resolved_path_at_freeze": str(selected_ref.resolved_path),
                "resolved_path_authoritative": False,
                "external": selected_ref.external,
                "portable": not selected_ref.external,
            },
            "prompt_dir": (
                {
                    "declared_path": prompt_bundle.get("declared_prompt_dir"),
                    "canonical_path": prompt_bundle.get("canonical_prompt_dir"),
                    "path_policy": (
                        "absolute_external"
                        if prompt_bundle.get("path_policy") == "absolute"
                        else prompt_bundle.get("path_policy")
                    ),
                    "resolved_path_at_freeze": prompt_bundle.get(
                        "resolved_prompt_dir"
                    ),
                    "resolved_path_authoritative": False,
                    "external": prompt_bundle.get("external") is True,
                    "portable": prompt_bundle.get("external") is not True,
                }
                if prompt_bundle
                else None
            ),
        },
        "runner_bundle": runner_bundle_block,
        "llm": llm_out,
        "embedding": emb_out,
        "indexes": index_payload,
    }
    if include_legacy_compat_fields:
        payload.update(
            {
                "runner_bundle_checksum": consumer_computed_hash,
                "producer_declared_checksum": producer_declared_checksum,
                "input_cases_sha256": input_cases_sha256,
                "corpus_checksum": corpus_checksum,
                "prediction_schema_checksum": schema_checksum,
            }
        )
    return payload


def _require_nonempty_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FormalConfigError(f"{label} not found: {path}")
    if path.stat().st_size <= 0:
        raise FormalConfigError(f"{label} is empty: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise FormalConfigError(f"{label} is empty: {path}")


def _check_hash(
    freeze_value: Any,
    actual: str | None,
    *,
    label: str,
    require: bool,
) -> None:
    if freeze_value in (None, ""):
        if require:
            raise FormalConfigError(f"freeze_manifest missing {label}.")
        return
    if actual is None:
        if require:
            raise FormalConfigError(f"freeze_manifest {label} set but actual value unavailable.")
        return
    if str(freeze_value) != str(actual):
        raise FormalConfigError(f"freeze_manifest {label} mismatch.")


def _check_optional_pair(
    freeze_value: Any,
    expected: Any,
    *,
    label: str,
    require: bool = False,
) -> None:
    if expected is None and freeze_value in (None, ""):
        if require:
            raise FormalConfigError(f"freeze_manifest missing {label}.")
        return
    if expected is None:
        return
    if freeze_value in (None, ""):
        if require:
            raise FormalConfigError(f"freeze_manifest missing {label}.")
        return
    if str(freeze_value) != str(expected):
        raise FormalConfigError(f"freeze_manifest {label} mismatch.")


def _require_sha256_value(value: Any, *, label: str) -> str:
    if type(value) is not str or not SHA256_HEX_RE.fullmatch(value):
        raise FormalConfigError(f"freeze_manifest {label} must be a valid SHA-256.")
    return value


def _require_index_block(
    value: Any,
    *,
    label: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FormalConfigError(f"freeze_manifest missing indexes.{label}.")
    for field in fields:
        _require_sha256_value(value.get(field), label=f"indexes.{label}.{field}")
    return value


def _require_complete_index_block(
    value: Any,
    *,
    label: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FormalConfigError(f"freeze_manifest missing indexes.{label}.")
    for field in fields:
        field_label = f"indexes.{label}.{field}"
        raw = value.get(field)
        if field in INDEX_SHA256_FIELDS:
            _require_sha256_value(raw, label=field_label)
        elif field in INDEX_STRING_FIELDS:
            if type(raw) is not str or not raw:
                raise FormalConfigError(f"freeze_manifest {field_label} must be a non-empty string.")
        elif field in INDEX_POSITIVE_INT_FIELDS:
            if type(raw) is not int or raw <= 0:
                raise FormalConfigError(f"freeze_manifest {field_label} must be a positive integer.")
        elif field in INDEX_BOOL_FIELDS:
            if type(raw) is not bool:
                raise FormalConfigError(f"freeze_manifest {field_label} must be an exact boolean.")
        elif raw in (None, ""):
            raise FormalConfigError(f"freeze_manifest missing {field_label}.")
    if "actual_embedding_used" in fields and value["actual_embedding_used"] is not True:
        raise FormalConfigError(
            f"freeze_manifest indexes.{label}.actual_embedding_used must be true."
        )
    if "smoke_fallback_used" in fields and value["smoke_fallback_used"] is not False:
        raise FormalConfigError(
            f"freeze_manifest indexes.{label}.smoke_fallback_used must be false."
        )
    return value


def _compare_index_identity_fields(
    frozen: dict[str, Any],
    live: dict[str, Any],
    *,
    label: str,
    fields: tuple[str, ...],
    require_live: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for field in fields:
        if field not in frozen:
            continue
        field_label = f"indexes.{label}.{field}"
        if field not in live:
            if require_live:
                raise FormalConfigError(f"freeze_manifest loaded index missing {field_label}.")
            continue
        frozen_value = frozen.get(field)
        live_value = live.get(field)
        ok = frozen_value == live_value
        checks.append({"field": field_label, "ok": ok})
        if not ok:
            raise FormalConfigError(f"freeze_manifest {field_label} mismatch.")
    return checks


def _identity_match_report(checks: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        f"{str(check['field']).rsplit('.', 1)[-1]}_match": check["ok"] is True
        for check in checks
    }


def validate_freeze_manifest(
    freeze_path: str | Path,
    *,
    experiment_manifest_path: str | Path,
    experiment_raw: dict[str, Any],
    require_complete: bool = False,
    expected_runner_bundle_checksum: str | None = None,
    expected_corpus_checksum: str | None = None,
    expected_prediction_schema_checksum: str | None = None,
    expected_indexes: dict[str, Any] | None = None,
    loaded_index_manifests: dict[str, Any] | None = None,
    method_config_paths: dict[str, str] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    freeze = freeze_path if isinstance(freeze_path, dict) else read_json(freeze_path)
    if not isinstance(freeze, dict):
        raise FormalConfigError("freeze_manifest must be a JSON object.")
    if str(freeze.get("freeze_status") or "").lower() != "frozen":
        raise FormalConfigError("freeze_manifest.freeze_status must be frozen.")

    migration_warning: str | None = None
    if freeze.get("experiment_manifest_sha256") and not freeze.get(
        "experiment_core_sha256"
    ):
        raise FormalConfigError("legacy_freeze_requires_regeneration")
    if require_complete:
        runner_block_raw = freeze.get("runner_bundle")
        if not isinstance(runner_block_raw, dict):
            if any(
                freeze.get(key)
                for key in (
                    "runner_bundle_checksum",
                    "corpus_checksum",
                    "prediction_schema_checksum",
                )
            ):
                raise FormalConfigError("legacy_freeze_requires_regeneration")
            raise FormalConfigError("freeze_manifest missing runner_bundle block.")
        for field in REQUIRED_COMPLETE_FIELDS:
            if field not in freeze or freeze.get(field) in (None, ""):
                if field == "selected_dev_run_evidence":
                    raise FormalConfigError(
                        "legacy_selected_dev_evidence_identity_requires_regeneration"
                    )
                if field == "ekell_prompt_bundle" and freeze.get(
                    "prompt_tree_sha256"
                ):
                    raise FormalConfigError(
                        "legacy_freeze_prompt_identity_requires_regeneration"
                    )
                raise FormalConfigError(f"freeze_manifest incomplete: missing {field}.")
    elif any(
        freeze.get(key)
        for key in ("runner_bundle_checksum", "corpus_checksum", "prediction_schema_checksum")
    ) and not isinstance(freeze.get("runner_bundle"), dict):
        migration_warning = "legacy_freeze_requires_regeneration"

    evidence = freeze.get("selected_dev_run_evidence")
    if not evidence:
        if require_complete:
            raise FormalConfigError(
                "legacy_selected_dev_evidence_identity_requires_regeneration"
            )
        raise FormalConfigError("freeze_manifest requires selected_dev_run_evidence.")
    if isinstance(evidence, str):
        if require_complete:
            raise FormalConfigError(
                "legacy_selected_dev_evidence_identity_requires_regeneration"
            )
        migration_warning = (
            migration_warning
            or "legacy_selected_dev_evidence_identity_requires_regeneration"
        )
        evidence_path = _resolve_path(evidence)
        _require_nonempty_file(evidence_path, label="selected_dev_run_evidence")
    elif isinstance(evidence, dict):
        required_evidence_fields = {
            "declared_path",
            "canonical_path",
            "path_policy",
            "sha256",
        }
        if require_complete and not required_evidence_fields.issubset(evidence):
            raise FormalConfigError(
                "legacy_selected_dev_evidence_identity_requires_regeneration"
            )
        try:
            evidence_declared = require_exact_nonempty_string(
                evidence.get("declared_path"),
                field="selected_dev_run_evidence.declared_path",
            )
            evidence_canonical = require_exact_nonempty_string(
                evidence.get("canonical_path"),
                field="selected_dev_run_evidence.canonical_path",
            )
            evidence_policy = require_exact_nonempty_string(
                evidence.get("path_policy"),
                field="selected_dev_run_evidence.path_policy",
            )
        except ValueError as exc:
            raise FormalConfigError(str(exc)) from exc
        if require_complete:
            if evidence_policy == "absolute_external":
                raise FormalConfigError(
                    "selected_dev_evidence_external_path_not_portable"
                )
            if evidence_policy != "repository_relative":
                raise FormalConfigError(
                    "selected_dev_evidence_path_policy_must_be_repository_relative"
                )
        try:
            if evidence_policy == "absolute_external":
                evidence_ref = resolve_path_reference(
                    evidence_canonical,
                    context=PathContext(
                        repository_root=Path(
                            repository_root or Path(__file__).resolve().parents[3]
                        ),
                        experiment_manifest_path=Path(experiment_manifest_path),
                    ),
                    policy="absolute_external",
                    expected_kind="file",
                )
            elif evidence_policy in {"repository_relative", "experiment_relative"}:
                if Path(evidence_canonical).is_absolute():
                    raise FormalConfigError(
                        "selected_dev_evidence_canonical_path_must_be_relative"
                    )
                evidence_ref = resolve_path_reference(
                    evidence_canonical,
                    context=PathContext(
                        repository_root=Path(
                            repository_root or Path(__file__).resolve().parents[3]
                        ),
                        experiment_manifest_path=Path(experiment_manifest_path),
                    ),
                    policy=evidence_policy,  # type: ignore[arg-type]
                    expected_kind="file",
                    allow_external_absolute=False,
                )
            else:
                raise FormalConfigError(
                    f"selected_dev_run_evidence.path_policy invalid: {evidence_policy!r}"
                )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            if isinstance(exc, FormalConfigError):
                raise
            raise FormalConfigError(str(exc)) from exc
        if _is_placeholder(evidence_declared) or _is_placeholder(evidence_canonical):
            raise FormalConfigError("selected_dev_run_evidence path must not be a placeholder")
        evidence_path = evidence_ref.resolved_path
        _require_nonempty_file(evidence_path, label="selected_dev_run_evidence")
        frozen_evidence_sha = _require_sha256_value(
            evidence.get("sha256"),
            label="selected_dev_run_evidence.sha256",
        )
        if frozen_evidence_sha != sha256_file(evidence_path):
            raise FormalConfigError("selected_dev_run_evidence.sha256 mismatch.")
    else:
        raise FormalConfigError(
            "legacy_selected_dev_evidence_identity_requires_regeneration"
        )

    try:
        current_experiment = load_experiment_manifest(
            experiment_manifest_path,
            repository_root=Path(
                repository_root or Path(__file__).resolve().parents[3]
            ),
        )
    except (FileNotFoundError, ValueError):
        current_experiment = experiment_raw
    raw_experiment = _raw_experiment_mapping(current_experiment)
    frozen_core = freeze.get("experiment_core_sha256")
    if frozen_core is not None or require_complete:
        frozen_core = _require_sha256_value(
            frozen_core,
            label="experiment_core_sha256",
        )
        if frozen_core != experiment_core_sha256(raw_experiment):
            raise FormalConfigError("experiment_core_sha256_mismatch")
    candidate_sha = freeze.get("experiment_manifest_sha256_at_freeze_candidate")
    if candidate_sha is not None:
        _require_sha256_value(
            candidate_sha,
            label="experiment_manifest_sha256_at_freeze_candidate",
        )
        if freeze.get("experiment_manifest_sha256_authoritative") is not False:
            raise FormalConfigError(
                "freeze_manifest experiment_manifest_sha256_authoritative must be false."
            )

    base_sha, expected_shared_sha, expected_method_hashes, expected_merged_hashes = (
        _freeze_config_identities(
            experiment_manifest_path=experiment_manifest_path,
            experiment=experiment_raw,
            method_config_paths=method_config_paths,
            repository_root=Path(
                repository_root or Path(__file__).resolve().parents[3]
            ).resolve(),
        )
    )
    if freeze.get("base_config_sha256") is not None or require_complete:
        _require_sha256_value(
            freeze.get("base_config_sha256"),
            label="base_config_sha256",
        )
        _check_hash(
            freeze.get("base_config_sha256"),
            base_sha,
            label="base_config_sha256",
            require=True,
        )

    shared = experiment_raw.get("shared_model_config") or raw_experiment.get(
        "shared_model_config"
    )
    if shared and (freeze.get("shared_model_config_sha256") or require_complete):
        _require_sha256_value(
            freeze.get("shared_model_config_sha256"),
            label="shared_model_config_sha256",
        )
        _check_hash(
            freeze.get("shared_model_config_sha256"),
            expected_shared_sha,
            label="shared_model_config_sha256",
            require=require_complete or bool(freeze.get("shared_model_config_sha256")),
        )

    method_hashes = freeze.get("method_config_sha256") or {}
    if require_complete or method_hashes:
        if not isinstance(method_hashes, dict):
            raise FormalConfigError("freeze_manifest method_config_sha256 must be an object.")
        for mid in COMPARISON_METHOD_IDS:
            frozen_hash = method_hashes.get(mid)
            if frozen_hash in (None, "") and not require_complete:
                continue
            if frozen_hash in (None, "") and require_complete:
                raise FormalConfigError(f"freeze_manifest missing method_config_sha256.{mid}.")
            _require_sha256_value(
                frozen_hash,
                label=f"method_config_sha256.{mid}",
            )
            _check_hash(
                frozen_hash,
                expected_method_hashes.get(mid),
                label=f"method_config_sha256.{mid}",
                require=True,
            )

    merged_hashes = freeze.get("merged_method_config_sha256") or {}
    if require_complete or merged_hashes:
        if not isinstance(merged_hashes, dict):
            raise FormalConfigError(
                "freeze_manifest merged_method_config_sha256 must be an object."
            )
        if require_complete and set(merged_hashes) != set(COMPARISON_METHOD_IDS):
            raise FormalConfigError(
                "freeze_manifest merged_method_config_sha256 must contain exactly the comparison methods."
            )
        for mid in COMPARISON_METHOD_IDS:
            frozen_hash = merged_hashes.get(mid)
            if frozen_hash in (None, "") and not require_complete:
                continue
            _require_sha256_value(
                frozen_hash,
                label=f"merged_method_config_sha256.{mid}",
            )
            _check_hash(
                frozen_hash,
                expected_merged_hashes.get(mid),
                label=f"merged_method_config_sha256.{mid}",
                require=True,
            )

    prompt_hash = freeze.get("prompt_tree_sha256")
    frozen_prompt_bundle = freeze.get("ekell_prompt_bundle")
    if require_complete and not isinstance(frozen_prompt_bundle, dict):
        if prompt_hash:
            raise FormalConfigError(
                "legacy_freeze_prompt_identity_requires_regeneration"
            )
        raise FormalConfigError("freeze_manifest missing ekell_prompt_bundle.")
    if isinstance(frozen_prompt_bundle, dict):
        actual_prompt_bundle = _ekell_prompt_bundle_from_experiment(
            current_experiment,
            repository_root=Path(
                repository_root or Path(__file__).resolve().parents[3]
            ).resolve(),
        )
        if actual_prompt_bundle is None:
            raise FormalConfigError("ekell_prompt_bundle_actual_identity_unavailable")
        frozen_tree = _require_sha256_value(
            frozen_prompt_bundle.get("prompt_tree_sha256"),
            label="ekell_prompt_bundle.prompt_tree_sha256",
        )
        if prompt_hash is not None:
            _require_sha256_value(prompt_hash, label="prompt_tree_sha256")
            if prompt_hash != frozen_tree:
                raise FormalConfigError(
                    "freeze_manifest prompt_tree_sha256 mismatches ekell_prompt_bundle."
                )
        for field in (
            "declared_prompt_dir",
            "canonical_prompt_dir",
            "path_policy",
            "prompt_tree_sha256",
        ):
            frozen_value = frozen_prompt_bundle.get(field)
            if type(frozen_value) is not str or not frozen_value:
                raise FormalConfigError(
                    f"freeze_manifest ekell_prompt_bundle.{field} must be a non-empty string."
                )
            if frozen_value != actual_prompt_bundle.get(field):
                raise FormalConfigError(
                    f"freeze_manifest ekell_prompt_bundle.{field} mismatch."
                )
        frozen_files = frozen_prompt_bundle.get("required_prompt_files")
        if not isinstance(frozen_files, dict):
            raise FormalConfigError(
                "freeze_manifest ekell_prompt_bundle.required_prompt_files must be an object."
            )
        if set(frozen_files) != set(EKELL_REQUIRED_PROMPTS):
            raise FormalConfigError(
                "freeze_manifest ekell_prompt_bundle.required_prompt_files must contain "
                "exactly all required E-KELL prompts."
            )
        actual_files = actual_prompt_bundle["required_prompt_files"]
        for name in EKELL_REQUIRED_PROMPTS:
            frozen_file_sha = _require_sha256_value(
                frozen_files.get(name),
                label=f"ekell_prompt_bundle.required_prompt_files.{name}",
            )
            if frozen_file_sha != actual_files.get(name):
                raise FormalConfigError(
                    f"freeze_manifest ekell_prompt_bundle.required_prompt_files.{name} mismatch."
                )
    elif prompt_hash:
        if require_complete:
            raise FormalConfigError(
                "legacy_freeze_prompt_identity_requires_regeneration"
            )
        _require_sha256_value(prompt_hash, label="prompt_tree_sha256")

    # Bundle / corpus / schema: standard runner_bundle block is authoritative for formal runs.
    runner_block = runner_bundle_block_from_freeze(freeze)
    if require_complete:
        if not isinstance(freeze.get("runner_bundle"), dict):
            raise FormalConfigError("freeze_manifest missing runner_bundle block.")
        if runner_block.get("bundle_checksum") and not (
            runner_block.get("producer_declared_checksum") or runner_block.get("consumer_computed_hash")
        ):
            raise FormalConfigError("legacy_ambiguous_bundle_checksum_not_allowed")
        for field in RUNNER_BUNDLE_IDENTITY_FIELDS:
            if not runner_block.get(field):
                raise FormalConfigError(f"freeze_manifest missing runner_bundle.{field}.")
            _require_sha256_value(
                runner_block.get(field),
                label=f"runner_bundle.{field}",
            )
        producer_available = runner_block.get("producer_checksum_available")
        if type(producer_available) is not bool:
            raise FormalConfigError(
                "freeze_manifest runner_bundle.producer_checksum_available must be an exact boolean."
            )
        if producer_available and not runner_block.get("producer_declared_checksum"):
            raise FormalConfigError("freeze_manifest missing runner_bundle.producer_declared_checksum.")
        if producer_available:
            producer = _require_sha256_value(
                runner_block.get("producer_declared_checksum"),
                label="runner_bundle.producer_declared_checksum",
            )
            if producer != runner_block.get("consumer_computed_hash"):
                raise FormalConfigError(
                    "freeze_manifest runner_bundle producer checksum must match consumer hash."
                )
        elif runner_block.get("producer_declared_checksum") not in (None, ""):
            raise FormalConfigError(
                "freeze_manifest runner_bundle producer checksum availability is inconsistent."
            )

    for block_field, expected, label in (
        ("producer_declared_checksum", None, "runner_bundle.producer_declared_checksum"),
        ("consumer_computed_hash", expected_runner_bundle_checksum, "runner_bundle.consumer_computed_hash"),
        ("input_cases_sha256", None, "runner_bundle.input_cases_sha256"),
        ("prediction_schema_sha256", expected_prediction_schema_checksum, "runner_bundle.prediction_schema_sha256"),
        ("corpus_aggregate_sha256", expected_corpus_checksum, "runner_bundle.corpus_aggregate_sha256"),
    ):
        freeze_val = runner_block.get(block_field) or freeze.get(
            {
                "producer_declared_checksum": "producer_declared_checksum",
                "consumer_computed_hash": "runner_bundle_checksum",
                "input_cases_sha256": "input_cases_sha256",
                "prediction_schema_sha256": "prediction_schema_checksum",
                "corpus_aggregate_sha256": "corpus_checksum",
            }[block_field]
        )
        if block_field == "input_cases_sha256" and require_complete and not freeze_val:
            raise FormalConfigError("freeze_manifest missing runner_bundle.input_cases_sha256.")
        if block_field == "consumer_computed_hash" and require_complete and not freeze_val:
            raise FormalConfigError("freeze_manifest missing runner_bundle.consumer_computed_hash.")
        if expected is not None:
            _check_optional_pair(freeze_val, expected, label=label, require=require_complete)
        elif require_complete and block_field in RUNNER_BUNDLE_IDENTITY_FIELDS and freeze_val in (None, ""):
            raise FormalConfigError(f"freeze_manifest missing {label}.")

    # LLM identity vs shared config
    freeze_llm = freeze.get("llm") or {}
    if require_complete or freeze_llm:
        if shared:
            from external_baselines.common.io import read_yaml

            shared_path = _resolve_path(str(shared))
            if shared_path.is_file():
                shared_cfg = read_yaml(shared_path)
                llm_block = shared_cfg.get("llm") or {}
                for field in ("provider", "model"):
                    expected = llm_block.get(field)
                    frozen = freeze_llm.get(field)
                    if frozen or require_complete:
                        _check_optional_pair(frozen, expected, label=f"llm.{field}", require=require_complete)
                expected_ver = llm_block.get("model_version") or llm_block.get("version")
                frozen_ver = freeze_llm.get("model_version") or freeze_llm.get("version")
                if frozen_ver or require_complete:
                    _check_optional_pair(
                        frozen_ver, expected_ver, label="llm.model_version", require=require_complete
                    )

    emb = freeze.get("embedding") or {}
    if emb.get("model_version") is None or _is_placeholder(emb.get("model_version")):
        raise FormalConfigError("freeze_manifest embedding.model_version must be set.")
    if require_complete:
        for field in ("backend", "model_name", "dimension", "normalize_embeddings"):
            if field not in emb or emb.get(field) in (None, ""):
                raise FormalConfigError(f"freeze_manifest embedding.{field} must be set.")
        for field in ("backend", "model_name", "model_version"):
            if type(emb.get(field)) is not str or not emb[field]:
                raise FormalConfigError(
                    f"freeze_manifest embedding.{field} must be a non-empty string."
                )
        if type(emb.get("dimension")) is not int or emb["dimension"] <= 0:
            raise FormalConfigError("freeze_manifest embedding.dimension must be a positive integer.")
        if type(emb.get("normalize_embeddings")) is not bool:
            raise FormalConfigError(
                "freeze_manifest embedding.normalize_embeddings must be an exact boolean."
            )

    # Index checksums from loaded manifests or expected_indexes
    freeze_indexes = freeze.get("indexes") or {}
    expected_idx = dict(expected_indexes or {})
    loaded = dict(loaded_index_manifests or {})

    def _loaded_block(key: str) -> dict[str, Any]:
        raw = loaded.get(key) or {}
        if not isinstance(raw, dict):
            return {}
        # Accept either nested manifest or flat fields
        if "index_checksum" in raw or "checksum" in raw:
            return raw
        return dict(raw.get("manifest") or raw)

    dense_loaded = _loaded_block("dense")
    ekell_loaded = _loaded_block("ekell")
    hybrid_loaded = _loaded_block("hybrid_dense_dependency") or dense_loaded

    dense_freeze = freeze_indexes.get("dense") or {}
    hybrid_freeze = freeze_indexes.get("hybrid_dense_dependency") or {}
    ekell_freeze = freeze_indexes.get("ekell") or {}

    if require_complete:
        dense_freeze = _require_complete_index_block(
            freeze_indexes.get("dense"),
            label="dense",
            fields=DENSE_COMPLETE_INDEX_FIELDS,
        )
        hybrid_freeze = _require_complete_index_block(
            freeze_indexes.get("hybrid_dense_dependency"),
            label="hybrid_dense_dependency",
            fields=HYBRID_COMPLETE_INDEX_FIELDS,
        )
        ekell_freeze = _require_complete_index_block(
            freeze_indexes.get("ekell"),
            label="ekell",
            fields=EKELL_COMPLETE_INDEX_FIELDS,
        )

    if dense_loaded or expected_idx.get("dense") or (require_complete and dense_freeze):
        expected_dense = (expected_idx.get("dense") or {}) if isinstance(expected_idx.get("dense"), dict) else {}
        actual_checksum = (
            expected_dense.get("index_checksum")
            or dense_loaded.get("index_checksum")
            or dense_loaded.get("checksum")
        )
        if dense_freeze.get("index_checksum") or require_complete:
            _check_optional_pair(
                dense_freeze.get("index_checksum"),
                actual_checksum,
                label="indexes.dense.index_checksum",
                require=require_complete,
            )
        if dense_freeze.get("index_manifest_sha256") or require_complete:
            _check_optional_pair(
                dense_freeze.get("index_manifest_sha256"),
                expected_dense.get("index_manifest_sha256") or dense_loaded.get("index_manifest_sha256"),
                label="indexes.dense.index_manifest_sha256",
                require=False,
            )
        if dense_freeze.get("corpus_checksum") and (
            expected_dense.get("corpus_checksum") or dense_loaded.get("corpus_checksum")
        ):
            _check_optional_pair(
                dense_freeze.get("corpus_checksum"),
                expected_dense.get("corpus_checksum") or dense_loaded.get("corpus_checksum"),
                label="indexes.dense.corpus_checksum",
            )
        if dense_freeze.get("model_version") and (
            expected_dense.get("model_version") or dense_loaded.get("model_version")
        ):
            _check_optional_pair(
                dense_freeze.get("model_version"),
                expected_dense.get("model_version") or dense_loaded.get("model_version"),
                label="indexes.dense.model_version",
            )
        dense_live = {**dense_loaded, **expected_dense}
        _compare_index_identity_fields(
            dense_freeze,
            dense_live,
            label="dense",
            fields=DENSE_COMPLETE_INDEX_FIELDS,
            require_live=require_complete
            and ("dense" in loaded or "dense" in expected_idx),
        )

    if hybrid_loaded or expected_idx.get("hybrid_dense_dependency") or (require_complete and hybrid_freeze):
        expected_hybrid = (
            expected_idx.get("hybrid_dense_dependency")
            if isinstance(expected_idx.get("hybrid_dense_dependency"), dict)
            else {}
        ) or {}
        actual_hybrid = (
            expected_hybrid.get("index_checksum")
            or hybrid_loaded.get("index_checksum")
            or hybrid_loaded.get("checksum")
            or dense_loaded.get("index_checksum")
            or dense_loaded.get("checksum")
        )
        if hybrid_freeze.get("index_checksum") or require_complete:
            _check_optional_pair(
                hybrid_freeze.get("index_checksum"),
                actual_hybrid,
                label="indexes.hybrid_dense_dependency.index_checksum",
                require=require_complete,
            )
        dense_cs = dense_freeze.get("index_checksum") or dense_loaded.get("index_checksum") or dense_loaded.get(
            "checksum"
        )
        hybrid_cs = hybrid_freeze.get("index_checksum") or actual_hybrid
        if dense_cs and hybrid_cs and str(dense_cs) != str(hybrid_cs):
            raise FormalConfigError("freeze_manifest hybrid dense dependency checksum must match dense.")
        dense_sha = dense_freeze.get("index_manifest_sha256") or dense_loaded.get("index_manifest_sha256")
        hybrid_sha = hybrid_freeze.get("index_manifest_sha256") or expected_hybrid.get("index_manifest_sha256")
        if dense_sha and hybrid_sha and str(dense_sha) != str(hybrid_sha):
            raise FormalConfigError("freeze_manifest hybrid dense dependency manifest SHA must match dense.")
        hybrid_live = {
            **({"index_checksum": dense_loaded.get("index_checksum") or dense_loaded.get("checksum")} if dense_loaded else {}),
            **hybrid_loaded,
            **expected_hybrid,
        }
        _compare_index_identity_fields(
            hybrid_freeze,
            hybrid_live,
            label="hybrid_dense_dependency",
            fields=HYBRID_COMPLETE_INDEX_FIELDS,
            require_live=require_complete
            and (
                "hybrid_dense_dependency" in loaded
                or "hybrid_dense_dependency" in expected_idx
            ),
        )

    if ekell_loaded or expected_idx.get("ekell") or (require_complete and ekell_freeze):
        expected_ekell = (expected_idx.get("ekell") or {}) if isinstance(expected_idx.get("ekell"), dict) else {}
        actual_ekell = (
            expected_ekell.get("index_checksum")
            or ekell_loaded.get("index_checksum")
            or ekell_loaded.get("checksum")
        )
        if ekell_freeze.get("index_checksum") or require_complete:
            _check_optional_pair(
                ekell_freeze.get("index_checksum"),
                actual_ekell,
                label="indexes.ekell.index_checksum",
                require=require_complete,
            )
        for field in ("index_manifest_sha256", "kg_checksum", "corpus_checksum", "model_version"):
            frozen = ekell_freeze.get(field)
            actual = expected_ekell.get(field) or ekell_loaded.get(field)
            if frozen or (require_complete and field == "index_manifest_sha256"):
                _check_optional_pair(frozen, actual, label=f"indexes.ekell.{field}", require=False)
        ekell_live = {**ekell_loaded, **expected_ekell}
        _compare_index_identity_fields(
            ekell_freeze,
            ekell_live,
            label="ekell",
            fields=EKELL_COMPLETE_INDEX_FIELDS,
            require_live=require_complete
            and ("ekell" in loaded or "ekell" in expected_idx),
        )

    if require_complete:
        dense_normalize = dense_freeze["normalize_embeddings"]
        ekell_normalize = ekell_freeze["normalize_embeddings"]
        frozen_normalize = emb["normalize_embeddings"]
        if not (dense_normalize is ekell_normalize is frozen_normalize):
            raise FormalConfigError("cross_method_normalize_embeddings_mismatch")
        assert_freeze_provenance_portable(
            freeze.get("path_provenance") or {},
            require_complete_entries=True,
        )

    result = {"ok": True, "freeze_id": freeze.get("freeze_id"), "require_complete": require_complete}
    if migration_warning:
        result["migration_warning"] = migration_warning
    return result


def validate_frozen_runtime_inputs(
    freeze_manifest: str | Path | dict[str, Any],
    *,
    bundle: dict[str, Any] | None = None,
    method_configs: dict[str, dict[str, Any]] | None = None,
    loaded_index_manifests: dict[str, Any] | None = None,
    require_complete_indexes: bool | None = None,
) -> dict[str, Any]:
    """Validate a freeze manifest against live runtime bundle/configs/indexes."""
    freeze = (
        freeze_manifest
        if isinstance(freeze_manifest, dict)
        else read_json(freeze_manifest)
    )
    if not isinstance(freeze, dict):
        raise FormalConfigError("freeze_manifest must be a JSON object.")

    bundle = bundle or {}
    method_configs = method_configs or {}
    if require_complete_indexes is None:
        require_complete_indexes = isinstance(freeze.get("runner_bundle"), dict)

    expected_bundle = bundle.get("producer_declared_checksum") or bundle.get(
        "consumer_computed_bundle_hash"
    )
    from external_baselines.interop.bundle import runner_bundle_corpus_aggregate_sha256

    expected_corpus = runner_bundle_corpus_aggregate_sha256(
        bundle,
        required=bool(require_complete_indexes and bundle),
    )
    expected_schema = bundle.get("prediction_schema_sha256")

    # Embedding identity from method configs when available
    emb_freeze = freeze.get("embedding") or {}
    configured_embedding_blocks: list[tuple[str, dict[str, Any]]] = []
    for mid in ("dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"):
        cfg = method_configs.get(mid) or {}
        if mid == "ekell_style_controlled_shared_llm":
            block = cfg.get("ekell_vector") or {}
        else:
            block = cfg.get("dense_rag") or {}
        if not block:
            continue
        configured_embedding_blocks.append((mid, block))
        for field in ("backend", "model_name", "model_version"):
            if emb_freeze.get(field) and block.get(field) and str(emb_freeze[field]) != str(block[field]):
                raise FormalConfigError(f"freeze embedding.{field} mismatches runtime {mid}.")
        if "normalize_embeddings" in emb_freeze and "normalize_embeddings" in block:
            if (
                type(emb_freeze["normalize_embeddings"]) is not bool
                or type(block["normalize_embeddings"]) is not bool
                or emb_freeze["normalize_embeddings"] != block["normalize_embeddings"]
            ):
                raise FormalConfigError("freeze embedding.normalize_embeddings mismatches runtime.")
        if emb_freeze.get("dimension") is not None and block.get("dimension") is not None:
            if emb_freeze["dimension"] != block["dimension"]:
                raise FormalConfigError("freeze embedding.dimension mismatches runtime.")
    if require_complete_indexes and method_configs:
        for mid in ("dense_rag", "hybrid_rag", "ekell_style_controlled_shared_llm"):
            if not any(configured_mid == mid for configured_mid, _block in configured_embedding_blocks):
                raise FormalConfigError(f"runtime method config missing embedding identity for {mid}.")

    # LLM identity
    freeze_llm = freeze.get("llm") or {}
    for cfg in method_configs.values():
        llm = cfg.get("llm") or {}
        if not llm:
            continue
        for field in ("provider", "model"):
            if freeze_llm.get(field) and llm.get(field) and str(freeze_llm[field]) != str(llm[field]):
                raise FormalConfigError(f"freeze llm.{field} mismatches runtime config.")
        frozen_ver = freeze_llm.get("model_version") or freeze_llm.get("version")
        runtime_ver = llm.get("model_version") or llm.get("version")
        if frozen_ver and runtime_ver and str(frozen_ver) != str(runtime_ver):
            raise FormalConfigError("freeze llm.model_version mismatches runtime config.")
        break

    if expected_bundle and freeze.get("runner_bundle_checksum"):
        if str(freeze["runner_bundle_checksum"]) != str(expected_bundle):
            raise FormalConfigError("freeze runner_bundle_checksum mismatches loaded bundle.")
    if expected_corpus and freeze.get("corpus_checksum"):
        if str(freeze["corpus_checksum"]) != str(expected_corpus):
            raise FormalConfigError("freeze corpus_checksum mismatches loaded bundle.")
    if expected_schema and freeze.get("prediction_schema_checksum"):
        if str(freeze["prediction_schema_checksum"]) != str(expected_schema):
            raise FormalConfigError("freeze prediction_schema_checksum mismatches loaded bundle.")

    freeze_indexes = freeze.get("indexes") or {}
    loaded = loaded_index_manifests or {}
    if require_complete_indexes:
        if not isinstance(freeze.get("indexes"), dict):
            raise FormalConfigError("freeze_manifest missing indexes block.")
        dense_freeze = _require_complete_index_block(
            freeze_indexes.get("dense"),
            label="dense",
            fields=DENSE_COMPLETE_INDEX_FIELDS,
        )
        hybrid_freeze = _require_complete_index_block(
            freeze_indexes.get("hybrid_dense_dependency"),
            label="hybrid_dense_dependency",
            fields=HYBRID_COMPLETE_INDEX_FIELDS,
        )
        ekell_freeze = _require_complete_index_block(
            freeze_indexes.get("ekell"),
            label="ekell",
            fields=EKELL_COMPLETE_INDEX_FIELDS,
        )
    else:
        dense_freeze = freeze_indexes.get("dense") or {}
        hybrid_freeze = freeze_indexes.get("hybrid_dense_dependency") or {}
        ekell_freeze = freeze_indexes.get("ekell") or {}

    def _loaded_identity(key: str) -> dict[str, Any]:
        block = loaded.get(key) or {}
        if not isinstance(block, dict):
            return {}
        if "index_checksum" in block or "checksum" in block:
            return block
        return dict(block.get("manifest") or {})

    dense_loaded = _loaded_identity("dense")
    hybrid_loaded = _loaded_identity("hybrid_dense_dependency")
    ekell_loaded = _loaded_identity("ekell")
    if require_complete_indexes:
        for label, block in (
            ("dense", dense_loaded),
            ("hybrid_dense_dependency", hybrid_loaded),
            ("ekell", ekell_loaded),
        ):
            if not block:
                raise FormalConfigError(f"freeze_manifest loaded index missing indexes.{label}.")
    dense_cs = dense_loaded.get("index_checksum") or dense_loaded.get("checksum")
    hybrid_cs = (
        hybrid_loaded.get("index_checksum")
        or hybrid_loaded.get("checksum")
        or dense_cs
    )
    ekell_cs = ekell_loaded.get("index_checksum") or ekell_loaded.get("checksum")

    dense_checks: list[dict[str, Any]] = []
    hybrid_checks: list[dict[str, Any]] = []
    ekell_checks: list[dict[str, Any]] = []
    if isinstance(dense_freeze, dict):
        dense_checks = _compare_index_identity_fields(
            dense_freeze,
            dense_loaded,
            label="dense",
            fields=DENSE_COMPLETE_INDEX_FIELDS,
            require_live=require_complete_indexes or bool(dense_freeze),
        )
    if isinstance(hybrid_freeze, dict):
        hybrid_live = {
            "index_checksum": dense_cs,
            "index_manifest_sha256": dense_loaded.get("index_manifest_sha256"),
        }
        for field in HYBRID_COMPLETE_INDEX_FIELDS:
            if field in hybrid_loaded and hybrid_loaded[field] != hybrid_live[field]:
                raise FormalConfigError(
                    f"runtime hybrid dense dependency {field} must match dense."
                )
        hybrid_checks = _compare_index_identity_fields(
            hybrid_freeze,
            hybrid_live,
            label="hybrid_dense_dependency",
            fields=HYBRID_COMPLETE_INDEX_FIELDS,
            require_live=require_complete_indexes or bool(hybrid_freeze),
        )
    hybrid_freeze_checksum = hybrid_freeze.get("index_checksum") if isinstance(hybrid_freeze, dict) else None
    if hybrid_freeze_checksum and hybrid_cs and str(hybrid_freeze_checksum) != str(hybrid_cs):
        raise FormalConfigError(
            "freeze indexes.hybrid_dense_dependency.index_checksum mismatches loaded index."
        )
    if dense_cs and hybrid_cs and str(dense_cs) != str(hybrid_cs):
        raise FormalConfigError("runtime hybrid dense dependency checksum must match dense.")
    dense_sha = dense_loaded.get("index_manifest_sha256")
    hybrid_sha = hybrid_loaded.get("index_manifest_sha256") or dense_sha
    if dense_sha and hybrid_sha and str(dense_sha) != str(hybrid_sha):
        raise FormalConfigError("runtime hybrid dense dependency manifest SHA must match dense.")
    if isinstance(ekell_freeze, dict):
        ekell_checks = _compare_index_identity_fields(
            ekell_freeze,
            ekell_loaded,
            label="ekell",
            fields=EKELL_COMPLETE_INDEX_FIELDS,
            require_live=require_complete_indexes or bool(ekell_freeze),
        )

    if require_complete_indexes:
        frozen_normalize = emb_freeze.get("normalize_embeddings")
        if type(frozen_normalize) is not bool:
            raise FormalConfigError(
                "freeze embedding.normalize_embeddings must be an exact boolean."
            )
        if not (
            dense_loaded.get("normalize_embeddings")
            is ekell_loaded.get("normalize_embeddings")
            is frozen_normalize
        ):
            raise FormalConfigError("cross_method_normalize_embeddings_mismatch")

    result = {
        "ok": True,
        "freeze_manifest": (
            str(freeze_manifest) if not isinstance(freeze_manifest, dict) else freeze.get("freeze_id")
        ),
        "runner_bundle_checksum": expected_bundle,
        "corpus_checksum": expected_corpus,
        "prediction_schema_checksum": expected_schema,
        "indexes": {
            "dense": dict(dense_loaded, checksum=dense_cs),
            "hybrid_dense_dependency": dict(hybrid_loaded, checksum=hybrid_cs),
            "ekell": dict(ekell_loaded, checksum=ekell_cs),
        },
        "dense": _identity_match_report(dense_checks),
        "hybrid_dense_dependency": _identity_match_report(hybrid_checks),
        "ekell": _identity_match_report(ekell_checks),
    }
    return result
