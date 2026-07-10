"""Validate formal experiment and method configs before paper-facing runs.

LOCAL GUARD — rejects heuristic LLM, smoke embeddings, placeholders, and legacy IDs.

Two modes:
- Template validation: allow_placeholders=True (permits .example paths and placeholder values)
- Formal validation: default (rejects .example paths, placeholders, missing config files)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from external_baselines.common.experiment_manifest import build_method_config, load_experiment_manifest
from external_baselines.common.io import read_yaml
from external_baselines.method_registry import (
    canonicalize_method_id,
    comparison_suite_methods,
    fallback_methods,
    legacy_methods,
    main_table_methods,
    paper_fidelity_methods,
)

PLACEHOLDER_EXACT = frozenset(
    {
        "REQUIRED_BEFORE_FORMAL_RUN",
        "TBD",
        "TODO",
        "CHANGEME",
        "REPLACE",
        "EXAMPLE",
        "<REQUIRED>",
    }
)

PLACEHOLDER_PREFIXES = (
    "REPLACE",
    "REPLACE_WITH_",
    "REQUIRED_",
    "TODO",
    "TBD",
    "CHANGEME",
)

PLACEHOLDER_SUBSTRINGS = (
    "path/to/",
    "<required>",
    "<model-hash>",
    "<corpus-hash>",
)

FORMAL_EKELL_METHODS = frozenset(
    {
        "ekell_style_controlled_shared_llm",
        "ekell_style_paper_fidelity",
    }
)

FORMAL_METHOD_IDS = frozenset(main_table_methods()) | frozenset(paper_fidelity_methods())
COMPARISON_FORMAL_METHOD_IDS = frozenset(comparison_suite_methods())
COMPARISON_SUITE_EXACT = list(comparison_suite_methods())

SMOKE_LLM_PROVIDERS = frozenset({"heuristic", "local", "smoke", ""})
SMOKE_EMBEDDING_BACKENDS = frozenset(
    {
        "smoke",
        "hash",
        "smoke_hash",
        "hash_smoke",
        "deterministic_hash_smoke",
        "smoke_hash_embedding",
    }
)

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9]{16,}", re.I),
)

ROOT_REL = Path(__file__).resolve().parents[3]


class FormalConfigError(ValueError):
    """Raised when a formal config violates paper-facing safety rules."""


def _is_example_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".example") or path.lower().endswith(".example")


def _resolve_repo_path(rel: str) -> Path:
    candidate = Path(rel)
    if candidate.is_file():
        return candidate
    return ROOT_REL / rel


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    upper = text.upper()
    if upper in PLACEHOLDER_EXACT:
        return True
    if any(upper.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES):
        return True
    lower = text.lower()
    if any(token in lower for token in PLACEHOLDER_SUBSTRINGS):
        return True
    return False


def _validate_positive_dimension(value: Any, *, allow_placeholders: bool = False) -> int:
    if allow_placeholders and _is_placeholder(value):
        return 0
    if _is_placeholder(value):
        raise FormalConfigError(f"ekell_vector.dimension must be a positive integer (got placeholder {value!r}).")
    try:
        dim = int(value)
    except (TypeError, ValueError) as exc:
        raise FormalConfigError(
            f"ekell_vector.dimension must be a positive integer (got {value!r})."
        ) from exc
    if dim <= 0:
        raise FormalConfigError(f"ekell_vector.dimension must be > 0 (got {dim}).")
    return dim


def _walk_strings(obj: Any, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(_walk_strings(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found.extend(_walk_strings(item, f"{path}[{i}]"))
    elif isinstance(obj, str):
        found.append((path, obj))
    return found


def _check_no_secrets(config: dict[str, Any]) -> None:
    for path, value in _walk_strings(config):
        for pattern in SECRET_PATTERNS:
            if pattern.search(value):
                raise FormalConfigError(f"Possible secret at {path}; use env vars instead.")


def _llm_block(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm", {})
    return dict(llm) if isinstance(llm, dict) else {}


def validate_llm_for_formal(config: dict[str, Any], *, allow_placeholders: bool = False) -> None:
    llm = _llm_block(config)
    provider = str(llm.get("provider", "heuristic")).lower().strip()
    model = str(llm.get("model") or "").strip()
    model_version = str(llm.get("model_version") or llm.get("version") or "").strip()

    if provider in SMOKE_LLM_PROVIDERS:
        raise FormalConfigError(f"Formal config rejects smoke LLM provider: {provider!r}")
    if _is_placeholder(model) and not allow_placeholders:
        raise FormalConfigError("Formal config requires llm.model (non-placeholder).")
    if _is_placeholder(model_version) and not allow_placeholders:
        raise FormalConfigError("Formal config requires llm.model_version (non-placeholder).")
    if bool(config.get("paper_final", False)) and bool(llm.get("allow_model_env_override", False)):
        raise FormalConfigError(
            "paper_final=true forbids llm.allow_model_env_override=true; "
            "formal model identity must come from YAML."
        )

def validate_dense_config_for_real_run(
    config: dict[str, Any],
    *,
    allow_placeholders: bool = False,
    validation_stage: str = "formal",
) -> None:
    dense = config.get("dense_rag") or {}
    if not isinstance(dense, dict) or not dense:
        raise FormalConfigError("Formal Dense RAG requires explicit dense_rag block.")
    backend = str(dense.get("backend", "smoke")).casefold().replace("-", "_")
    if backend in SMOKE_EMBEDDING_BACKENDS:
        raise FormalConfigError(f"Formal Dense RAG rejects smoke/hash backend: {backend!r}")
    if not bool(dense.get("reject_smoke", False)):
        raise FormalConfigError("Formal Dense RAG requires dense_rag.reject_smoke=true.")
    for field in ("model_name", "model_version"):
        value = dense.get(field)
        if _is_placeholder(value) and not allow_placeholders:
            raise FormalConfigError(f"Formal Dense RAG requires dense_rag.{field} (non-placeholder).")
    dim = dense.get("dimension", dense.get("dim"))
    if allow_placeholders and _is_placeholder(dim):
        pass
    else:
        try:
            if int(dim) <= 0:
                raise FormalConfigError(f"dense_rag.dimension must be > 0 (got {dim}).")
        except (TypeError, ValueError) as exc:
            raise FormalConfigError(f"dense_rag.dimension must be a positive integer (got {dim!r}).") from exc
    if "normalize_embeddings" not in dense and not allow_placeholders:
        raise FormalConfigError("Formal Dense RAG requires dense_rag.normalize_embeddings to be set.")
    index_path = dense.get("index_path")
    if not index_path or (_is_placeholder(index_path) and not allow_placeholders):
        raise FormalConfigError("Formal Dense RAG requires non-placeholder dense_rag.index_path.")
    if validation_stage != "template" and not allow_placeholders:
        path = _resolve_repo_path(str(index_path))
        if not path.exists():
            raise FormalConfigError(f"Formal Dense RAG index_path does not exist: {index_path}")


def validate_hybrid_config_for_real_run(
    config: dict[str, Any],
    *,
    allow_placeholders: bool = False,
    validation_stage: str = "formal",
    dense_config: dict[str, Any] | None = None,
) -> None:
    hybrid = config.get("hybrid_rag") or {}
    dense = config.get("dense_rag") or {}
    if not isinstance(hybrid, dict) or not hybrid:
        raise FormalConfigError("Formal Hybrid RAG requires explicit hybrid_rag block.")
    lexical = str(hybrid.get("lexical_method") or "bm25").lower()
    if lexical != "bm25":
        raise FormalConfigError(f"Formal Hybrid RAG requires lexical_method=bm25 (got {lexical!r}).")
    backend = str(
        dense.get("backend") or hybrid.get("dense_method") or "smoke"
    ).casefold().replace("-", "_")
    if backend in SMOKE_EMBEDDING_BACKENDS:
        raise FormalConfigError(f"Formal Hybrid RAG rejects smoke/hash dense backend: {backend!r}")
    if not bool(hybrid.get("reject_smoke", dense.get("reject_smoke", False))):
        raise FormalConfigError("Formal Hybrid RAG requires reject_smoke=true.")
    model_name = dense.get("model_name") or hybrid.get("dense_model_name")
    model_version = dense.get("model_version") or hybrid.get("dense_model_version")
    for field, value in (("model_name", model_name), ("model_version", model_version)):
        if _is_placeholder(value) and not allow_placeholders:
            raise FormalConfigError(f"Formal Hybrid RAG requires dense {field} (non-placeholder).")
    dim = dense.get("dimension", hybrid.get("dimension"))
    if not (allow_placeholders and _is_placeholder(dim)):
        try:
            if int(dim) <= 0:
                raise FormalConfigError(f"hybrid dense dimension must be > 0 (got {dim}).")
        except (TypeError, ValueError) as exc:
            raise FormalConfigError(f"hybrid dense dimension must be a positive integer (got {dim!r}).") from exc
    try:
        top_k = int(hybrid.get("top_k", (config.get("retrieval") or {}).get("top_k", 5)))
        candidate_pool = int(hybrid.get("candidate_pool", top_k))
        rrf_k = float(hybrid.get("rrf_k", 0))
        lexical_weight = float(hybrid.get("lexical_weight", 0))
        dense_weight = float(hybrid.get("dense_weight", 0))
    except (TypeError, ValueError) as exc:
        raise FormalConfigError(f"Formal Hybrid RAG has invalid RRF parameters: {exc}") from exc
    if top_k <= 0:
        raise FormalConfigError("Formal Hybrid RAG requires top_k > 0.")
    if candidate_pool < top_k:
        raise FormalConfigError("Formal Hybrid RAG requires candidate_pool >= top_k.")
    if rrf_k <= 0:
        raise FormalConfigError("Formal Hybrid RAG requires rrf_k > 0.")
    if lexical_weight <= 0 or dense_weight <= 0:
        raise FormalConfigError("Formal Hybrid RAG requires lexical_weight > 0 and dense_weight > 0.")
    index_path = dense.get("index_path") or hybrid.get("index_path")
    if not index_path or (_is_placeholder(index_path) and not allow_placeholders):
        raise FormalConfigError("Formal Hybrid RAG requires non-placeholder dense index_path.")
    if validation_stage != "template" and not allow_placeholders:
        path = _resolve_repo_path(str(index_path))
        if not path.exists():
            raise FormalConfigError(f"Formal Hybrid RAG index_path does not exist: {index_path}")
    if dense_config is not None:
        other = dense_config.get("dense_rag") or {}
        other_path = other.get("index_path")
        if other_path and index_path and str(Path(str(other_path))) != str(Path(str(index_path))):
            raise FormalConfigError("Hybrid and Dense must share the same dense index_path.")
        for field in ("backend", "model_name", "model_version"):
            left = dense.get(field) or hybrid.get(f"dense_{field}" if field != "backend" else "dense_method")
            right = other.get(field)
            if left and right and str(left) != str(right):
                raise FormalConfigError(f"Hybrid/Dense embedding {field} mismatch.")
        left_cs = dense.get("index_checksum") or config.get("dense_index_checksum")
        right_cs = other.get("index_checksum") or dense_config.get("dense_index_checksum")
        if validation_stage == "formal" and left_cs and right_cs and str(left_cs) != str(right_cs):
            raise FormalConfigError("Hybrid/Dense index_checksum mismatch.")


def validate_ekell_vector_for_formal(
    config: dict[str, Any], *, allow_placeholders: bool = False
) -> None:
    vector = config.get("ekell_vector") or (config.get("ekell_style") or {}).get("vector") or {}
    if not isinstance(vector, dict) or not vector:
        raise FormalConfigError("Formal E-KELL config requires explicit ekell_vector block.")
    backend = str(vector.get("backend", "smoke")).casefold().replace("-", "_")
    if backend in SMOKE_EMBEDDING_BACKENDS:
        raise FormalConfigError(f"Formal E-KELL rejects smoke/hash backend: {backend!r}")
    if not bool(vector.get("reject_smoke", False)):
        raise FormalConfigError("Formal E-KELL requires ekell_vector.reject_smoke=true.")
    for field in ("model_name", "model_version"):
        value = vector.get(field)
        if _is_placeholder(value) and not allow_placeholders:
            raise FormalConfigError(f"Formal E-KELL requires ekell_vector.{field} (non-placeholder).")
    _validate_positive_dimension(vector.get("dimension"), allow_placeholders=allow_placeholders)
    index_path = vector.get("index_path")
    if index_path and not allow_placeholders and _is_placeholder(index_path):
        raise FormalConfigError("Formal E-KELL rejects placeholder ekell_vector.index_path.")


def validate_paper_fidelity_method_config(
    config: dict[str, Any], *, allow_placeholders: bool = False
) -> None:
    track = str(config.get("track") or "").lower()
    if track and track not in {"paper_fidelity", "b_paper_fidelity"}:
        raise FormalConfigError(f"paper-fidelity method requires track=paper_fidelity (got {track!r}).")
    if config.get("paper_original_output_format") is not True:
        raise FormalConfigError("paper-fidelity requires paper_original_output_format=true.")
    if config.get("controlled_output_format") is not False:
        raise FormalConfigError("paper-fidelity requires controlled_output_format=false.")
    if config.get("official_reproduction") is not False:
        raise FormalConfigError("paper-fidelity requires official_reproduction=false.")
    if bool(config.get("paper_fidelity_model_run")):
        validate_llm_for_formal(config, allow_placeholders=False)
        validate_ekell_vector_for_formal(config, allow_placeholders=False)
        evidence = str(config.get("paper_fidelity_run_evidence") or "").strip()
        if not evidence or _is_placeholder(evidence):
            raise FormalConfigError(
                "paper_fidelity_model_run=true requires non-placeholder paper_fidelity_run_evidence path."
            )


def _assert_config_path(
    path_str: str,
    *,
    label: str,
    allow_placeholders: bool,
    errors: list[str],
) -> None:
    if not path_str:
        errors.append(f"{label} is required.")
        return
    if _is_example_path(path_str) and not allow_placeholders:
        errors.append(f"{label} must not use .example path for formal runs: {path_str}")
        return
    if not allow_placeholders and not _resolve_repo_path(path_str).is_file():
        errors.append(f"{label} file not found: {path_str}")


def validate_method_config(
    config: dict[str, Any],
    *,
    method_id: str | None = None,
    allow_placeholders: bool = False,
    require_formal: bool = True,
    validation_stage: str = "formal",
    dense_config: dict[str, Any] | None = None,
) -> list[str]:
    """Return warnings; raise FormalConfigError on hard violations."""
    warnings: list[str] = []
    mid = canonicalize_method_id(method_id or str(config.get("method_id") or ""))
    if mid in legacy_methods():
        raise FormalConfigError(f"Legacy method {mid!r} cannot be used in formal configs.")
    if mid in fallback_methods():
        raise FormalConfigError(f"Fallback method {mid!r} cannot enter formal tables.")

    if str(config.get("method_id") or "") != mid and config.get("method_id"):
        warnings.append(f"method_id canonicalized {config.get('method_id')!r} → {mid!r}")

    paper_final = bool(config.get("paper_final", False))
    formal_ids = FORMAL_METHOD_IDS | COMPARISON_FORMAL_METHOD_IDS
    if require_formal or paper_final:
        if "llm" in config and _llm_block(config):
            validate_llm_for_formal(config, allow_placeholders=allow_placeholders)
        elif mid in formal_ids and not allow_placeholders:
            raise FormalConfigError(
                f"Formal method {mid!r} requires llm block in merged config (from shared_model_config)."
            )
        if mid in FORMAL_EKELL_METHODS:
            validate_ekell_vector_for_formal(config, allow_placeholders=allow_placeholders)
        if mid == "dense_rag":
            validate_dense_config_for_real_run(
                config,
                allow_placeholders=allow_placeholders,
                validation_stage=validation_stage,
            )
        if mid == "hybrid_rag":
            validate_hybrid_config_for_real_run(
                config,
                allow_placeholders=allow_placeholders,
                validation_stage=validation_stage,
                dense_config=dense_config,
            )
        if mid == "ekell_style_paper_fidelity":
            validate_paper_fidelity_method_config(config, allow_placeholders=allow_placeholders)
        enhanced = config.get("ekell_style") or {}
        if mid == "ekell_style_controlled_shared_llm":
            for flag in (
                "dense_entity_retrieval",
                "hybrid_subgraph_ranking",
                "reranker",
                "self_consistency",
                "structured_verification",
            ):
                if bool(enhanced.get(flag)):
                    raise FormalConfigError(
                        f"Controlled E-KELL forbids enhanced hook ekell_style.{flag}=true"
                    )

    _check_no_secrets(config)
    return warnings


def _apply_template_fallback_path(path_str: str, *, allow_placeholders: bool) -> str:
    if not allow_placeholders or not path_str:
        return path_str
    if _resolve_repo_path(path_str).is_file():
        return path_str
    if _is_example_path(path_str):
        return path_str
    alt = f"{path_str}.example"
    if _resolve_repo_path(alt).is_file():
        return alt
    return path_str


def validate_experiment_manifest(
    path: str | Path,
    *,
    allow_placeholders: bool = False,
    validation_stage: str | None = None,
    method_set: str | None = None,
    runtime_bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(path)
    stage = str(validation_stage or ("template" if allow_placeholders else "formal")).strip().lower()
    if stage not in {"template", "dry_run", "formal"}:
        raise FormalConfigError(f"Unknown validation_stage={validation_stage!r}")
    allow_placeholders = stage == "template" or allow_placeholders
    if stage == "template":
        allow_placeholders = True

    manifest = load_experiment_manifest(path)
    raw = manifest.get("raw") or read_yaml(path)
    if not isinstance(raw, dict):
        raise FormalConfigError(f"Experiment manifest must be a mapping: {path}")

    method_set_name = str(method_set or "main_table").strip().lower()
    if method_set_name not in {"main_table", "comparison_suite"}:
        raise FormalConfigError(f"Unknown method_set={method_set!r}")

    manifest_for_merge = dict(manifest)
    if allow_placeholders:
        manifest_for_merge["shared_model_config"] = _apply_template_fallback_path(
            str(manifest.get("shared_model_config") or ""),
            allow_placeholders=True,
        )

    errors: list[str] = []
    if stage != "template" and _is_example_path(str(path)):
        errors.append(
            f"{stage} validation rejects .example manifest paths; "
            "copy the template to a non-.example file first."
        )

    run_mode = str(raw.get("run_mode", "formal")).lower()
    if run_mode != "formal":
        errors.append(f"run_mode must be 'formal' for paper-facing manifest (got {run_mode!r})")

    if not bool(raw.get("paper_final", False)):
        errors.append("paper_final must be true for formal experiment manifest.")

    for key in (
        "require_bundle_checksum",
        "require_external_schema",
        "require_complete_case_match",
        "fail_on_schema_error",
        "fail_on_duplicate_case_id",
        "fail_on_missing_case",
        "fail_on_extra_case",
    ):
        if not bool(raw.get(key, False)):
            errors.append(f"{key} must be true for formal manifest.")

    freeze_status = str(raw.get("freeze_status", "")).lower()
    if freeze_status in {"paper_ready", "empirically_validated"}:
        errors.append(
            f"freeze_status {freeze_status!r} is not a valid config freeze state; "
            "use provisional or frozen."
        )
    if stage == "template":
        if freeze_status != "provisional":
            errors.append(f"template validation requires freeze_status=provisional (got {freeze_status!r}).")
    elif stage == "dry_run":
        if freeze_status not in {"provisional", "frozen"}:
            errors.append(f"dry_run validation requires freeze_status provisional|frozen (got {freeze_status!r}).")
    else:  # formal
        if freeze_status != "frozen":
            errors.append(f"formal validation requires freeze_status=frozen (got {freeze_status!r}).")
        freeze_path = raw.get("freeze_manifest")
        if not freeze_path or _is_placeholder(freeze_path):
            errors.append("formal validation requires freeze_manifest path.")
        else:
            freeze_file = _resolve_repo_path(str(freeze_path))
            if not freeze_file.is_file():
                errors.append(f"freeze_manifest file not found: {freeze_path}")
            else:
                try:
                    from external_baselines.common.freeze_manifest import validate_freeze_manifest

                    validate_freeze_manifest(
                        freeze_file,
                        experiment_manifest_path=path,
                        experiment_raw=raw,
                        require_complete=(method_set_name == "comparison_suite"),
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"freeze_manifest validation failed: {exc}")

    shared = str(raw.get("shared_model_config") or "")
    if not shared:
        errors.append("shared_model_config is required.")
    elif "smoke" in shared.lower() or "heuristic" in shared.lower():
        errors.append("shared_model_config must not reference smoke/heuristic configs.")
    else:
        _assert_config_path(
            shared,
            label="shared_model_config",
            allow_placeholders=allow_placeholders,
            errors=errors,
        )

    methods = raw.get("methods") or []
    if not isinstance(methods, list) or not methods:
        errors.append("methods list is required and non-empty.")

    expected_main = list(main_table_methods())
    pf_expected = list(paper_fidelity_methods())

    declared_main = [canonicalize_method_id(str(m)) for m in (raw.get("main_table_methods") or [])]
    declared_pf = [canonicalize_method_id(str(m)) for m in (raw.get("paper_fidelity_methods") or [])]
    declared_supp = [canonicalize_method_id(str(m)) for m in (raw.get("supplemental_methods") or [])]
    declared_comparison = [
        canonicalize_method_id(str(m)) for m in (raw.get("comparison_suite_methods") or [])
    ]

    if method_set_name == "comparison_suite":
        if declared_comparison:
            if declared_comparison != COMPARISON_SUITE_EXACT:
                errors.append(
                    "comparison_suite_methods must be exactly "
                    f"{COMPARISON_SUITE_EXACT} (got {declared_comparison})"
                )
            forbidden = set(paper_fidelity_methods()) | {
                "ekell_style_enhanced",
                "lightrag",
                "microsoft_graphrag",
                "fallback_graph_retrieval",
            }
            extras = [m for m in declared_comparison if m in forbidden]
            if extras:
                errors.append(f"comparison_suite_methods must not include {extras}.")
            missing = [m for m in COMPARISON_SUITE_EXACT if m not in declared_comparison]
            if missing:
                errors.append(f"comparison_suite_methods missing required methods: {missing}.")

    for mid in declared_pf:
        if mid in declared_main:
            errors.append(f"paper-fidelity method {mid!r} must not appear in main_table_methods.")
        if mid in declared_supp:
            errors.append(f"paper-fidelity method {mid!r} must not appear in supplemental_methods.")

    # Which method IDs must be formally validated for this method_set
    if method_set_name == "comparison_suite":
        required_formal_ids = set(COMPARISON_FORMAL_METHOD_IDS)
    else:
        required_formal_ids = set(FORMAL_METHOD_IDS)

    seen_ids: set[str] = set()
    merged_by_id: dict[str, dict[str, Any]] = {}
    for entry in methods:
        if isinstance(entry, str):
            entry = {"method_id": entry}
        if not isinstance(entry, dict):
            errors.append(f"Invalid method entry: {entry!r}")
            continue
        mid = canonicalize_method_id(str(entry.get("method_id") or ""))
        if mid in seen_ids:
            errors.append(f"Duplicate method_id in manifest: {mid}")
        seen_ids.add(mid)
        if str(entry.get("method_id") or "") == "ekell_style_faithful":
            errors.append("Manifest must use canonical ekell_style_controlled_shared_llm, not faithful.")
        if mid in pf_expected and mid in declared_main:
            errors.append(f"paper-fidelity method {mid!r} listed in main_table_methods.")
        if mid in pf_expected and mid in declared_supp:
            errors.append(f"paper-fidelity method {mid!r} listed in supplemental_methods.")

        should_validate = bool(entry.get("enabled", True)) and mid in required_formal_ids
        if method_set_name == "comparison_suite" and mid in COMPARISON_FORMAL_METHOD_IDS:
            should_validate = True
        if should_validate:
            cfg_path = str(entry.get("config") or entry.get("method_config") or "")
            if not cfg_path:
                errors.append(f"Formal method {mid} missing config path.")
            else:
                _assert_config_path(
                    cfg_path,
                    label=f"{mid} config",
                    allow_placeholders=allow_placeholders,
                    errors=errors,
                )
            try:
                entry_for_merge = dict(entry)
                entry_for_merge["config"] = _apply_template_fallback_path(
                    cfg_path, allow_placeholders=allow_placeholders
                )
                merged = build_method_config(manifest_for_merge, entry_for_merge)
                merged["paper_final"] = True
                merged_by_id[mid] = merged
                validate_method_config(
                    merged,
                    method_id=mid,
                    allow_placeholders=allow_placeholders,
                    require_formal=True,
                    validation_stage=stage,
                    dense_config=merged_by_id.get("dense_rag"),
                )
            except FormalConfigError as exc:
                errors.append(f"{mid} merged config: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{mid} config merge failed: {exc}")

    # Second pass: hybrid vs dense identity once both merged
    if "hybrid_rag" in merged_by_id and "dense_rag" in merged_by_id:
        try:
            validate_hybrid_config_for_real_run(
                merged_by_id["hybrid_rag"],
                allow_placeholders=allow_placeholders,
                validation_stage=stage,
                dense_config=merged_by_id["dense_rag"],
            )
        except FormalConfigError as exc:
            errors.append(f"hybrid_rag/dense_rag identity: {exc}")

    enabled_main = [
        canonicalize_method_id(str(e.get("method_id")))
        for e in methods
        if isinstance(e, dict) and e.get("enabled", True)
        and canonicalize_method_id(str(e.get("method_id") or "")) in expected_main
    ]
    if enabled_main and sorted(declared_main) != sorted(enabled_main):
        errors.append(
            f"main_table_methods must list enabled main-table methods {sorted(enabled_main)} "
            f"(got {declared_main})"
        )

    enabled_pf = [
        canonicalize_method_id(str(e.get("method_id")))
        for e in methods
        if isinstance(e, dict) and e.get("enabled", True)
        and canonicalize_method_id(str(e.get("method_id") or "")) in pf_expected
    ]
    if enabled_pf and declared_pf != pf_expected:
        errors.append(f"paper_fidelity_methods must be exactly {pf_expected} (got {declared_pf})")

    if errors:
        raise FormalConfigError("Formal manifest validation failed:\n- " + "\n- ".join(errors))

    return {
        "path": str(path),
        "experiment_id": raw.get("experiment_id"),
        "valid": True,
        "allow_placeholders": allow_placeholders,
        "validation_stage": stage,
        "method_set": method_set_name,
        "runtime_bundle_path": str(runtime_bundle_path) if runtime_bundle_path else None,
        "mode": stage,
    }
