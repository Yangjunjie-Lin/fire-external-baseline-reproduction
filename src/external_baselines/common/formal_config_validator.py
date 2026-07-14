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

from external_baselines.common.experiment_manifest import (
    build_method_config,
    enabled_methods,
    load_experiment_manifest,
)
from external_baselines.common.io import read_yaml
from external_baselines.common.strict_config_types import (
    MISSING,
    exact_bool,
    exact_int,
    exact_nonempty_string,
    exact_number,
)
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
VALIDATION_STAGES = frozenset({"template", "dry_run", "freeze_candidate", "formal"})

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


def _requires_paper_facing_strictness(stage: str) -> bool:
    return stage in {"index_build_candidate", "freeze_candidate", "formal"}


def _requires_existing_freeze(stage: str) -> bool:
    return stage == "formal"


def _is_example_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".example") or path.lower().endswith(".example")


def _resolve_repo_path(rel: str) -> Path:
    candidate = Path(rel)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (ROOT_REL / rel).resolve(strict=False)


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


def _validate_positive_dimension(value: Any, *, allow_placeholders: bool = False, field: str = "ekell_vector.dimension") -> int:
    if allow_placeholders and _is_placeholder(value):
        return 0
    if _is_placeholder(value):
        raise FormalConfigError(f"{field} must be a positive integer (got placeholder {value!r}).")
    if type(value) is not int:
        raise FormalConfigError(f"{field} must be a positive integer with exact YAML integer type (got {value!r}).")
    if value <= 0:
        raise FormalConfigError(f"{field} must be > 0 (got {value}).")
    return value


def _validate_exact_bool(
    value: Any,
    *,
    field: str,
    allow_placeholders: bool = False,
    required: bool | None = None,
    default: Any = MISSING,
) -> bool | None:
    if allow_placeholders and _is_placeholder(value):
        return None
    try:
        if value is MISSING:
            resolved = exact_bool(MISSING, field=field, default=False if default is MISSING else default)
        else:
            resolved = exact_bool(value, field=field)
    except ValueError as exc:
        raise FormalConfigError(str(exc)) from exc
    if required is not None and resolved is not required:
        raise FormalConfigError(f"{field} must be {str(required).lower()}.")
    return resolved


def _validate_exact_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
    allow_placeholders: bool = False,
) -> int:
    if allow_placeholders and _is_placeholder(value):
        return 0
    try:
        return exact_int(
            value,
            field=field,
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=minimum_inclusive,
            maximum_inclusive=maximum_inclusive,
        )
    except ValueError as exc:
        raise FormalConfigError(str(exc)) from exc


def _validate_exact_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
    allow_placeholders: bool = False,
) -> float:
    if allow_placeholders and _is_placeholder(value):
        return 0.0
    try:
        return exact_number(
            value,
            field=field,
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=minimum_inclusive,
            maximum_inclusive=maximum_inclusive,
        )
    except ValueError as exc:
        raise FormalConfigError(str(exc)) from exc


def _validate_exact_nonempty_string(
    value: Any,
    *,
    field: str,
    allow_placeholders: bool = False,  # noqa: ARG001 - placeholder checks run after exact type validation.
) -> str:
    try:
        return exact_nonempty_string(value, field=field)
    except ValueError as exc:
        raise FormalConfigError(str(exc)) from exc


def _validate_llm_numeric_params_for_formal(llm: dict[str, Any]) -> None:
    _validate_exact_number(
        llm["temperature"],
        field="llm.temperature",
        minimum=0,
        minimum_inclusive=True,
    )
    _validate_exact_number(
        llm["top_p"],
        field="llm.top_p",
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
        maximum_inclusive=True,
    )
    _validate_exact_int(llm["max_tokens"], field="llm.max_tokens", minimum=1)
    _validate_exact_int(llm["seed"], field="llm.seed")
    optional_numeric = (
        ("timeout_sec", 0, False),
        ("connect_timeout_sec", 0, False),
        ("read_timeout_sec", 0, False),
        ("write_timeout_sec", 0, False),
    )
    for field, minimum, inclusive in optional_numeric:
        if field in llm:
            _validate_exact_number(
                llm[field],
                field=f"llm.{field}",
                minimum=minimum,
                minimum_inclusive=inclusive,
            )
    if "max_retries" in llm:
        _validate_exact_int(
            llm["max_retries"],
            field="llm.max_retries",
            minimum=0,
            minimum_inclusive=True,
        )


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


def validate_llm_for_formal(
    config: dict[str, Any],
    *,
    allow_placeholders: bool = False,
    validation_stage: str = "formal",
) -> None:
    llm = _llm_block(config)
    if "allow_model_env_override" in llm:
        allow_override = _validate_exact_bool(
            llm["allow_model_env_override"],
            field="llm.allow_model_env_override",
            allow_placeholders=allow_placeholders,
        )
    else:
        allow_override = False
    raw_paper_final = config.get("paper_final", False)
    if type(raw_paper_final) is bool and raw_paper_final is True and allow_override is True:
        raise FormalConfigError(
            "paper_final=true requires llm.allow_model_env_override=false"
        )
    stage = str(validation_stage or "formal").strip().lower()
    provider = _validate_exact_nonempty_string(
        llm["provider"] if "provider" in llm else "heuristic",
        field="llm.provider",
        allow_placeholders=allow_placeholders,
    ).lower()
    model = _validate_exact_nonempty_string(
        llm.get("model"),
        field="llm.model",
        allow_placeholders=allow_placeholders,
    )
    model_version = _validate_exact_nonempty_string(
        llm["model_version"] if "model_version" in llm else llm.get("version"),
        field="llm.model_version",
        allow_placeholders=allow_placeholders,
    )
    api_key_env = ""
    if "api_key_env" in llm:
        api_key_env = _validate_exact_nonempty_string(
            llm["api_key_env"],
            field="llm.api_key_env",
            allow_placeholders=allow_placeholders,
        )
    if "base_url_env" in llm:
        _validate_exact_nonempty_string(
            llm["base_url_env"],
            field="llm.base_url_env",
            allow_placeholders=allow_placeholders,
        )

    if provider in SMOKE_LLM_PROVIDERS:
        raise FormalConfigError(f"Formal config rejects smoke LLM provider: {provider!r}")
    strict_stage = _requires_paper_facing_strictness(stage)
    if strict_stage and not provider:
        raise FormalConfigError("Formal config requires llm.provider (non-empty).")
    if _is_placeholder(model) and not allow_placeholders:
        raise FormalConfigError("Formal config requires llm.model (non-placeholder).")
    if _is_placeholder(model_version) and not allow_placeholders:
        raise FormalConfigError("Formal config requires llm.model_version (non-placeholder).")
    if strict_stage and not allow_placeholders and not api_key_env:
        raise FormalConfigError("Formal config requires llm.api_key_env (non-empty).")
    if strict_stage and not allow_placeholders:
        for field in ("temperature", "top_p", "max_tokens", "seed"):
            if field not in llm:
                raise FormalConfigError(f"Formal config requires llm.{field} to be set explicitly.")
        if "enable_thinking" in llm:
            _validate_exact_bool(llm["enable_thinking"], field="llm.enable_thinking")
        _validate_llm_numeric_params_for_formal(llm)
        smoke_tokens = ("heuristic", "smoke", "fixture", "mock", "fake")
        model_lower = model.lower()
        if any(token in model_lower for token in smoke_tokens):
            raise FormalConfigError(f"Formal config rejects smoke/heuristic LLM model name: {model!r}")

def validate_dense_config_for_real_run(
    config: dict[str, Any],
    *,
    allow_placeholders: bool = False,
    validation_stage: str = "formal",
    validate_index_integrity: bool = True,
) -> None:
    dense = config.get("dense_rag") or {}
    if not isinstance(dense, dict) or not dense:
        raise FormalConfigError("Formal Dense RAG requires explicit dense_rag block.")
    backend = _validate_exact_nonempty_string(
        dense["backend"] if "backend" in dense else "smoke",
        field="dense_rag.backend",
        allow_placeholders=allow_placeholders,
    ).casefold().replace("-", "_")
    if backend in SMOKE_EMBEDDING_BACKENDS:
        raise FormalConfigError(f"Formal Dense RAG rejects smoke/hash backend: {backend!r}")
    _validate_exact_bool(dense.get("reject_smoke", False), field="dense_rag.reject_smoke", required=True)
    for field in ("model_name", "model_version"):
        value = _validate_exact_nonempty_string(
            dense.get(field),
            field=f"dense_rag.{field}",
            allow_placeholders=allow_placeholders,
        )
        if _is_placeholder(value) and not allow_placeholders:
            raise FormalConfigError(f"Formal Dense RAG requires dense_rag.{field} (non-placeholder).")
    dim = dense.get("dimension", dense.get("dim"))
    validated_dim: int | None = None
    if not (allow_placeholders and _is_placeholder(dim)):
        validated_dim = _validate_positive_dimension(
            dim, allow_placeholders=allow_placeholders, field="dense_rag.dimension"
        )
    if "batch_size" in dense and not allow_placeholders:
        _validate_exact_int(dense["batch_size"], field="dense_rag.batch_size", minimum=1)
    if "normalize_embeddings" not in dense and not allow_placeholders:
        raise FormalConfigError("Formal Dense RAG requires dense_rag.normalize_embeddings to be set.")
    elif "normalize_embeddings" in dense and not allow_placeholders:
        _validate_exact_bool(
            dense.get("normalize_embeddings"),
            field="dense_rag.normalize_embeddings",
        )
    if "allow_index_rebuild" in dense and not allow_placeholders:
        _validate_exact_bool(
            dense.get("allow_index_rebuild"),
            field="dense_rag.allow_index_rebuild",
            required=False,
        )
    index_path = dense.get("index_path")
    if index_path:
        index_path = _validate_exact_nonempty_string(
            index_path,
            field="dense_rag.index_path",
            allow_placeholders=allow_placeholders,
        )
    if not index_path or (_is_placeholder(index_path) and not allow_placeholders):
        raise FormalConfigError("Formal Dense RAG requires non-placeholder dense_rag.index_path.")
    if validation_stage not in {"template", "index_build_candidate"} and not allow_placeholders:
        path = _resolve_repo_path(index_path)
        if not path.exists():
            raise FormalConfigError(f"Formal Dense RAG index_path does not exist: {index_path}")
        if _requires_paper_facing_strictness(validation_stage) and validate_index_integrity:
            from external_baselines.retrieval.dense_index import (
                DenseIndexError,
                validate_dense_index_integrity_for_freeze,
            )

            try:
                validate_dense_index_integrity_for_freeze(
                    path,
                    expected_model_name=_validate_exact_nonempty_string(
                        dense.get("model_name"),
                        field="dense_rag.model_name",
                    ),
                    expected_model_version=_validate_exact_nonempty_string(
                        dense.get("model_version"),
                        field="dense_rag.model_version",
                    ),
                    expected_backend=backend,
                    expected_dimension=validated_dim,
                    expected_corpus_checksum=str(config.get("corpus_checksum") or "") or None,
                    expected_normalize_embeddings=_validate_exact_bool(
                        dense.get("normalize_embeddings"),
                        field="dense_rag.normalize_embeddings",
                    ),
                )
            except DenseIndexError as exc:
                raise FormalConfigError(str(exc)) from exc


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
    lexical = _validate_exact_nonempty_string(
        hybrid["lexical_method"] if "lexical_method" in hybrid else "bm25",
        field="hybrid_rag.lexical_method",
        allow_placeholders=allow_placeholders,
    ).lower()
    if lexical != "bm25":
        raise FormalConfigError(f"Formal Hybrid RAG requires lexical_method=bm25 (got {lexical!r}).")
    if dense.get("backend"):
        backend_source = dense.get("backend")
        backend_field = "dense_rag.backend"
    elif hybrid.get("dense_method"):
        backend_source = hybrid.get("dense_method")
        backend_field = "hybrid_rag.dense_method"
    else:
        backend_source = "smoke"
        backend_field = "hybrid_rag.dense_method"
    backend = _validate_exact_nonempty_string(
        backend_source,
        field=backend_field,
        allow_placeholders=allow_placeholders,
    ).casefold().replace("-", "_")
    if backend in SMOKE_EMBEDDING_BACKENDS:
        raise FormalConfigError(f"Formal Hybrid RAG rejects smoke/hash dense backend: {backend!r}")
    hybrid_reject = hybrid.get("reject_smoke", dense.get("reject_smoke", False))
    _validate_exact_bool(hybrid_reject, field="hybrid_rag.reject_smoke", required=True)
    if dense.get("model_name"):
        model_name = _validate_exact_nonempty_string(
            dense.get("model_name"),
            field="dense_rag.model_name",
            allow_placeholders=allow_placeholders,
        )
    else:
        model_name = _validate_exact_nonempty_string(
            hybrid.get("dense_model_name"),
            field="hybrid_rag.dense_model_name",
            allow_placeholders=allow_placeholders,
        )
    if dense.get("model_version"):
        model_version = _validate_exact_nonempty_string(
            dense.get("model_version"),
            field="dense_rag.model_version",
            allow_placeholders=allow_placeholders,
        )
    else:
        model_version = _validate_exact_nonempty_string(
            hybrid.get("dense_model_version"),
            field="hybrid_rag.dense_model_version",
            allow_placeholders=allow_placeholders,
        )
    for field, value in (("model_name", model_name), ("model_version", model_version)):
        if _is_placeholder(value) and not allow_placeholders:
            raise FormalConfigError(f"Formal Hybrid RAG requires dense {field} (non-placeholder).")
    dim = dense.get("dimension", hybrid.get("dimension"))
    if not (allow_placeholders and _is_placeholder(dim)):
        _validate_positive_dimension(dim, allow_placeholders=allow_placeholders, field="hybrid_rag.dimension")
    top_k_raw = hybrid.get("top_k", (config.get("retrieval") or {}).get("top_k", 5))
    top_k = _validate_exact_int(top_k_raw, field="hybrid_rag.top_k", minimum=1)
    candidate_pool_raw = hybrid.get("candidate_pool", top_k)
    candidate_pool = _validate_exact_int(
        candidate_pool_raw,
        field="hybrid_rag.candidate_pool",
        minimum=top_k,
    )
    _validate_exact_number(
        hybrid.get("rrf_k", 0),
        field="hybrid_rag.rrf_k",
        minimum=0,
        minimum_inclusive=False,
    )
    _validate_exact_number(
        hybrid.get("lexical_weight", 0),
        field="hybrid_rag.lexical_weight",
        minimum=0,
        minimum_inclusive=False,
    )
    _validate_exact_number(
        hybrid.get("dense_weight", 0),
        field="hybrid_rag.dense_weight",
        minimum=0,
        minimum_inclusive=False,
    )
    if candidate_pool < top_k:
        raise FormalConfigError("Formal Hybrid RAG requires candidate_pool >= top_k.")
    index_path = dense.get("index_path") or hybrid.get("index_path")
    if index_path:
        index_path = _validate_exact_nonempty_string(
            index_path,
            field="dense_rag.index_path" if dense.get("index_path") else "hybrid_rag.index_path",
            allow_placeholders=allow_placeholders,
        )
    if not index_path or (_is_placeholder(index_path) and not allow_placeholders):
        raise FormalConfigError("Formal Hybrid RAG requires non-placeholder dense index_path.")
    if validation_stage not in {"template", "index_build_candidate"} and not allow_placeholders:
        path = _resolve_repo_path(index_path)
        if not path.exists():
            raise FormalConfigError(f"Formal Hybrid RAG index_path does not exist: {index_path}")
    if dense_config is not None:
        other = dense_config.get("dense_rag") or {}
        other_path = other.get("index_path")
        if other_path:
            other_path = _validate_exact_nonempty_string(
                other_path,
                field="dense_rag.index_path",
                allow_placeholders=allow_placeholders,
            )
        if other_path and index_path and str(Path(other_path)) != str(Path(index_path)):
            raise FormalConfigError("Hybrid and Dense must share the same dense index_path.")
        for field in ("backend", "model_name", "model_version"):
            left = dense.get(field) or hybrid.get(f"dense_{field}" if field != "backend" else "dense_method")
            right = other.get(field)
            if left and right and str(left) != str(right):
                raise FormalConfigError(f"Hybrid/Dense embedding {field} mismatch.")
        left_normalize = dense.get("normalize_embeddings", hybrid.get("normalize_embeddings"))
        right_normalize = other.get("normalize_embeddings")
        if (
            _requires_paper_facing_strictness(validation_stage)
            and left_normalize is not None
            and right_normalize is not None
        ):
            left_bool = _validate_exact_bool(left_normalize, field="hybrid_rag.normalize_embeddings")
            right_bool = _validate_exact_bool(right_normalize, field="dense_rag.normalize_embeddings")
            if left_bool is not right_bool:
                raise FormalConfigError("Hybrid/Dense normalize_embeddings mismatch.")
        left_cs = dense.get("index_checksum") or config.get("dense_index_checksum")
        right_cs = other.get("index_checksum") or dense_config.get("dense_index_checksum")
        if _requires_paper_facing_strictness(validation_stage) and left_cs and right_cs and str(left_cs) != str(right_cs):
            raise FormalConfigError("Hybrid/Dense index_checksum mismatch.")


def validate_ekell_vector_for_formal(
    config: dict[str, Any],
    *,
    allow_placeholders: bool = False,
    validation_stage: str = "formal",
    validate_index_integrity: bool = True,
) -> None:
    vector = config.get("ekell_vector") or (config.get("ekell_style") or {}).get("vector") or {}
    if not isinstance(vector, dict) or not vector:
        raise FormalConfigError("Formal E-KELL config requires explicit ekell_vector block.")
    backend = _validate_exact_nonempty_string(
        vector["backend"] if "backend" in vector else "smoke",
        field="ekell_vector.backend",
        allow_placeholders=allow_placeholders,
    ).casefold().replace("-", "_")
    if backend in SMOKE_EMBEDDING_BACKENDS:
        raise FormalConfigError(f"Formal E-KELL rejects smoke/hash backend: {backend!r}")
    _validate_exact_bool(vector.get("reject_smoke", False), field="ekell_vector.reject_smoke", required=True)
    for field in ("model_name", "model_version"):
        value = _validate_exact_nonempty_string(
            vector.get(field),
            field=f"ekell_vector.{field}",
            allow_placeholders=allow_placeholders,
        )
        if _is_placeholder(value) and not allow_placeholders:
            raise FormalConfigError(f"Formal E-KELL requires ekell_vector.{field} (non-placeholder).")
    validated_dimension = _validate_positive_dimension(
        vector.get("dimension"),
        allow_placeholders=allow_placeholders,
        field="ekell_vector.dimension",
    )
    if "normalize_embeddings" not in vector and not allow_placeholders:
        raise FormalConfigError("Formal E-KELL requires ekell_vector.normalize_embeddings to be set.")
    elif "normalize_embeddings" in vector and not allow_placeholders:
        _validate_exact_bool(
            vector.get("normalize_embeddings"),
            field="ekell_vector.normalize_embeddings",
        )
    index_path = vector.get("index_path")
    if not index_path:
        if not allow_placeholders:
            raise FormalConfigError("Formal E-KELL requires a persisted directory index_path.")
    else:
        index_path = _validate_exact_nonempty_string(
            index_path,
            field="ekell_vector.index_path",
            allow_placeholders=allow_placeholders,
        )
        if _is_placeholder(index_path) and not allow_placeholders:
            raise FormalConfigError("Formal E-KELL requires a persisted directory index_path.")
    if index_path and _is_placeholder(index_path) and not allow_placeholders:
        raise FormalConfigError("Formal E-KELL requires a persisted directory index_path.")
    ekell_style = config.get("ekell_style") or {}
    if "prompt_dir" in ekell_style:
        _validate_exact_nonempty_string(
            ekell_style["prompt_dir"],
            field="ekell_style.prompt_dir",
            allow_placeholders=allow_placeholders,
        )
    if (
        not allow_placeholders
        and validation_stage != "index_build_candidate"
        and index_path
        and not _is_placeholder(index_path)
    ):
        path = _resolve_repo_path(index_path)
        if not path.exists():
            raise FormalConfigError(f"Formal E-KELL index_path does not exist: {index_path}")
        from external_baselines.ekell_style.vector_index import VectorIndex, VectorIndexError

        try:
            if _requires_paper_facing_strictness(validation_stage) and validate_index_integrity:
                VectorIndex.validate_directory_for_freeze(
                    path,
                    expected_backend=backend,
                    expected_model_name=_validate_exact_nonempty_string(
                        vector.get("model_name"),
                        field="ekell_vector.model_name",
                    ),
                    expected_model_version=_validate_exact_nonempty_string(
                        vector.get("model_version"),
                        field="ekell_vector.model_version",
                    ),
                    expected_dimension=validated_dimension,
                    expected_kg_checksum=str(config.get("kg_checksum") or "") or None,
                    expected_corpus_checksum=str(config.get("corpus_checksum") or "") or None,
                    expected_normalize_embeddings=_validate_exact_bool(
                        vector.get("normalize_embeddings"),
                        field="ekell_vector.normalize_embeddings",
                    ),
                )
            else:
                VectorIndex.load_directory(
                    path,
                    expected_backend=backend,
                    expected_model_name=_validate_exact_nonempty_string(
                        vector.get("model_name"),
                        field="ekell_vector.model_name",
                    ),
                    expected_model_version=_validate_exact_nonempty_string(
                        vector.get("model_version"),
                        field="ekell_vector.model_version",
                    ),
                    expected_dimension=validated_dimension,
                    require_real_embedding=True,
                )
        except VectorIndexError as exc:
            raise FormalConfigError(str(exc)) from exc


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
    model_run = _validate_exact_bool(
        config.get("paper_fidelity_model_run", False),
        field="paper_fidelity_model_run",
        allow_placeholders=allow_placeholders,
    )
    if model_run is True:
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
    validate_index_integrity: bool = True,
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

    raw_paper_final = config.get("paper_final", False)
    if type(raw_paper_final) is not bool:
        raise FormalConfigError("paper_final must be an exact boolean")
    paper_final = raw_paper_final
    formal_ids = FORMAL_METHOD_IDS | COMPARISON_FORMAL_METHOD_IDS
    if require_formal or paper_final:
        if "llm" in config and _llm_block(config):
            validate_llm_for_formal(
                config,
                allow_placeholders=allow_placeholders,
                validation_stage=validation_stage,
            )
        elif mid in formal_ids and not allow_placeholders:
            raise FormalConfigError(
                f"Formal method {mid!r} requires llm block in merged config (from shared_model_config)."
            )
        if mid in FORMAL_EKELL_METHODS:
            validate_ekell_vector_for_formal(
                config,
                allow_placeholders=allow_placeholders,
                validation_stage=validation_stage,
                validate_index_integrity=validate_index_integrity,
            )
        if mid == "dense_rag":
            validate_dense_config_for_real_run(
                config,
                allow_placeholders=allow_placeholders,
                validation_stage=validation_stage,
                validate_index_integrity=validate_index_integrity,
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
                raw = enhanced.get(flag, False)
                if type(raw) is not bool:
                    raise FormalConfigError(
                        f"ekell_style.{flag} must be an exact boolean"
                    )
                if raw is True:
                    raise FormalConfigError(
                        f"Controlled E-KELL forbids enhanced hook "
                        f"ekell_style.{flag}=true"
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
    if stage not in VALIDATION_STAGES:
        raise FormalConfigError(f"Unknown validation_stage={validation_stage!r}")
    strict_stage = _requires_paper_facing_strictness(stage)
    requires_existing_freeze = _requires_existing_freeze(stage)
    allow_placeholders = stage == "template" or allow_placeholders
    if stage == "template":
        allow_placeholders = True

    try:
        manifest = load_experiment_manifest(path)
    except ValueError as exc:
        raise FormalConfigError(str(exc)) from exc
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

    if strict_stage:
        for key in (
            "experiment_id",
            "schema_version",
            "track",
            "run_mode",
            "base_config",
            "shared_model_config",
        ):
            if key not in raw:
                errors.append(f"formal manifest requires explicit {key}.")
                continue
            try:
                value = _validate_exact_nonempty_string(
                    raw.get(key),
                    field=key,
                    allow_placeholders=allow_placeholders,
                )
            except FormalConfigError as exc:
                errors.append(str(exc))
                continue
            if key == "schema_version" and value != "firebench-interop-v1":
                errors.append(
                    "schema_version must be exactly 'firebench-interop-v1' "
                    f"(got {value!r})."
                )
        if requires_existing_freeze and "freeze_manifest" not in raw:
            errors.append("formal manifest requires explicit freeze_manifest.")
        for key in ("output", "run_manifest"):
            if key in raw:
                try:
                    _validate_exact_nonempty_string(
                        raw.get(key),
                        field=key,
                        allow_placeholders=allow_placeholders,
                    )
                except FormalConfigError as exc:
                    errors.append(str(exc))

    if strict_stage and "run_mode" not in raw:
        run_mode = ""
    else:
        run_mode = _validate_exact_nonempty_string(
            raw["run_mode"] if "run_mode" in raw else "formal",
            field="run_mode",
            allow_placeholders=allow_placeholders,
        ).lower()
    if run_mode != "formal":
        errors.append(f"run_mode must be 'formal' for paper-facing manifest (got {run_mode!r})")

    if not isinstance(raw.get("paper_final"), bool):
        errors.append("paper_final must be an exact boolean for formal experiment manifest.")
    elif raw.get("paper_final") is not True:
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
        flag = raw.get(key, False)
        if type(flag) is not bool or flag is not True:
            errors.append(f"{key} must be true for formal manifest.")

    freeze_status = _validate_exact_nonempty_string(
        raw.get("freeze_status"),
        field="freeze_status",
        allow_placeholders=allow_placeholders,
    ).lower()
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
    elif stage == "freeze_candidate":
        if freeze_status != "provisional":
            errors.append(
                "freeze_candidate validation requires freeze_status=provisional "
                f"(got {freeze_status!r})."
            )
    else:  # formal
        if freeze_status != "frozen":
            errors.append(f"formal validation requires freeze_status=frozen (got {freeze_status!r}).")
        freeze_path = raw.get("freeze_manifest")
        if not freeze_path or _is_placeholder(freeze_path):
            errors.append("formal validation requires freeze_manifest path.")
        else:
            freeze_path = _validate_exact_nonempty_string(
                freeze_path,
                field="freeze_manifest",
                allow_placeholders=allow_placeholders,
            )
            freeze_file = _resolve_repo_path(freeze_path)
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

    if strict_stage and method_set_name == "comparison_suite":
        bundle_candidate = str(runtime_bundle_path) if runtime_bundle_path is not None else raw.get("bundle")
        try:
            bundle_path = _validate_exact_nonempty_string(
                bundle_candidate,
                field="bundle",
                allow_placeholders=allow_placeholders,
            )
        except FormalConfigError as exc:
            errors.append(str(exc))
        else:
            if _is_placeholder(bundle_path):
                errors.append("Formal comparison suite requires non-placeholder Runner Bundle path.")
            else:
                try:
                    from external_baselines.interop.bundle import (
                        load_runner_bundle,
                        validate_formal_bundle_aggregate_checksum,
                    )

                    validate_formal_bundle_aggregate_checksum(
                        load_runner_bundle(bundle_path, formal=True)
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Formal Runner Bundle authority validation failed: {exc}")

    shared = (
        _validate_exact_nonempty_string(
            raw.get("shared_model_config"),
            field="shared_model_config",
            allow_placeholders=allow_placeholders,
        )
        if raw.get("shared_model_config") is not None
        else ""
    )
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

    declared_main = [
        canonicalize_method_id(_validate_exact_nonempty_string(m, field="main_table_methods[]"))
        for m in (raw.get("main_table_methods") or [])
    ]
    declared_pf = [
        canonicalize_method_id(_validate_exact_nonempty_string(m, field="paper_fidelity_methods[]"))
        for m in (raw.get("paper_fidelity_methods") or [])
    ]
    declared_supp = [
        canonicalize_method_id(_validate_exact_nonempty_string(m, field="supplemental_methods[]"))
        for m in (raw.get("supplemental_methods") or [])
    ]
    declared_comparison: list[str] = []
    raw_comparison_methods = raw.get("comparison_suite_methods")
    if strict_stage and method_set_name == "comparison_suite":
        if "comparison_suite_methods" not in raw:
            errors.append("formal comparison suite requires explicit comparison_suite_methods.")
        elif raw_comparison_methods is None or not isinstance(raw_comparison_methods, list):
            errors.append("comparison_suite_methods must be a YAML list.")
        elif not raw_comparison_methods:
            errors.append("comparison_suite_methods must be a non-empty YAML list.")
        else:
            for item in raw_comparison_methods:
                try:
                    exact = _validate_exact_nonempty_string(item, field="comparison_suite_methods[]")
                except FormalConfigError as exc:
                    errors.append(str(exc))
                    continue
                canonical = canonicalize_method_id(exact)
                if canonical != exact:
                    errors.append(f"comparison_suite_methods must use canonical method IDs, not alias {exact!r}.")
                declared_comparison.append(canonical)
    else:
        declared_comparison = [
            canonicalize_method_id(_validate_exact_nonempty_string(m, field="comparison_suite_methods[]"))
            for m in (raw_comparison_methods or [])
        ]

    if method_set_name == "comparison_suite" and (strict_stage or declared_comparison):
        if declared_comparison != COMPARISON_SUITE_EXACT:
            errors.append(
                "comparison_suite_methods must be exactly "
                f"{COMPARISON_SUITE_EXACT} (got {declared_comparison})"
            )
        duplicates = sorted({m for m in declared_comparison if declared_comparison.count(m) > 1})
        if duplicates:
            errors.append(f"comparison_suite_methods contains duplicates: {duplicates}.")
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
    entries_by_id: dict[str, dict[str, Any]] = {}
    merged_by_id: dict[str, dict[str, Any]] = {}
    for entry in methods:
        if isinstance(entry, str):
            entry = {"method_id": entry}
        if not isinstance(entry, dict):
            errors.append(f"Invalid method entry: {entry!r}")
            continue
        mid = canonicalize_method_id(
            _validate_exact_nonempty_string(
                entry.get("method_id"),
                field="methods[].method_id",
                allow_placeholders=allow_placeholders,
            )
        )
        if mid in seen_ids:
            errors.append(f"Duplicate method_id in manifest: {mid}")
        seen_ids.add(mid)
        if str(entry.get("method_id") or "") == "ekell_style_faithful":
            errors.append("Manifest must use canonical ekell_style_controlled_shared_llm, not faithful.")
        if mid in pf_expected and mid in declared_main:
            errors.append(f"paper-fidelity method {mid!r} listed in main_table_methods.")
        if mid in pf_expected and mid in declared_supp:
            errors.append(f"paper-fidelity method {mid!r} listed in supplemental_methods.")

        if "enabled" not in entry:
            enabled = True
        else:
            raw_enabled = entry.get("enabled")
            if type(raw_enabled) is not bool:
                errors.append(
                    f"methods entry for {mid!r}: enabled must be an exact YAML boolean"
                )
                continue
            enabled = raw_enabled
        entries_by_id.setdefault(mid, {**entry, "method_id": mid, "enabled": enabled})
        should_validate = enabled and mid in required_formal_ids
        if method_set_name == "comparison_suite" and mid in COMPARISON_FORMAL_METHOD_IDS:
            should_validate = True
        if should_validate:
            cfg_raw = entry["config"] if "config" in entry else entry.get("method_config")
            cfg_path = (
                _validate_exact_nonempty_string(
                    cfg_raw,
                    field=f"{mid}.config",
                    allow_placeholders=allow_placeholders,
                )
                if cfg_raw is not None
                else ""
            )
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

    if strict_stage and method_set_name == "comparison_suite":
        missing_entries = [m for m in COMPARISON_SUITE_EXACT if m not in entries_by_id]
        if missing_entries:
            errors.append(f"comparison suite methods missing method entries: {missing_entries}.")
        for mid in COMPARISON_SUITE_EXACT:
            entry = entries_by_id.get(mid)
            if entry is not None and entry.get("enabled", True) is not True:
                errors.append(f"Required comparison suite method entry disabled: {mid}.")
        for mid, entry in entries_by_id.items():
            if mid not in COMPARISON_SUITE_EXACT and entry.get("enabled", True) is True:
                errors.append(
                    "Non-comparison method entry must be disabled during Formal "
                    f"comparison suite validation: {mid}."
                )
        try:
            resolved_entries = enabled_methods(manifest, method_set="comparison_suite")
            resolved_ids = [entry["method_id"] for entry in resolved_entries]
        except Exception as exc:  # noqa: BLE001
            errors.append(
                "comparison suite resolver failed to select the exact declared method set: "
                f"{exc}"
            )
        else:
            if resolved_ids != COMPARISON_SUITE_EXACT:
                errors.append(
                    "comparison suite resolver must select exactly "
                    f"{COMPARISON_SUITE_EXACT} in declared order (got {resolved_ids})."
                )

    enabled_main = [
        canonicalize_method_id(_validate_exact_nonempty_string(e.get("method_id"), field="methods[].method_id"))
        for e in methods
        if isinstance(e, dict) and e.get("enabled", True)
        and canonicalize_method_id(_validate_exact_nonempty_string(e.get("method_id"), field="methods[].method_id")) in expected_main
    ]
    if enabled_main and sorted(declared_main) != sorted(enabled_main):
        errors.append(
            f"main_table_methods must list enabled main-table methods {sorted(enabled_main)} "
            f"(got {declared_main})"
        )

    enabled_pf = [
        canonicalize_method_id(_validate_exact_nonempty_string(e.get("method_id"), field="methods[].method_id"))
        for e in methods
        if isinstance(e, dict) and e.get("enabled", True)
        and canonicalize_method_id(_validate_exact_nonempty_string(e.get("method_id"), field="methods[].method_id")) in pf_expected
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
