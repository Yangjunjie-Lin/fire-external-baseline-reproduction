"""Cross-method fairness for controlled comparison suites."""

from __future__ import annotations

from typing import Any

FAIRNESS_FIELDS = (
    "provider",
    "model",
    "model_version",
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
)

EMBEDDING_IDENTITY_FIELDS = (
    "backend",
    "model_name",
    "model_version",
    "dimension",
)


class CrossMethodFairnessError(ValueError):
    """Raised when paper-final methods do not share required settings."""


def _llm_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm") or {}
    return {field: llm.get(field) for field in FAIRNESS_FIELDS}


def _embedding_identity(config: dict[str, Any], method_id: str) -> dict[str, Any] | None:
    mid = method_id.lower()
    if mid == "dense_rag":
        block = config.get("dense_rag") or {}
    elif mid == "hybrid_rag":
        block = dict(config.get("dense_rag") or {})
        hybrid = config.get("hybrid_rag") or {}
        block.setdefault("backend", hybrid.get("dense_method"))
        block.setdefault("model_name", hybrid.get("dense_model_name"))
        block.setdefault("model_version", hybrid.get("dense_model_version"))
        block.setdefault("dimension", hybrid.get("dimension"))
    elif mid in {"ekell_style_controlled_shared_llm", "ekell_style_paper_fidelity"}:
        block = config.get("ekell_vector") or {}
    else:
        return None
    dim = block.get("dimension", block.get("dim"))
    return {
        "backend": block.get("backend"),
        "model_name": block.get("model_name") or block.get("dense_model_name"),
        "model_version": block.get("model_version") or block.get("dense_model_version"),
        "dimension": int(dim) if dim not in (None, "") else None,
    }


def validate_cross_method_fairness(
    method_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require identical generation settings and shared embedding identity where applicable."""
    paper_configs = {
        method: config
        for method, config in method_configs.items()
        if bool(config.get("paper_final")) or bool(config.get("enforce_cross_method_fairness"))
    }
    if len(paper_configs) < 2:
        return {
            "ok": True,
            "checked": list(paper_configs),
            "settings": None,
            "dataset": None,
            "shared_llm": True,
            "shared_bundle": True,
            "shared_corpus": True,
            "shared_embedding_identity": True,
            "hybrid_reuses_dense_index": True,
        }

    snapshots = {method: _llm_snapshot(config) for method, config in paper_configs.items()}
    dataset_snapshots = {
        method: {
            "corpus_checksum": (config.get("paths") or {}).get("corpus_checksum")
            or config.get("corpus_checksum"),
            "scenario_checksum": (config.get("paths") or {}).get("scenario_checksum")
            or config.get("scenario_checksum"),
            "bundle_checksum": config.get("bundle_checksum"),
            "output_schema_version": config.get("schema_version") or "firebench-interop-v1",
        }
        for method, config in paper_configs.items()
    }
    reference_method = next(iter(snapshots))
    reference = snapshots[reference_method]
    mismatches = {
        method: {
            field: {"expected": reference[field], "actual": values[field]}
            for field in FAIRNESS_FIELDS
            if values[field] != reference[field]
        }
        for method, values in snapshots.items()
        if values != reference
    }
    mismatches = {method: values for method, values in mismatches.items() if values}
    if mismatches:
        raise CrossMethodFairnessError(
            f"paper_final model settings differ from {reference_method}: {mismatches}"
        )

    emb_ids = {
        mid: _embedding_identity(cfg, mid)
        for mid, cfg in paper_configs.items()
        if _embedding_identity(cfg, mid) is not None
    }
    shared_embedding = True
    if len(emb_ids) >= 2:
        ref_mid = next(iter(emb_ids))
        ref_emb = emb_ids[ref_mid]
        for mid, emb in emb_ids.items():
            for field in EMBEDDING_IDENTITY_FIELDS:
                if emb.get(field) != ref_emb.get(field):
                    # Ignore placeholder-ish None mismatches only when both missing.
                    if emb.get(field) in (None, "") and ref_emb.get(field) in (None, ""):
                        continue
                    shared_embedding = False
                    raise CrossMethodFairnessError(
                        f"embedding identity mismatch between {ref_mid} and {mid} on {field}: "
                        f"{ref_emb.get(field)!r} vs {emb.get(field)!r}"
                    )

    hybrid_reuses = True
    if "hybrid_rag" in paper_configs and "dense_rag" in paper_configs:
        dense_cfg = paper_configs["dense_rag"].get("dense_rag") or {}
        hybrid_cfg = paper_configs["hybrid_rag"].get("dense_rag") or {}
        hybrid_top = paper_configs["hybrid_rag"].get("hybrid_rag") or {}
        dense_path = dense_cfg.get("index_path")
        hybrid_path = hybrid_cfg.get("index_path") or hybrid_top.get("dense_index_path")
        dense_checksum = paper_configs["dense_rag"].get("dense_index_checksum") or dense_cfg.get("index_checksum")
        hybrid_checksum = (
            paper_configs["hybrid_rag"].get("dense_index_checksum")
            or hybrid_cfg.get("index_checksum")
            or hybrid_top.get("dense_index_checksum")
        )
        if dense_path and hybrid_path and str(dense_path) != str(hybrid_path):
            # Paths may differ by placeholder tokens; checksums must match when both present.
            if dense_checksum and hybrid_checksum and dense_checksum != hybrid_checksum:
                hybrid_reuses = False
                raise CrossMethodFairnessError(
                    "hybrid_rag must reuse dense_rag index checksum for fair comparison."
                )
        if dense_checksum and hybrid_checksum and dense_checksum != hybrid_checksum:
            hybrid_reuses = False
            raise CrossMethodFairnessError(
                "hybrid_rag dense_index_checksum does not match dense_rag index checksum."
            )

    bundle_values = {v.get("bundle_checksum") for v in dataset_snapshots.values()}
    corpus_values = {v.get("corpus_checksum") for v in dataset_snapshots.values()}
    shared_bundle = len(bundle_values) == 1
    shared_corpus = len(corpus_values) == 1

    return {
        "ok": True,
        "checked": list(paper_configs),
        "settings": reference,
        "dataset": dataset_snapshots[reference_method],
        "dataset_snapshots": dataset_snapshots,
        "shared_llm": True,
        "shared_bundle": shared_bundle,
        "shared_corpus": shared_corpus,
        "shared_embedding_identity": shared_embedding,
        "hybrid_reuses_dense_index": hybrid_reuses,
        "note": "Architecture-specific prompts/retrieval/KG modules may differ by design.",
    }
