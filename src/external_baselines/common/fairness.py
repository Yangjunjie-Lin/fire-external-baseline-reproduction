from __future__ import annotations

"""Cross-method model-setting checks for formal comparisons."""

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

DATASET_FAIRNESS_FIELDS = (
    "corpus_checksum",
    "scenario_checksum",
    "bundle_checksum",
    "output_schema_version",
)


class CrossMethodFairnessError(ValueError):
    """Raised when paper-final methods do not share model settings."""


def validate_cross_method_fairness(
    method_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require identical generation settings across paper-final / controlled methods."""
    paper_configs = {
        method: config
        for method, config in method_configs.items()
        if bool(config.get("paper_final")) or bool(config.get("enforce_cross_method_fairness"))
    }
    if len(paper_configs) < 2:
        return {"ok": True, "checked": list(paper_configs), "settings": None, "dataset": None}

    snapshots = {
        method: {
            field: (config.get("llm") or {}).get(field)
            for field in FAIRNESS_FIELDS
        }
        for method, config in paper_configs.items()
    }
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
    return {
        "ok": True,
        "checked": list(paper_configs),
        "settings": reference,
        "dataset": dataset_snapshots[reference_method],
        "dataset_snapshots": dataset_snapshots,
        "note": "Architecture-specific prompts/retrieval/KG modules may differ by design.",
    }
