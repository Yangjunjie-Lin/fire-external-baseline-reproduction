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


class CrossMethodFairnessError(ValueError):
    """Raised when paper-final methods do not share model settings."""


def validate_cross_method_fairness(
    method_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require identical generation settings across paper-final methods."""
    paper_configs = {
        method: config
        for method, config in method_configs.items()
        if bool(config.get("paper_final"))
    }
    if len(paper_configs) < 2:
        return {"ok": True, "checked": list(paper_configs), "settings": None}

    snapshots = {
        method: {
            field: (config.get("llm") or {}).get(field)
            for field in FAIRNESS_FIELDS
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
    }
