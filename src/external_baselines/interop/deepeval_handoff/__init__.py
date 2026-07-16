"""Prediction-only handoff for centralized evaluation by fire-agent-demo."""

from external_baselines.interop.deepeval_handoff.adapter import (
    adapt_firebench_interop_to_external_prediction,
)
from external_baselines.interop.deepeval_handoff.exporter import export_handoff
from external_baselines.interop.deepeval_handoff.validator import validate_handoff

__all__ = [
    "adapt_firebench_interop_to_external_prediction",
    "export_handoff",
    "validate_handoff",
]
