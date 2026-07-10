"""Compatibility facade — prefer importing from external_baselines.method_registry.

Physical path retained as method_registry.py for import stability.
This package re-exports the same single source of truth.
"""

from external_baselines.method_registry import *  # noqa: F403
from external_baselines.method_registry import METHOD_REGISTRY  # noqa: F401
