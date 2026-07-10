"""Independence: never import fire_agent_demo or target SAFE modules."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


FORBIDDEN_IMPORT_ROOTS = {
    "fire_agent_demo",
    "fireagent",
    "safe_router",
    "safety_checker",
}


def _iter_py_files():
    for path in SRC.rglob("*.py"):
        yield path


def test_no_fire_agent_demo_import():
    offenders = []
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_IMPORT_ROOTS or "fire_agent_demo" in alias.name:
                        offenders.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                root = mod.split(".")[0] if mod else ""
                if root in FORBIDDEN_IMPORT_ROOTS or "fire_agent_demo" in mod:
                    offenders.append(f"{path}: from {mod}")
    assert not offenders, "Forbidden imports found:\n" + "\n".join(offenders)


def test_no_target_module_string_calls_in_pipelines():
    """Pipelines must not call target-system module names as APIs."""
    banned = ["SAFE-Router", "SafetyChecker", "DynamicREG", "HITLGate"]
    hits = []
    for path in (SRC / "external_baselines").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Mentions in comments/docs strings about NOT using them are OK; flag executable-looking calls.
        for name in banned:
            if f"{name}(" in text or f"import {name}" in text:
                hits.append(f"{path}: {name}")
    assert not hits
