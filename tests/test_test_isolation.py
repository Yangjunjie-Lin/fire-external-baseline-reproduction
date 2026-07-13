"""Guard against pytest tests calling other test functions."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_test_function_directly_calls_another_test_function():
    violations: list[str] = []
    for path in (ROOT / "tests").rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id.startswith("test_"):
                        violations.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} "
                            f"{node.name} calls {child.func.id}"
                        )
    assert not violations, "\n".join(violations)
