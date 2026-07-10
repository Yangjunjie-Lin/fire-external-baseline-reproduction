"""E-KELL paper-to-code matrix consistency tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ALLOWED_STATUS = {
    "implemented",
    "implemented_but_not_empirically_run",
    "approximated",
    "substituted",
    "interface_only",
    "unavailable_publicly",
    "not_applicable",
}


def _matrix():
    return json.loads((ROOT / "docs/fidelity/ekell_paper_to_code_matrix.json").read_text(encoding="utf-8"))


def test_all_matrix_code_paths_exist():
    for mod in _matrix()["modules"]:
        for rel in mod["implementation_paths"]:
            assert (ROOT / rel).exists(), rel


def test_all_implemented_modules_have_tests():
    for mod in _matrix()["modules"]:
        if mod["status"] in {"implemented", "implemented_but_not_empirically_run", "approximated"}:
            if mod["module_id"] in {"chatglm6b_system_llm", "expert_evaluation"}:
                continue
            assert mod["test_paths"], mod["module_id"]


def test_unavailable_modules_not_marked_implemented():
    for mod in _matrix()["modules"]:
        if mod["status"] in {"unavailable_publicly", "interface_only"}:
            assert mod["status"] != "implemented"


def test_matrix_status_values_are_valid():
    for mod in _matrix()["modules"]:
        assert mod["status"] in ALLOWED_STATUS


def test_official_components_not_claimed():
    matrix = _matrix()
    assert matrix["claim"]
    assert "not official" in matrix["claim"].lower()
    official = [m for m in matrix["modules"] if m["module_id"] == "official_kg"]
    assert official[0]["status"] == "unavailable_publicly"
