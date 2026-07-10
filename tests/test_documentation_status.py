"""Documentation and status consistency tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from external_baselines.method_registry import main_table_methods  # noqa: E402

FORBIDDEN_CLAIMS = (
    "fully reproduced",
    "official reproduction",
    "experimentally validated",
    "paper_ready=true",
    "paper-final results",
)


def test_status_docs_match_registry():
    status = (ROOT / "docs/status/current_project_status.md").read_text(encoding="utf-8")
    for mid in main_table_methods():
        assert mid in status


def test_readiness_false_for_unrun_experiments():
    text = (ROOT / "docs/status/readiness_summary.md").read_text(encoding="utf-8").lower()
    for gate in (
        "real_shared_llm_run",
        "real_chatglm_run",
        "paper_ready",
        "cross_repository_interop_verified",
    ):
        assert gate in text
        idx = text.index(gate)
        row = text[idx : idx + 80]
        assert "false" in row


def test_no_unsupported_claims():
    for path in (ROOT / "README.md", ROOT / "docs/status/current_project_status.md"):
        text = path.read_text(encoding="utf-8").lower()
        for claim in FORBIDDEN_CLAIMS:
            if claim in text:
                for line in text.splitlines():
                    if claim in line and not any(
                        neg in line for neg in ("≠", "not ", "without ", "never ", "forbidden", "do not")
                    ):
                        pytest.fail(f"{claim!r} found as affirmative claim in {path.name}: {line.strip()}")


def test_documented_paths_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for match in re.findall(r"`(docs/[^`]+)`", readme):
        assert (ROOT / match).is_file(), match
    assert (ROOT / "configs/experiments/controlled_main_table_v1.yaml.example").is_file()


def test_no_faithful_config_in_formal_manifest():
    text = (ROOT / "configs/experiments/controlled_main_table_v1.yaml.example").read_text(encoding="utf-8")
    assert "ekell_style_faithful_v1.yaml" not in text


def test_matrix_status_values_are_valid():
    matrix = json.loads((ROOT / "docs/fidelity/ekell_paper_to_code_matrix.json").read_text(encoding="utf-8"))
    allowed = {
        "implemented",
        "implemented_but_not_empirically_run",
        "approximated",
        "substituted",
        "interface_only",
        "unavailable_publicly",
        "not_applicable",
    }
    for mod in matrix["modules"]:
        assert mod["status"] in allowed
        assert mod["empirically_validated"] is False
