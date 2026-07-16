from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = ROOT / "tests/fixtures/deepeval_handoff"


@pytest.fixture
def fixture_main_repo() -> Path:
    return FIXTURE_ROOT / "main_project"


@pytest.fixture
def fixture_run(tmp_path: Path) -> Path:
    target = tmp_path / "formal_run"
    shutil.copytree(FIXTURE_ROOT / "formal_run", target)
    return target


@pytest.fixture
def exported_handoff(fixture_run: Path, fixture_main_repo: Path, tmp_path: Path) -> Path:
    from external_baselines.interop.deepeval_handoff.exporter import export_handoff

    target = tmp_path / "handoff"
    export_handoff(
        formal_run_root=fixture_run,
        main_repo=fixture_main_repo,
        output=target,
        top_k=1,
        allow_development_source=True,
    )
    return target


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
