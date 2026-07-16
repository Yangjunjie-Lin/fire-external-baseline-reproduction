from __future__ import annotations

import json
from pathlib import Path

from external_baselines.interop.deepeval_handoff.constants import FORBIDDEN_HANDOFF_KEYS

ROOT = Path(__file__).resolve().parents[3]


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | set().union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def test_handoff_code_has_no_evaluator_or_deepeval_imports() -> None:
    paths = [
        *(ROOT / "src/external_baselines/interop/deepeval_handoff").rglob("*.py"),
        ROOT / "scripts/export_deepeval_handoff.py",
        ROOT / "scripts/validate_deepeval_handoff.py",
    ]
    forbidden = (
        "import deepeval",
        "from deepeval",
        "import fire_agent_demo",
        "expected_output_builder",
        "gold loader",
    )
    offenders = []
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        offenders.extend(f"{path}:{marker}" for marker in forbidden if marker in text)
    assert offenders == []


def test_prediction_records_contain_no_gold_or_score_fields(exported_handoff: Path) -> None:
    for path in (exported_handoff / "predictions").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not (_keys(json.loads(line)) & FORBIDDEN_HANDOFF_KEYS)
