from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baselines.interop.deepeval_handoff.exporter import HandoffExportError, export_handoff


def test_failed_replace_preserves_existing_handoff_and_leaves_no_temp(
    fixture_run: Path,
    fixture_main_repo: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "handoff"
    target.mkdir()
    marker = target / "valid.marker"
    marker.write_text("keep", encoding="utf-8")
    prediction = fixture_run / "predictions/bm25_rag.jsonl"
    record = json.loads(prediction.read_text(encoding="utf-8"))
    record["method_metadata"]["retrieved_contexts"] = []
    prediction.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(HandoffExportError, match="staged_handoff_validation_failed"):
        export_handoff(
            formal_run_root=fixture_run,
            main_repo=fixture_main_repo,
            output=target,
            allow_development_source=True,
            replace_existing=True,
        )
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".handoff.tmp_*"))
