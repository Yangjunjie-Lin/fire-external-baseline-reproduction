from __future__ import annotations

from pathlib import Path

import pytest

from external_baselines.interop.deepeval_handoff.exporter import HandoffExportError, export_handoff


def test_development_source_is_rejected_by_default(
    fixture_run: Path,
    fixture_main_repo: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(HandoffExportError, match="formal_source_validation_failed"):
        export_handoff(
            formal_run_root=fixture_run,
            main_repo=fixture_main_repo,
            output=tmp_path / "handoff",
        )
    assert not (tmp_path / "handoff").exists()


def test_development_source_requires_explicit_marking(
    fixture_run: Path,
    fixture_main_repo: Path,
    tmp_path: Path,
) -> None:
    result = export_handoff(
        formal_run_root=fixture_run,
        main_repo=fixture_main_repo,
        output=tmp_path / "handoff",
        allow_development_source=True,
    )
    assert result["formal_source"] is False
    assert result["publication_eligible"] is False
    assert result["development_artifact"] is True
