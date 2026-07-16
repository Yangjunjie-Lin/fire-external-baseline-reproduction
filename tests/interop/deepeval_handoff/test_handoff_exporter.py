from __future__ import annotations

import json
from pathlib import Path

from external_baselines.interop.deepeval_handoff.exporter import export_handoff


def test_development_export_writes_one_file_per_method(
    fixture_run: Path,
    fixture_main_repo: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "handoff"
    result = export_handoff(
        formal_run_root=fixture_run,
        main_repo=fixture_main_repo,
        output=target,
        top_k=1,
        allow_development_source=True,
    )
    assert result["ok"] is True
    assert result["development_artifact"] is True
    direct = json.loads((target / "predictions/direct_llm.jsonl").read_text(encoding="utf-8"))
    bm25 = json.loads((target / "predictions/bm25_rag.jsonl").read_text(encoding="utf-8"))
    assert direct["system_name"] == "direct_llm"
    assert "retrieval_context" not in direct
    assert bm25["system_name"] == "bm25_rag"
    assert [item["text"] for item in bm25["retrieval_context"]] == ["First exact retrieved passage."]
    assert bm25["metadata"]["native_retrieval_context_count"] == 2
    assert bm25["metadata"]["retrieval_context_truncated_for_handoff"] is True
    assert (target / "validation_report.json").is_file()
    assert (target / "contract_provenance.json").is_file()
    assert "does not contain benchmark Gold" in (target / "README.md").read_text(encoding="utf-8")
