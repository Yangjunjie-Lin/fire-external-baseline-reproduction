from __future__ import annotations

import json
from pathlib import Path

from external_baselines.interop.deepeval_handoff.manifest import sha256_file


def test_manifest_freezes_protocol_safety_hashes_and_coverage(exported_handoff: Path) -> None:
    manifest = json.loads((exported_handoff / "handoff_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "deepeval-handoff-manifest-v1"
    assert manifest["evaluation_handoff"] == {
        "context_selection_policy": "original_rank_prefix",
        "deepeval_executed": False,
        "gold_accessed": False,
        "handoff_top_k": 1,
        "judge_called": False,
        "paid_api_used": False,
        "real_world_execution_allowed": False,
    }
    assert manifest["validation"]["publication_eligible"] is False
    assert manifest["source"]["development_artifact"] is True
    for method_id, entry in manifest["methods"].items():
        assert entry["sha256"] == sha256_file(exported_handoff / f"predictions/{method_id}.jsonl")
