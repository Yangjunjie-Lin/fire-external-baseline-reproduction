from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from external_baselines.common.llm_client import llm_config_summary
from external_baselines.ekell_style.kg_loader import audit_corpus


def git_commit_if_available(repo_root: str | Path = ".") -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def build_run_manifest(*, methods: list[str], dataset: str, limit: int | None, config: dict[str, Any], repo_root: str | Path = ".") -> dict[str, Any]:
    corpus_dir = str(config.get("paths", {}).get("corpus_dir", "data/corpus"))
    counts = audit_corpus(corpus_dir)
    llm_summary = llm_config_summary(config)
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
        "dataset": dataset,
        "limit": limit,
        "corpus_dir": corpus_dir,
        "llm_provider": llm_summary["provider"],
        "llm_model": llm_summary["model"],
        "temperature": llm_summary["temperature"],
        "heuristic_fallback": llm_summary["heuristic_fallback"],
        "git_commit_if_available": git_commit_if_available(repo_root),
        "data_counts": {
            "entities": counts["entity_count"],
            "relations": counts["relation_count"],
            "triples": counts["triple_count"],
            "evidence_chunks": counts["evidence_chunk_count"],
        },
        "reproduction_label": "E-KELL-style paper-faithful pipeline-level reimplementation",
    }


def write_run_manifest(manifest: dict[str, Any], output_path: str | Path = "outputs/run_manifest.json") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
