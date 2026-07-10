from __future__ import annotations

import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from external_baselines.common.io import write_json
from external_baselines.common.llm_client import llm_config_summary

REPRODUCTION_LABEL = "E-KELL-style paper-faithful pipeline-level reimplementation"


def get_git_commit_if_available(repo_root: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=3,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_run_manifest(
    *,
    methods: list[str],
    dataset: str | Path,
    limit: int | None,
    corpus_dir: str | Path,
    config: dict[str, Any],
    data_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    llm_summary = llm_config_summary(config)
    return {
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
        "dataset": str(dataset),
        "limit": limit,
        "corpus_dir": str(corpus_dir),
        "llm_provider": llm_summary["provider"],
        "llm_model": llm_summary["model"],
        "llm_model_version": llm_summary.get("model_version"),
        "llm_model_source": llm_summary.get("model_source"),
        "temperature": llm_summary["temperature"],
        "heuristic_fallback": llm_summary["heuristic_fallback"],
        "git_commit_if_available": get_git_commit_if_available(),
        "data_counts": data_counts or {},
        "reproduction_label": REPRODUCTION_LABEL,
    }


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)
