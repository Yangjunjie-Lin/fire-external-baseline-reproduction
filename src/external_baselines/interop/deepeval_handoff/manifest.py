"""Manifest and byte-stable JSON helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from external_baselines.interop.deepeval_handoff.constants import (
    CONTEXT_SELECTION_POLICY,
    HANDOFF_MANIFEST_VERSION,
    HANDOFF_PROTOCOL,
    REPOSITORY_NAME,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))


def case_ids_sha256(case_ids: Iterable[str]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(case_ids)))


def repository_identity(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "repository": REPOSITORY_NAME,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "worktree_clean": status == "" if status is not None else None,
    }


def build_manifest(
    *,
    repository_root: Path,
    formal_run_root: Path,
    source_manifest_sha256: str,
    suite_summary_sha256: str,
    formal_source: bool,
    transactional_publish_complete: bool,
    contract: dict[str, Any],
    split: str,
    case_ids: set[str],
    input_cases_sha256: str | None,
    runner_bundle_sha256: str | None,
    top_k: int,
    methods: dict[str, Any],
) -> dict[str, Any]:
    source = repository_identity(repository_root)
    source.update(
        {
            "formal_run_root": formal_run_root.as_posix(),
            "formal_run_manifest_sha256": source_manifest_sha256,
            "suite_summary_sha256": suite_summary_sha256,
            "formal_result": formal_source,
            "transactional_publish_complete": transactional_publish_complete,
            "development_artifact": not formal_source,
        }
    )
    return {
        "schema_version": HANDOFF_MANIFEST_VERSION,
        "protocol": HANDOFF_PROTOCOL,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": source,
        "contract": contract,
        "dataset": {
            "split": split,
            "case_count": len(case_ids),
            "case_ids_sha256": case_ids_sha256(case_ids),
            "input_cases_sha256": input_cases_sha256,
            "runner_bundle_sha256": runner_bundle_sha256,
        },
        "evaluation_handoff": {
            "handoff_top_k": top_k,
            "context_selection_policy": CONTEXT_SELECTION_POLICY,
            "gold_accessed": False,
            "deepeval_executed": False,
            "judge_called": False,
            "paid_api_used": False,
            "real_world_execution_allowed": False,
        },
        "methods": methods,
        "validation": {
            "schema_validation_passed": True,
            "coverage_validation_passed": True,
            "gold_isolation_passed": True,
            "formal_source_validation_passed": formal_source,
            "handoff_valid": True,
            "publication_eligible": formal_source,
        },
    }
