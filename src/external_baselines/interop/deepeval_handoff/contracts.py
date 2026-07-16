"""Read-only contract snapshot and provenance operations."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from external_baselines.interop.deepeval_handoff.constants import (
    CONTRACT_ID,
    CONTRACT_SOURCE_PATH,
    CONTRACT_SOURCE_REPOSITORY,
    PROVENANCE_PATH,
    SNAPSHOT_PATH,
)
from external_baselines.interop.deepeval_handoff.schema_validation import (
    check_draft202012_schema,
    load_json_object,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_commit(main_repo: Path) -> str | None:
    if not (main_repo / ".git").exists():
        marker = main_repo / "contract_source_commit.txt"
        commit = marker.read_text(encoding="utf-8").strip().lower() if marker.is_file() else ""
    else:
        try:
            result = subprocess.run(
                ["git", "-C", str(main_repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            commit = result.stdout.strip().lower()
        except (OSError, subprocess.CalledProcessError):
            commit = ""
    return commit if COMMIT_RE.fullmatch(commit) else None


def _committed_source_bytes(main_repo: Path, commit: str) -> bytes | None:
    if not (main_repo / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(main_repo), "show", f"{commit}:{CONTRACT_SOURCE_PATH.as_posix()}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def contract_report(
    *,
    repository_root: Path,
    main_repo: Path,
) -> dict[str, Any]:
    source = main_repo / CONTRACT_SOURCE_PATH
    snapshot = repository_root / SNAPSHOT_PATH
    provenance_path = repository_root / PROVENANCE_PATH
    errors: list[str] = []
    if not main_repo.is_dir():
        errors.append("main_repository_missing")
    if not source.is_file():
        errors.append("authoritative_schema_missing")
    if not snapshot.is_file():
        errors.append("local_snapshot_missing")
    if not provenance_path.is_file():
        errors.append("contract_provenance_missing")

    source_sha = sha256_file(source) if source.is_file() else None
    snapshot_sha = sha256_file(snapshot) if snapshot.is_file() else None
    commit = _repository_commit(main_repo) if main_repo.is_dir() else None
    provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        try:
            provenance = load_json_object(provenance_path)
        except ValueError as exc:
            errors.append(str(exc))

    for label, path in (("source", source), ("snapshot", snapshot)):
        if path.is_file():
            try:
                check_draft202012_schema(load_json_object(path))
            except ValueError as exc:
                errors.append(f"{label}_schema_invalid:{exc}")

    if source_sha and snapshot_sha and source_sha != snapshot_sha:
        errors.append("source_snapshot_sha256_mismatch")
    if not commit:
        errors.append("source_commit_missing_or_invalid")
    committed_bytes = _committed_source_bytes(main_repo, commit) if commit else None
    committed_sha = hashlib.sha256(committed_bytes).hexdigest() if committed_bytes is not None else None
    if committed_sha and source_sha and committed_sha != source_sha:
        errors.append("authoritative_schema_has_uncommitted_changes")

    expected = {
        "schema_version": "deepeval-handoff-contract-provenance-v1",
        "contract_id": CONTRACT_ID,
        "source_repository": CONTRACT_SOURCE_REPOSITORY,
        "source_commit": commit,
        "source_path": CONTRACT_SOURCE_PATH.as_posix(),
        "source_sha256": source_sha,
        "local_snapshot_path": SNAPSHOT_PATH.as_posix(),
        "local_snapshot_sha256": snapshot_sha,
        "snapshot_matches_source": True,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            errors.append(f"provenance_mismatch:{key}")
    for key in ("source_sha256", "local_snapshot_sha256"):
        if not SHA256_RE.fullmatch(str(provenance.get(key) or "")):
            errors.append(f"provenance_invalid_sha256:{key}")
    if not COMMIT_RE.fullmatch(str(provenance.get("source_commit") or "")):
        errors.append("provenance_invalid_source_commit")

    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "contract_id": CONTRACT_ID,
        "main_repository": str(main_repo),
        "source_commit": commit,
        "source_path": CONTRACT_SOURCE_PATH.as_posix(),
        "source_sha256": source_sha,
        "committed_source_sha256": committed_sha,
        "local_snapshot_sha256": snapshot_sha,
        "snapshot_matches_source": bool(source_sha and source_sha == snapshot_sha),
    }


def update_contract_snapshot(*, repository_root: Path, main_repo: Path) -> dict[str, Any]:
    source = main_repo / CONTRACT_SOURCE_PATH
    if not source.is_file():
        raise ValueError(f"authoritative_schema_missing:{source}")
    commit = _repository_commit(main_repo)
    if not commit:
        raise ValueError("source_commit_missing_or_invalid")
    source_bytes = source.read_bytes()
    committed_bytes = _committed_source_bytes(main_repo, commit)
    if committed_bytes is not None and committed_bytes != source_bytes:
        raise ValueError("authoritative_schema_has_uncommitted_changes")
    schema = load_json_object(source)
    check_draft202012_schema(schema)
    snapshot = repository_root / SNAPSHOT_PATH
    provenance_path = repository_root / PROVENANCE_PATH
    before = sha256_file(snapshot) if snapshot.is_file() else None
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()
    provenance = {
        "schema_version": "deepeval-handoff-contract-provenance-v1",
        "contract_id": CONTRACT_ID,
        "source_repository": CONTRACT_SOURCE_REPOSITORY,
        "source_commit": commit,
        "source_path": CONTRACT_SOURCE_PATH.as_posix(),
        "source_sha256": digest,
        "local_snapshot_path": SNAPSHOT_PATH.as_posix(),
        "local_snapshot_sha256": digest,
        "snapshot_matches_source": True,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "before_sha256": before, "after_sha256": digest, "source_commit": commit}
