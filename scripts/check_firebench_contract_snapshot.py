#!/usr/bin/env python3
"""Compare local firebench-interop schema snapshot to main-project schema (read-only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.checksums import sha256_file  # noqa: E402

LOCAL_SCHEMA = ROOT / "schemas" / "firebench_interop_v1_prediction.schema.json"
MAIN_SCHEMA_RELPATH = Path("schemas/firebench_interop_v1/prediction_schema.json")


def _committed_schema_bytes(main_root: Path) -> bytes | None:
    """Read the committed blob so checkout line-ending filters do not change authority."""
    if not (main_root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(main_root), "show", f"HEAD:{MAIN_SCHEMA_RELPATH.as_posix()}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check FireBench contract schema snapshot")
    parser.add_argument("--main-repo", default=None, help="Path to fire-agent-demo (optional)")
    args = parser.parse_args(argv)

    if not LOCAL_SCHEMA.is_file():
        print(json.dumps({"ok": False, "error": "local_schema_missing", "path": str(LOCAL_SCHEMA)}))
        return 1

    local_sha = sha256_file(LOCAL_SCHEMA)
    result = {
        "ok": True,
        "local_schema": str(LOCAL_SCHEMA),
        "local_sha256": local_sha,
        "main_repo_present": False,
        "match": None,
    }

    if args.main_repo:
        main_root = Path(args.main_repo)
        if not main_root.is_dir():
            print(
                json.dumps(
                    {"ok": False, "error": "explicit_main_repo_missing", "path": str(main_root)},
                    indent=2,
                )
            )
            return 1
        main_schema = main_root / MAIN_SCHEMA_RELPATH
        if not main_schema.is_file():
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "explicit_main_schema_missing",
                        "path": str(main_schema),
                    },
                    indent=2,
                )
            )
            return 1
        main_worktree_sha = sha256_file(main_schema)
        committed = _committed_schema_bytes(main_root)
        main_sha = hashlib.sha256(committed).hexdigest() if committed is not None else main_worktree_sha
        checkout_matches_commit = (
            committed is None
            or main_schema.read_bytes().replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n")
        )
        result["main_repo_present"] = True
        result["main_schema"] = str(main_schema)
        result["main_sha256"] = main_sha
        result["main_worktree_sha256"] = main_worktree_sha
        result["main_authority"] = "committed_blob" if main_sha != main_worktree_sha else "working_tree"
        result["main_worktree_matches_commit"] = checkout_matches_commit
        result["match"] = local_sha == main_sha
        result["ok"] = bool(result["match"] and checkout_matches_commit)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    result["note"] = "main_repo_not_provided_using_local_snapshot"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
