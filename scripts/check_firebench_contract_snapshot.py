#!/usr/bin/env python3
"""Compare local firebench-interop schema snapshot to main-project schema (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.checksums import sha256_file  # noqa: E402

LOCAL_SCHEMA = ROOT / "schemas" / "firebench_interop_v1_prediction.schema.json"


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
        main_schema = main_root / "schemas" / "firebench_interop_v1" / "prediction_schema.json"
        if not main_schema.is_file():
            result["ok"] = True
            result["warning"] = "main_schema_missing_using_local_snapshot"
            print(json.dumps(result, indent=2))
            return 0
        main_sha = sha256_file(main_schema)
        result["main_repo_present"] = True
        result["main_schema"] = str(main_schema)
        result["main_sha256"] = main_sha
        result["match"] = local_sha == main_sha
        result["ok"] = bool(result["match"])
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    result["note"] = "main_repo_not_provided_using_local_snapshot"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
