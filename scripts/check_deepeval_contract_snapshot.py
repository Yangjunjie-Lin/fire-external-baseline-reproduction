#!/usr/bin/env python3
"""Check or explicitly update the committed external-prediction snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.interop.deepeval_handoff.contracts import (  # noqa: E402
    contract_report,
    update_contract_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check fireagent external prediction contract parity")
    parser.add_argument("--main-repo", required=True, type=Path)
    parser.add_argument("--update-snapshot", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.update_snapshot:
            result = update_contract_snapshot(repository_root=ROOT, main_repo=args.main_repo.resolve())
        else:
            result = contract_report(repository_root=ROOT, main_repo=args.main_repo.resolve())
    except Exception as exc:  # CLI boundary
        result = {"ok": False, "errors": [str(exc)]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
