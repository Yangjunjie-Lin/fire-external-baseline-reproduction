#!/usr/bin/env python3
"""Package a validated handoff into a deterministic ZIP archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.interop.deepeval_handoff.packaging import package_handoff  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package a validated DeepEval handoff")
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--main-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = package_handoff(
            handoff=args.handoff.resolve(),
            main_repo=args.main_repo.resolve(),
            archive=args.output.resolve(),
        )
    except Exception as exc:  # CLI boundary
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
