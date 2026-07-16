#!/usr/bin/env python3
"""Export completed baseline predictions for centralized DeepEval evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.interop.deepeval_handoff.exporter import (  # noqa: E402
    HandoffExportError,
    export_handoff,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a DeepEval-compatible handoff bundle")
    parser.add_argument("--formal-run-root", required=True, type=Path)
    parser.add_argument("--main-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--allow-development-source", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_handoff(
            formal_run_root=args.formal_run_root,
            main_repo=args.main_repo,
            output=args.output,
            top_k=args.top_k,
            methods=args.methods,
            allow_development_source=args.allow_development_source,
            replace_existing=args.replace_existing,
        )
    except HandoffExportError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
