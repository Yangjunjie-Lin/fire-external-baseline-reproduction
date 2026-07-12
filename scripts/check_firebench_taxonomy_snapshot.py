#!/usr/bin/env python3
"""Compare local FireBench taxonomy/alias snapshots to main-project taxonomy.py (read-only)."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.checksums import sha256_file  # noqa: E402
from external_baselines.common.firebench_taxonomy import (  # noqa: E402
    ALIASES_PATH,
    TAXONOMY_PATH,
    load_formal_aliases,
    load_taxonomy,
)

MAIN_TAXONOMY_REL = Path("src/fire_agent_demo/evaluation/fireagent_bench/core/taxonomy.py")
MAIN_ALIAS_MAP = {
    "RISK_SIGNAL_ALIASES": "risk_signals",
    "REQUIRED_ACTION_ALIASES": "recommended_action_ids",
    "BLOCKED_ACTION_ALIASES": "blocked_action_ids",
    "MISSING_CONFIRMATION_ALIASES": "confirmation_ids",
}
MAIN_CANONICAL_MAP = {
    "CANONICAL_RISK_SIGNALS": "risk_signals",
    "CANONICAL_REQUIRED_ACTIONS": "recommended_action_ids",
    "CANONICAL_BLOCKED_ACTIONS": "blocked_action_ids",
    "CANONICAL_MISSING_CONFIRMATIONS": "confirmation_ids",
}


def _eval_constant_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "frozenset":
        if node.args and isinstance(node.args[0], (ast.Set, ast.List)):
            return frozenset(ast.literal_eval(node.args[0]))
    return ast.literal_eval(node)


def _extract_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = _eval_constant_node(node.value)
                    except (ValueError, SyntaxError):
                        continue
    return out


def _diff_alias_maps(main_map: dict[str, str], external_map: dict[str, str]) -> dict[str, Any]:
    extra = {k: v for k, v in external_map.items() if k not in main_map}
    missing = {k: v for k, v in main_map.items() if k not in external_map}
    different_targets = {
        k: {"main": main_map[k], "external": external_map[k]}
        for k in main_map.keys() & external_map.keys()
        if main_map[k] != external_map[k]
    }
    return {
        "extra_aliases": extra,
        "missing_aliases": missing,
        "different_targets": different_targets,
    }


def compare_taxonomy_snapshots(main_repo: Path) -> dict[str, Any]:
    main_taxonomy_path = main_repo / MAIN_TAXONOMY_REL
    if not main_taxonomy_path.is_file():
        return {
            "ok": False,
            "error": "explicit_main_taxonomy_missing",
            "path": str(main_taxonomy_path),
        }

    main_constants = _extract_constants(main_taxonomy_path)
    external_taxonomy = load_taxonomy()
    external_aliases = load_formal_aliases()

    canonical_diffs: dict[str, Any] = {}
    canonical_sets_match = True
    for main_key, external_key in MAIN_CANONICAL_MAP.items():
        main_values = sorted(str(v) for v in (main_constants.get(main_key) or ()))
        external_values = sorted(str(v) for v in (external_taxonomy.get(external_key) or ()))
        if main_values != external_values:
            canonical_sets_match = False
            canonical_diffs[external_key] = {
                "main_count": len(main_values),
                "external_count": len(external_values),
                "match": False,
            }

    alias_category_reports: dict[str, Any] = {}
    formal_alias_maps_match = True
    for main_key, external_key in MAIN_ALIAS_MAP.items():
        main_map = {str(k): str(v) for k, v in (main_constants.get(main_key) or {}).items()}
        external_map = {str(k): str(v) for k, v in (external_aliases.get(external_key) or {}).items()}
        diff = _diff_alias_maps(main_map, external_map)
        category_match = not diff["extra_aliases"] and not diff["missing_aliases"] and not diff["different_targets"]
        if not category_match:
            formal_alias_maps_match = False
        alias_category_reports[external_key] = {
            "main_count": len(main_map),
            "external_count": len(external_map),
            "match": category_match,
            **diff,
        }

    result = {
        "ok": canonical_sets_match and formal_alias_maps_match,
        "canonical_sets_match": canonical_sets_match,
        "formal_alias_maps_match": formal_alias_maps_match,
        "canonical_diffs": canonical_diffs,
        "alias_categories": alias_category_reports,
        "extra_aliases": {
            key: report["extra_aliases"] for key, report in alias_category_reports.items() if report["extra_aliases"]
        },
        "missing_aliases": {
            key: report["missing_aliases"]
            for key, report in alias_category_reports.items()
            if report["missing_aliases"]
        },
        "different_targets": {
            key: report["different_targets"]
            for key, report in alias_category_reports.items()
            if report["different_targets"]
        },
        "external_taxonomy_sha256": sha256_file(TAXONOMY_PATH),
        "external_alias_sha256": sha256_file(ALIASES_PATH),
        "main_taxonomy_path": str(main_taxonomy_path),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check FireBench taxonomy snapshot parity with main project")
    parser.add_argument("--main-repo", required=True, help="Path to fire-agent-demo (read-only)")
    args = parser.parse_args(argv)

    main_repo = Path(args.main_repo)
    if not main_repo.is_dir():
        print(json.dumps({"ok": False, "error": "explicit_main_repo_missing", "path": str(main_repo)}))
        return 1

    result = compare_taxonomy_snapshots(main_repo)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
