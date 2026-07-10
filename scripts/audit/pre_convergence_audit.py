#!/usr/bin/env python3
"""Scan repository state and write pre-convergence audit artifacts (local only)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT_JSON = ROOT / "outputs" / "diagnostics" / "pre_convergence_audit.json"
OUT_MD = ROOT / "outputs" / "diagnostics" / "pre_convergence_audit.md"

PLACEHOLDER_PATTERNS = (
    "REQUIRED_BEFORE_FORMAL_RUN",
    "path/to/",
    "ekell_style_faithful",
    "paper_main_table",
)


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _glob_rel(pattern: str) -> list[str]:
    return sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.glob(pattern))


def _find_duplicate_method_lists() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    patterns = [
        (r"MAIN_TABLE_METHODS\s*=", "MAIN_TABLE_METHODS literal"),
        (r"SUPPLEMENTAL_METHODS\s*=", "SUPPLEMENTAL_METHODS literal"),
        (r"METHOD_ID_ALIASES\s*=\s*\{", "METHOD_ID_ALIASES literal dict"),
        (r'"main_table_methods"\s*:\s*\[', "embedded main_table_methods JSON"),
    ]
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for pat, label in patterns:
            if re.search(pat, text) and "method_registry" not in rel:
                if "experiment_manifest" in rel and "MAIN_TABLE_METHODS = main_table_methods" in text:
                    continue
                if "interop/schema" in rel and "method_id_aliases()" in text:
                    continue
                hits.append({"path": rel, "pattern": label})
    return hits


def _scan_stale_refs() -> list[str]:
    stale: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_dir() or ".git" in path.parts or "outputs" in path.parts:
            continue
        if path.suffix not in {".md", ".yaml", ".yml", ".py", ".json", ".example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for token in PLACEHOLDER_PATTERNS:
            if token in text and "pre_convergence_audit" not in rel:
                stale.append(f"{rel}: contains `{token}`")
    return sorted(set(stale))[:200]


def build_audit() -> dict[str, Any]:
    from external_baselines.method_registry import (  # noqa: WPS433
        METHOD_REGISTRY,
        fallback_methods,
        legacy_methods,
        main_table_methods,
        method_id_aliases,
        paper_fidelity_methods,
        supplemental_methods,
    )

    formal_manifests = _glob_rel("configs/experiments/*.yaml*")
    smoke_configs = _glob_rel("configs/smoke/*.yaml*")
    frozen = _glob_rel("configs/frozen/*.yaml")
    schemas = _glob_rel("schemas/**/*.json") + _glob_rel("schemas/**/README.md")

    cli_formal = ["scripts/run_interop_baselines.py"]
    cli_legacy = sorted(
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in (ROOT / "scripts").glob("*.py")
        if p.name not in {"run_interop_baselines.py", "smoke_interop.py", "smoke_main_runner_bundle.py"}
    )
    proxy_eval = ["scripts/evaluate_predictions.py"]

    test_count = len(list((ROOT / "tests").rglob("test_*.py")))

    main_project = ROOT.parent / "fire-agent-demo"
    mp_status: dict[str, Any] = {"available": main_project.exists()}
    if main_project.exists():
        mp_status["branch"] = _git(["-C", str(main_project), "branch", "--show-current"])
        mp_status["head"] = _git(["-C", str(main_project), "log", "-1", "--oneline"])

    audit: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "git": {
            "branch": _git(["branch", "--show-current"]),
            "head": _git(["log", "-1", "--oneline"]),
            "dirty": bool(_git(["status", "--short"])),
        },
        "method_registry": {
            "source_of_truth": "src/external_baselines/method_registry.py",
            "method_count": len(METHOD_REGISTRY),
            "main_table": list(main_table_methods()),
            "paper_fidelity": list(paper_fidelity_methods()),
            "supplemental": list(supplemental_methods()),
            "fallback": list(fallback_methods()),
            "legacy": list(legacy_methods()),
            "alias_count": len(method_id_aliases()),
        },
        "alias_map_location": "derived from method_registry.method_id_aliases()",
        "experiment_manifests": formal_manifests,
        "smoke_manifests": smoke_configs,
        "legacy_manifests": [p for p in formal_manifests if "archive" in p],
        "frozen_configs": frozen,
        "schema_copies": schemas,
        "formal_cli": cli_formal,
        "legacy_cli": cli_legacy,
        "proxy_evaluator": proxy_eval,
        "formal_evaluator_authority": "fire-agent-demo shared evaluator (external)",
        "ekell_pipeline_modules": sorted(
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in (ROOT / "src/external_baselines/ekell_style").rglob("*.py")
        ),
        "graphrag_adapters": sorted(
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in (ROOT / "src/external_baselines/graphrag_adapter").glob("*.py")
        ),
        "status_docs": _glob_rel("docs/status/*.md"),
        "ci_workflow": _glob_rel(".github/workflows/*.yml"),
        "test_file_count": test_count,
        "duplicate_method_list_suspects": _find_duplicate_method_lists(),
        "stale_path_references_sample": _scan_stale_refs(),
        "main_project_checkout": mp_status,
    }
    return audit


def render_md(audit: dict[str, Any]) -> str:
    lines = [
        "# Pre-convergence audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "## Git",
        f"- branch: `{audit['git']['branch']}`",
        f"- HEAD: `{audit['git']['head']}`",
        f"- dirty: `{audit['git']['dirty']}`",
        "",
        "## Method registry",
        f"- source: `{audit['method_registry']['source_of_truth']}`",
        f"- main table: {', '.join(audit['method_registry']['main_table'])}",
        "",
        "## Manifests",
        f"- experiments: {len(audit['experiment_manifests'])}",
        f"- smoke: {len(audit['smoke_manifests'])}",
        f"- frozen: {len(audit['frozen_configs'])}",
        "",
        "## CLI",
        f"- formal: {', '.join(audit['formal_cli'])}",
        f"- proxy: {', '.join(audit['proxy_evaluator'])}",
        "",
        "## Tests",
        f"- test files: {audit['test_file_count']}",
        "",
        "## Stale references (sample)",
    ]
    for item in audit.get("stale_path_references_sample", [])[:30]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Duplicate method list suspects")
    for item in audit.get("duplicate_method_list_suspects", []):
        lines.append(f"- `{item['path']}`: {item['pattern']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    audit = build_audit()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(audit), encoding="utf-8")
    print(json.dumps({"written": [str(OUT_JSON), str(OUT_MD)]}, indent=2))


if __name__ == "__main__":
    main()
