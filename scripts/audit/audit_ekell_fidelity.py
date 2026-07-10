#!/usr/bin/env python3
"""Automated E-KELL fidelity architecture audit (no empirical claims)."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT_JSON = ROOT / "outputs" / "diagnostics" / "ekell_fidelity_audit.json"
OUT_MD = ROOT / "outputs" / "diagnostics" / "ekell_fidelity_audit.md"

VALID_STATUS = frozenset(
    {
        "implemented",
        "implemented_but_not_empirically_run",
        "approximated",
        "substituted",
        "interface_only",
        "unavailable_publicly",
        "not_applicable",
    }
)

CHECKS: list[dict[str, Any]] = [
    {
        "id": "query_decomposition",
        "module": "src/external_baselines/ekell_style/logical_query/query_decomposer.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "decompose_query",
        "status": "approximated",
    },
    {
        "id": "ast_validation",
        "module": "src/external_baselines/ekell_style/logical_query/validator.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "validate_query",
        "status": "approximated",
    },
    {
        "id": "fol_executor",
        "module": "src/external_baselines/ekell_style/logical_query/fol_executor.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "execute_query",
        "status": "approximated",
    },
    {
        "id": "vector_retriever",
        "module": "src/external_baselines/ekell_style/vector_retriever.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "VectorRetriever",
        "status": "implemented_but_not_empirically_run",
    },
    {
        "id": "neighborhood_expansion",
        "module": "src/external_baselines/ekell_style/neighborhood_expander.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "expand_neighborhood",
        "status": "approximated",
    },
    {
        "id": "stepwise_prompt_chain",
        "module": "src/external_baselines/ekell_style/stepwise_prompt_chain.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "run_stepwise_prompt_chain",
        "status": "approximated",
    },
    {
        "id": "controlled_track",
        "module": "src/external_baselines/ekell_style/full_pipeline.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "run_controlled_shared_llm",
        "status": "implemented_but_not_empirically_run",
    },
    {
        "id": "paper_fidelity_track",
        "module": "src/external_baselines/ekell_style/full_pipeline.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "run_paper_fidelity",
        "status": "interface_only",
    },
    {
        "id": "official_kg_flag",
        "module": "src/external_baselines/ekell_style/full_pipeline.py",
        "caller": "src/external_baselines/ekell_style/full_pipeline.py",
        "symbol": "official_ekell_kg",
        "status": "substituted",
    },
]


def _file_exists(rel: str) -> bool:
    return (ROOT / rel).is_file()


def _caller_uses_symbol(caller_rel: str, symbol: str) -> bool:
    path = ROOT / caller_rel
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return symbol in text


def _no_fire_agent_imports() -> bool:
    import_re = re.compile(r"^\s*(?:from|import)\s+fire_agent_demo\b", re.M)
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if import_re.search(text):
            return False
    return True


def _tests_for_module(module_rel: str) -> list[str]:
    stem = Path(module_rel).stem
    hits = []
    for test in (ROOT / "tests").rglob("test_*.py"):
        text = test.read_text(encoding="utf-8", errors="ignore")
        if stem in text or Path(module_rel).name.replace(".py", "") in text:
            hits.append(str(test.relative_to(ROOT)).replace("\\", "/"))
    return sorted(set(hits))


def build_audit() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for check in CHECKS:
        module_ok = _file_exists(check["module"])
        wired = _caller_uses_symbol(check["caller"], check["symbol"])
        items.append(
            {
                **check,
                "module_exists": module_ok,
                "wired_in_pipeline": wired,
                "tests": _tests_for_module(check["module"]),
                "pass": module_ok and wired,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": "E-KELL-style high-fidelity pipeline-level reimplementation, not official reproduction",
        "empirically_validated": False,
        "official_reproduction": False,
        "no_fire_agent_demo_import": _no_fire_agent_imports(),
        "enhanced_not_in_controlled": _file_exists(
            "src/external_baselines/ekell_style/enhanced_pipeline.py"
        ),
        "legacy_separate": _file_exists("src/external_baselines/ekell_style/pipeline.py"),
        "checks": items,
        "all_pass": all(i["pass"] for i in items) and _no_fire_agent_imports(),
    }


def render_md(audit: dict[str, Any]) -> str:
    lines = [
        "# E-KELL fidelity audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        f"- claim: {audit['claim']}",
        f"- empirically_validated: `{audit['empirically_validated']}`",
        f"- official_reproduction: `{audit['official_reproduction']}`",
        f"- no_fire_agent_demo_import: `{audit['no_fire_agent_demo_import']}`",
        "",
        "| module | wired | status | tests |",
        "|---|---|---|---|",
    ]
    for item in audit["checks"]:
        lines.append(
            f"| {item['id']} | {item['wired_in_pipeline']} | {item['status']} | {len(item['tests'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    audit = build_audit()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(audit), encoding="utf-8")
    print(json.dumps({"all_pass": audit["all_pass"], "written": [str(OUT_JSON), str(OUT_MD)]}, indent=2))
    if not audit["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
