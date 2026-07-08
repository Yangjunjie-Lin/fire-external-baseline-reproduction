#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.graphrag_adapter.lightrag_adapter import is_available as lightrag_available  # noqa: E402
from external_baselines.graphrag_adapter.microsoft_graphrag_adapter import is_available as ms_graphrag_available  # noqa: E402

REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "configs/default.yaml",
    "configs/prompts/ekell_stage1_situation_understanding.txt",
    "configs/prompts/ekell_stage2_kg_grounded_decision_reasoning.txt",
    "configs/prompts/ekell_stage3_final_response.txt",
    "src/external_baselines/runner.py",
    "src/external_baselines/ekell_style/pipeline.py",
    "scripts/run_all_baselines.py",
    "scripts/validate_data.py",
    "scripts/validate_outputs.py",
    "scripts/analyze_manual_scores.py",
]
DOCS = [
    "docs/reproduction_fidelity_audit.md",
    "docs/top_tier_readiness_audit.md",
    "docs/paper_experiment_protocol.md",
    "docs/data_card_template.md",
    "docs/scenario_matrix_card_template.md",
    "docs/model_run_card_template.md",
    "docs/statistical_analysis_plan.md",
    "docs/prompt_template_documentation.md",
    "docs/no_overclaim_policy.md",
    "docs/safe_output_export_contract.md",
]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def status_line(label: str, ok: bool, detail: str = "") -> str:
    return f"- [{'x' if ok else ' '}] {label}{': ' + detail if detail else ''}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a project readiness diagnostic report.")
    parser.add_argument("--output", default="outputs/project_readiness_report.md")
    args = parser.parse_args()

    lines = ["# Project Readiness Report", "", "This diagnostic is structural. It does not run final experiments or certify results.", "", "## Required code/config files"]
    critical_missing = []
    for path in REQUIRED_FILES:
        ok = exists(path)
        if not ok:
            critical_missing.append(path)
        lines.append(status_line(path, ok))

    lines += ["", "## Documentation readiness"]
    for path in DOCS:
        lines.append(status_line(path, exists(path)))

    lines += ["", "## Data readiness"]
    scenario = exists("data/scenarios/scenario_matrix_v2.json")
    evidence = exists("data/corpus/evidence_chunks.jsonl")
    lines.append(status_line("scenario matrix", scenario, "run scripts/prepare_data.py if missing"))
    lines.append(status_line("evidence chunks", evidence, "run scripts/prepare_data.py if missing"))
    lines.append(status_line("entities", exists("data/corpus/entities.jsonl"), "warn-only for text baselines"))
    lines.append(status_line("relations", exists("data/corpus/relations.jsonl"), "warn-only for text baselines"))
    lines.append(status_line("triples", exists("data/corpus/triples.jsonl"), "warn-only for text baselines"))

    lines += ["", "## Output readiness"]
    lines.append(status_line("baseline outputs", exists("outputs/baseline_outputs.jsonl"), "run scripts/run_all_baselines.py if missing"))
    lines.append(status_line("metrics CSV", exists("outputs/baseline_metrics.csv")))
    lines.append(status_line("baseline report", exists("outputs/baseline_report.md")))
    lines.append(status_line("run manifest", exists("outputs/run_manifest.json")))
    lines.append(status_line("side-by-side comparison", exists("outputs/side_by_side_comparison.md"), "requires exported SAFE output"))

    lines += ["", "## LLM / adapter status"]
    default_cfg = (ROOT / "configs/default.yaml").read_text(encoding="utf-8") if exists("configs/default.yaml") else ""
    lines.append(status_line("heuristic mode is default", "provider: heuristic" in default_cfg, "smoke-test only; use real LLM config for paper runs"))
    lines.append(status_line("LightRAG package installed", lightrag_available(), "fallback adapter is expected if false"))
    lines.append(status_line("Microsoft GraphRAG package installed", ms_graphrag_available(), "fallback adapter is expected if false"))

    lines += ["", "## Manual evaluation readiness"]
    lines.append(status_line("manual sheet template", exists("evaluation_forms/manual_evaluation_sheet_template.csv")))
    lines.append(status_line("manual guidelines", exists("evaluation_forms/manual_evaluation_guidelines.md")))
    lines.append(status_line("IAA notes", exists("evaluation_forms/inter_annotator_agreement_notes.md")))

    lines += ["", "## Summary"]
    if critical_missing:
        lines.append("Critical files are missing: " + ", ".join(critical_missing))
    else:
        lines.append("Critical structural files are present. Final paper validity still requires real experiments, SAFE export, expert evaluation, and statistical analysis.")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if critical_missing:
        print("Critical missing files detected. See report.")


if __name__ == "__main__":
    main()
