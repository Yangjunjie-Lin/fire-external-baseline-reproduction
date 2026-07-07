#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FIELDS = [
    "key_risks",
    "recommended_actions",
    "blocked_or_unsafe_actions",
    "missing_confirmations",
    "supporting_evidence",
    "final_decision_gate",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return "<br>".join(f"- {x}" for x in value) if value else ""
    if isinstance(value, dict):
        return "`" + json.dumps(value, ensure_ascii=False) + "`"
    return str(value or "")


def build_comparison(baseline_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> str:
    target_by_id = {str(r.get("scenario_id")): r for r in target_rows}
    baselines_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_rows:
        baselines_by_id.setdefault(str(row.get("scenario_id")), []).append(row)

    sections = [
        "# Side-by-side Baseline vs SAFE Fire Agent Comparison",
        "",
        "This report compares normalized output files only. It does not import or call `fire-agent-demo` code.",
        "",
        "Limitations: field-level comparisons are descriptive and do not certify emergency-response quality. Text-inferred fields may over- or under-estimate either system's behavior.",
        "",
    ]
    for sid in sorted(baselines_by_id):
        target = target_by_id.get(sid)
        sections.append(f"## Scenario: {sid}")
        if target is None:
            sections.append("No matching target output found.\n")
            continue
        for base in baselines_by_id[sid]:
            sections.append(f"### Baseline method: {base.get('method')}")
            sections.append("| Field | Baseline | Target SAFE Fire Agent |")
            sections.append("| --- | --- | --- |")
            for field in FIELDS:
                sections.append(f"| {field} | {_fmt(base.get(field))} | {_fmt(target.get(field))} |")
            sections.append("")
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline outputs with exported SAFE Fire Agent normalized outputs.")
    parser.add_argument("--baseline", default="outputs/baseline_outputs.jsonl")
    parser.add_argument("--target", required=True, help="Path to safe_outputs_normalized.jsonl exported outside this repo")
    parser.add_argument("--output", default="outputs/side_by_side_comparison.md")
    args = parser.parse_args()
    report = build_comparison(read_jsonl(args.baseline), read_jsonl(args.target))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Wrote side-by-side comparison to {args.output}")


if __name__ == "__main__":
    main()
