#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = [
    "scenario_id",
    "method",
    "situation_summary",
    "key_risks",
    "recommended_actions",
    "blocked_or_unsafe_actions",
    "missing_confirmations",
    "supporting_evidence",
    "citations",
    "final_decision_gate",
    "retrieved_contexts",
    "latency_sec",
    "raw_output",
    "method_specific",
]
LIST_FIELDS = ["key_risks", "recommended_actions", "blocked_or_unsafe_actions", "missing_confirmations", "supporting_evidence", "citations", "retrieved_contexts"]
PLACEHOLDERS = {"...", "todo", "tbd", "fill", "<fill>", "not implemented", "placeholder"}
ADAPTER_METHODS = {"lightrag", "microsoft_graphrag"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"__invalid_json__": str(exc), "__line__": line_no})
    return rows


def has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in PLACEHOLDERS
    if isinstance(value, list):
        return any(has_placeholder(v) for v in value)
    if isinstance(value, dict):
        return any(has_placeholder(v) for v in value.values())
    return False


def validate_row(row: dict[str, Any], index: int) -> list[str]:
    issues: list[str] = []
    prefix = f"row {index}"
    if "__invalid_json__" in row:
        return [f"{prefix}: invalid JSON at line {row.get('__line__')}: {row['__invalid_json__']}"]
    for field in REQUIRED_FIELDS:
        if field not in row:
            issues.append(f"{prefix}: missing required field `{field}`")
    for field in LIST_FIELDS:
        if field in row and not isinstance(row[field], list):
            issues.append(f"{prefix}: `{field}` must be a list")
    if not row.get("scenario_id"):
        issues.append(f"{prefix}: scenario_id is empty")
    if not row.get("method"):
        issues.append(f"{prefix}: method is empty")
    if not row.get("final_decision_gate"):
        issues.append(f"{prefix}: final_decision_gate is empty")
    method_specific = row.get("method_specific")
    if not isinstance(method_specific, dict):
        issues.append(f"{prefix}: method_specific must be an object")
    else:
        if row.get("method") in {"direct_llm", "vanilla_rag", "ekell_style", "lightrag", "microsoft_graphrag"}:
            if "llm_config_summary" not in method_specific:
                issues.append(f"{prefix}: method_specific.llm_config_summary missing")
            else:
                llm = method_specific.get("llm_config_summary") or {}
                if "heuristic_fallback" not in llm:
                    issues.append(f"{prefix}: llm_config_summary.heuristic_fallback missing")
        if row.get("method") in ADAPTER_METHODS:
            for key in ["adapter_status", "actual_external_package_used", "fallback_retrieval_used", "indexing_performed"]:
                if key not in method_specific:
                    issues.append(f"{prefix}: adapter method missing method_specific.{key}")
    for field in REQUIRED_FIELDS:
        if has_placeholder(row.get(field)):
            issues.append(f"{prefix}: field `{field}` appears placeholder-only")
    return issues


def build_report(rows: list[dict[str, Any]], issues: list[str], input_path: Path) -> str:
    methods = sorted({str(r.get("method")) for r in rows if isinstance(r, dict) and r.get("method")})
    return "\n".join([
        "# Output Validation Report",
        "",
        f"Input: `{input_path}`",
        f"Rows checked: {len(rows)}",
        f"Methods: {', '.join(methods) if methods else 'none'}",
        f"Issue count: {len(issues)}",
        "",
        "## Issues",
        "No issues found." if not issues else "\n".join(f"- {issue}" for issue in issues),
        "",
        "## Notes",
        "This script validates schema and obvious recording problems only. It does not judge emergency correctness.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate baseline output JSONL records.")
    parser.add_argument("--input", default="outputs/baseline_outputs.jsonl")
    parser.add_argument("--output", default="outputs/output_validation_report.md")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Missing output file: {input_path}")
        print("Run scripts/run_all_baselines.py first, or provide --input path/to/baseline_outputs.jsonl")
        return
    rows = read_jsonl(input_path)
    issues: list[str] = []
    for i, row in enumerate(rows, start=1):
        issues.extend(validate_row(row, i))
    report = build_report(rows, issues, input_path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if issues:
        print(f"Found {len(issues)} validation issue(s).")


if __name__ == "__main__":
    main()
