#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.evaluation.statistics import cohens_d_paired, mean, paired_difference, standard_deviation, summarize, win_rate  # noqa: E402

DIMENSIONS = [
    "correctness_0_3",
    "evidence_support_0_3",
    "safety_compliance_0_3",
    "completeness_0_3",
    "actionability_0_3",
    "conciseness_0_3",
    "comprehensibility_0_3",
    "overall_0_3",
]


def _float(value: str) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str]], safe_method: str = "safe_fire_agent") -> tuple[list[dict[str, Any]], str]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row.get("method", "")].append(row)

    summary_rows: list[dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        out: dict[str, Any] = {"method": method, "n_rows": len(items)}
        for dim in DIMENSIONS:
            vals = [_float(r.get(dim, "")) for r in items]
            stats = summarize([v for v in vals if v is not None])
            out[f"{dim}_mean"] = stats["mean"]
            out[f"{dim}_std"] = stats["std"]
            out[f"{dim}_ci_low"] = stats["bootstrap_ci_low"]
            out[f"{dim}_ci_high"] = stats["bootstrap_ci_high"]
        summary_rows.append(out)

    safe_rows = by_method.get(safe_method, [])
    pairwise_lines = ["# Manual Evaluation Summary", "", "## Pairwise differences vs SAFE", ""]
    if not safe_rows:
        pairwise_lines.append(f"No method named `{safe_method}` found; pairwise SAFE comparisons skipped.")
    else:
        safe_by_scenario = {r.get("scenario_id"): r for r in safe_rows}
        for method, items in sorted(by_method.items()):
            if method == safe_method:
                continue
            pairwise_lines.append(f"### {method} vs {safe_method}")
            pairwise_lines.append("| Dimension | Mean paired difference (SAFE - method) | Cohen d paired | SAFE win rate | n |")
            pairwise_lines.append("|---|---:|---:|---:|---:|")
            method_by_scenario = {r.get("scenario_id"): r for r in items}
            common = sorted(set(safe_by_scenario) & set(method_by_scenario))
            for dim in DIMENSIONS:
                safe_vals = [_float(safe_by_scenario[s].get(dim, "")) for s in common]
                base_vals = [_float(method_by_scenario[s].get(dim, "")) for s in common]
                diffs = paired_difference(safe_vals, base_vals)
                wr = win_rate(safe_vals, base_vals)
                pairwise_lines.append(f"| {dim} | {mean(diffs)} | {cohens_d_paired(safe_vals, base_vals)} | {wr['win_rate']} | {wr['n']} |")
            pairwise_lines.append("")
    return summary_rows, "\n".join(pairwise_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze manual evaluation scores without fabricating results.")
    parser.add_argument("--input", default="evaluation_forms/manual_evaluation_results.csv")
    parser.add_argument("--safe-method", default="safe_fire_agent")
    parser.add_argument("--summary-csv", default="outputs/manual_eval_summary.csv")
    parser.add_argument("--summary-md", default="outputs/manual_eval_summary.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Manual evaluation file not found: {input_path}")
        print("Copy evaluation_forms/manual_evaluation_sheet_template.csv to this path, fill real evaluator scores, then rerun.")
        return
    rows = read_rows(input_path)
    summary_rows, report = build_summary(rows, safe_method=args.safe_method)
    write_csv(Path(args.summary_csv), summary_rows)
    Path(args.summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_md).write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.summary_md}")


if __name__ == "__main__":
    main()
