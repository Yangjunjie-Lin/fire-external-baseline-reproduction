#!/usr/bin/env python3
"""Validate firebench-interop prediction JSONL against FireBench taxonomy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.firebench_taxonomy import (  # noqa: E402
    membership_set,
    taxonomy_provenance,
)
from external_baselines.common.io import ensure_dir, read_jsonl, write_json  # noqa: E402
from external_baselines.common.taxonomy_normalizer import (  # noqa: E402
    normalize_action_id,
    normalize_blocked_action_id,
    normalize_confirmation_id,
    normalize_final_gate,
    normalize_priority,
    normalize_response_status,
    normalize_risk_level,
    normalize_risk_signal,
)


def _looks_like_natural_language(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    if " " in text and len(text.split()) >= 3:
        return True
    return False


def validate_prediction_file(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    errors: list[dict[str, Any]] = []
    method_ids = set()
    for idx, row in enumerate(rows, start=1):
        if row.get("schema_version") != "firebench-interop-v1":
            errors.append({"line": idx, "error": "invalid_schema_version", "value": row.get("schema_version")})
        method_ids.add(str(row.get("method_id") or ""))
        pred = row.get("prediction") or {}
        case_id = row.get("case_id")

        for signal in pred.get("risk_signals") or []:
            if normalize_risk_signal(str(signal), strict=False) is None:
                errors.append({"line": idx, "case_id": case_id, "field": "risk_signals", "value": signal})
            if _looks_like_natural_language(str(signal)):
                errors.append({"line": idx, "case_id": case_id, "field": "risk_signals_nl", "value": signal})

        if normalize_risk_level(str(pred.get("risk_level") or ""), strict=False) is None:
            errors.append({"line": idx, "case_id": case_id, "field": "risk_level", "value": pred.get("risk_level")})

        for action in pred.get("recommended_actions") or []:
            if not isinstance(action, dict):
                errors.append({"line": idx, "case_id": case_id, "field": "recommended_actions", "value": action})
                continue
            aid = action.get("action_id")
            if normalize_action_id(str(aid or ""), strict=False) is None:
                errors.append({"line": idx, "case_id": case_id, "field": "action_id", "value": aid})
            if _looks_like_natural_language(str(aid or "")):
                errors.append({"line": idx, "case_id": case_id, "field": "action_id_nl", "value": aid})
            if normalize_priority(str(action.get("priority") or "unknown"), strict=False) is None:
                errors.append(
                    {"line": idx, "case_id": case_id, "field": "priority", "value": action.get("priority")}
                )

        for blocked in pred.get("blocked_actions") or []:
            if normalize_blocked_action_id(str(blocked), strict=False) is None:
                errors.append({"line": idx, "case_id": case_id, "field": "blocked_actions", "value": blocked})
            if _looks_like_natural_language(str(blocked)):
                errors.append({"line": idx, "case_id": case_id, "field": "blocked_actions_nl", "value": blocked})
            if str(blocked) != str(blocked).upper() and str(blocked).upper() in membership_set("blocked_action_ids"):
                errors.append({"line": idx, "case_id": case_id, "field": "blocked_actions_case", "value": blocked})

        for conf in pred.get("missing_confirmations") or []:
            if normalize_confirmation_id(str(conf), strict=False) is None:
                errors.append({"line": idx, "case_id": case_id, "field": "missing_confirmations", "value": conf})
            if _looks_like_natural_language(str(conf)):
                errors.append(
                    {"line": idx, "case_id": case_id, "field": "missing_confirmations_nl", "value": conf}
                )

        if normalize_final_gate(str(pred.get("final_decision_gate") or ""), strict=False) is None:
            errors.append(
                {
                    "line": idx,
                    "case_id": case_id,
                    "field": "final_decision_gate",
                    "value": pred.get("final_decision_gate"),
                }
            )
        fr = pred.get("final_response") or {}
        if normalize_response_status(str(fr.get("status") or ""), strict=False) is None:
            errors.append({"line": idx, "case_id": case_id, "field": "final_response.status", "value": fr.get("status")})

        for field_name in ("risk_signals", "blocked_actions", "missing_confirmations"):
            values = [str(v) for v in (pred.get(field_name) or [])]
            if len(values) != len(set(values)):
                errors.append({"line": idx, "case_id": case_id, "field": f"{field_name}_duplicate"})
            if any(v.strip() == "" for v in values):
                errors.append({"line": idx, "case_id": case_id, "field": f"{field_name}_empty"})

    if len(method_ids) != 1 or "" in method_ids:
        errors.append({"error": "prediction_file_must_contain_single_method_id", "method_ids": sorted(method_ids)})

    return {
        "path": str(path),
        "row_count": len(rows),
        "method_ids": sorted(method_ids),
        "error_count": len(errors),
        "errors": errors,
        "ok": len(errors) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check FireBench taxonomy compliance of prediction JSONL")
    parser.add_argument("--predictions", default=None, help="Single prediction JSONL path")
    parser.add_argument("--prediction-dir", default=None, help="Directory of *.jsonl prediction files")
    parser.add_argument(
        "--report",
        default="outputs/diagnostics/taxonomy_validation_report.json",
        help="Report output path",
    )
    parser.add_argument("--report-only", action="store_true", help="Always exit 0")
    args = parser.parse_args(argv)

    paths: list[Path] = []
    if args.predictions:
        paths.append(Path(args.predictions))
    if args.prediction_dir:
        paths.extend(sorted(Path(args.prediction_dir).glob("*.jsonl")))
    if not paths:
        raise SystemExit("Provide --predictions or --prediction-dir")

    file_reports = [validate_prediction_file(p) for p in paths]
    report = {
        "ok": all(r["ok"] for r in file_reports),
        "taxonomy_provenance": taxonomy_provenance(),
        "files": file_reports,
        "error_count": sum(int(r["error_count"]) for r in file_reports),
    }
    out = Path(args.report)
    ensure_dir(out.parent)
    write_json(out, report)
    print(json.dumps({"ok": report["ok"], "error_count": report["error_count"], "report": str(out)}, indent=2))
    if args.report_only:
        return 0
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
