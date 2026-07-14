#!/usr/bin/env python3
"""Read-only strict FireKG corpus audit.

Runs the strict FireKG contract against a public/DEV corpus directory and
writes a structure-only diagnostics report (counts, duplicate statistics,
rejected field-type statistics, schema errors with file names and original
JSONL line numbers).  The corpus is never modified and no record content is
copied into the report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.ekell_style.kg_loader import (  # noqa: E402
    CORPUS_FILES,
    JsonlObjectRow,
    _reject_duplicate_identities,
    _strict_identity,
    _validate_strict_row,
    load_kg_strict,
    read_jsonl_object_records_strict,
)


def _error_kind(message: str) -> str:
    """Stable machine-parseable schema error code without record content."""
    parts = message.split(":")
    if len(parts) >= 4 and parts[0] == "kg_schema_invalid":
        return parts[3]
    return parts[0]


def audit_strict_firekg(corpus_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "corpus_dir_declared": str(corpus_dir),
        "resolved_path_authoritative": False,
        "ok": True,
        "strict_loader_executed": False,
        "strict_loader_error": None,
        "counts": {},
        "duplicate_statistics": {},
        "rejected_field_type_statistics": {},
        "schema_errors": [],
    }
    if not corpus_dir.is_dir():
        report["ok"] = False
        report["schema_errors"].append(
            {"file": None, "line": None, "error": "kg_corpus_dir_missing"}
        )
        return report

    report["strict_loader_executed"] = True
    try:
        strict_kg = load_kg_strict(corpus_dir)
        strict_counts = strict_kg.counts()
    except ValueError as exc:
        report["ok"] = False
        report["strict_loader_error"] = str(exc)
        strict_counts = None

    for kind, filename in CORPUS_FILES.items():
        path = corpus_dir / filename
        rejected: Counter[str] = Counter()
        errors: list[dict[str, Any]] = []
        rows: list[JsonlObjectRow] = []
        try:
            rows = read_jsonl_object_records_strict(path, require_nonempty=True)
        except ValueError as exc:
            message = str(exc)
            report["ok"] = False
            errors.append({"file": filename, "line": None, "error": message})
            report["counts"][kind] = 0
            report["schema_errors"].extend(errors)
            continue

        valid_rows: list[JsonlObjectRow] = []
        for row in rows:
            try:
                _validate_strict_row(kind, row, filename=filename)
            except ValueError as exc:
                message = str(exc)
                report["ok"] = False
                rejected[_error_kind(message)] += 1
                errors.append(
                    {"file": filename, "line": row.line_no, "error": message}
                )
                continue
            valid_rows.append(row)

        duplicate_count = 0
        identity_counter: Counter[tuple[str, str]] = Counter()
        for row in valid_rows:
            try:
                identity_counter[_strict_identity(kind, row.value)] += 1
            except ValueError:
                continue
        duplicate_count = sum(
            count - 1 for count in identity_counter.values() if count > 1
        )
        if duplicate_count:
            report["ok"] = False
            try:
                _reject_duplicate_identities(
                    kind,
                    valid_rows,
                    filename=filename,
                )
            except ValueError as exc:
                message = str(exc)
                match = re.search(r":line_(\d+)", message)
                errors.append(
                    {
                        "file": filename,
                        "line": int(match.group(1)) if match else None,
                        "error": message,
                    }
                )

        report["counts"][kind] = len(rows)
        report["duplicate_statistics"][kind] = duplicate_count
        if rejected:
            report["rejected_field_type_statistics"][kind] = dict(rejected)
        report["schema_errors"].extend(errors)
    if strict_counts is not None:
        report["counts"] = strict_counts
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Strict FireKG corpus audit (read-only).")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = audit_strict_firekg(Path(args.corpus_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
