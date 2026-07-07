#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.io import load_scenarios, read_jsonl  # noqa: E402
from external_baselines.ekell_style.kg_loader import audit_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data availability for external baseline runs.")
    parser.add_argument("--data", default="data")
    parser.add_argument("--method", default="ekell_style")
    args = parser.parse_args()

    data_dir = Path(args.data)
    scenario_file = data_dir / "scenarios" / "scenario_matrix_v2.json"
    corpus_dir = data_dir / "corpus"
    warnings: list[str] = []
    errors: list[str] = []

    scenario_count = 0
    if not scenario_file.exists():
        errors.append(f"missing scenario file: {scenario_file}")
    else:
        try:
            scenario_count = len(load_scenarios(scenario_file))
        except Exception as exc:
            errors.append(f"failed to read scenarios: {exc}")

    evidence_file = corpus_dir / "evidence_chunks.jsonl"
    if not evidence_file.exists():
        errors.append(f"missing required evidence file for RAG/E-KELL-style methods: {evidence_file}")
    evidence_count = len(read_jsonl(evidence_file)) if evidence_file.exists() else 0

    audit = audit_corpus(corpus_dir)
    for filename in ["entities.jsonl", "relations.jsonl", "triples.jsonl"]:
        if filename in audit.get("missing_files", []):
            warnings.append(f"KG asset missing for KG-based methods: {filename}")

    warnings.extend(audit.get("schema_warnings", [])[:20])
    report = {
        "scenario_file_exists": scenario_file.exists(),
        "scenario_count": scenario_count,
        "evidence_file_exists": evidence_file.exists(),
        "evidence_chunk_count": evidence_count,
        "kg_asset_counts": {
            "entities": audit["entity_count"],
            "relations": audit["relation_count"],
            "triples": audit["triple_count"],
            "evidence_chunks": audit["evidence_chunk_count"],
        },
        "missing_files": audit.get("missing_files", []),
        "schema_warnings": warnings,
        "errors": errors,
        "valid_for_smoke_test": not errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
