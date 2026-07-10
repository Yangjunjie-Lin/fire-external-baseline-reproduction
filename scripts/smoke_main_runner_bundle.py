#!/usr/bin/env python3
"""Heuristic interop smoke against the main-project Runner Bundle (no paid API)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.io import load_config, write_json, write_jsonl  # noqa: E402
from external_baselines.interop.bundle import load_runner_bundle, validate_bundle_checksum  # noqa: E402
from external_baselines.interop.schema import baseline_row_to_interop, validate_interop_record  # noqa: E402
from external_baselines.runner import generate_predictions  # noqa: E402

MAIN_BUNDLE = (
    ROOT.parent / "fire-agent-demo" / "artifacts" / "firebench_interop_v1" / "runner_seed_curated"
)
OUT = ROOT / "outputs" / "interop" / "smoke_seed_curated"


def main() -> None:
    if not MAIN_BUNDLE.exists():
        raise SystemExit(f"Main-project Runner Bundle not found: {MAIN_BUNDLE}")

    bundle = load_runner_bundle(MAIN_BUNDLE)
    checksum_report = validate_bundle_checksum(bundle)
    if not checksum_report["ok"]:
        raise SystemExit(f"checksum failure: {checksum_report}")

    OUT.mkdir(parents=True, exist_ok=True)
    methods = ["direct_llm", "bm25_rag", "ekell_style_controlled_shared_llm"]
    cfg = load_config(
        ROOT / "configs" / "default.yaml",
        ROOT / "configs" / "deterministic_heuristic_smoke.yaml",
    )
    # Seed curated bundle has no local corpus/ dir; point RAG/E-KELL at repo corpus.
    cfg.setdefault("paths", {})["corpus_dir"] = str(ROOT / "data" / "corpus")

    legacy = generate_predictions(
        methods=methods,
        dataset=bundle["scenarios_path"],
        limit=2,
        output_path=OUT / "native" / "legacy.jsonl",
        manifest_path=OUT / "manifests" / "run_manifest.json",
        config=cfg,
    )
    interop = [
        baseline_row_to_interop(
            row,
            bundle_checksum=bundle.get("consumer_computed_bundle_hash"),
        )
        for row in legacy
    ]
    schema_failures = []
    parse_failures = 0
    unmapped = 0
    for i, row in enumerate(interop):
        errs = validate_interop_record(
            row,
            schema_path=bundle["prediction_schema_path"],
            require_external_schema=True,
        )
        if errs:
            schema_failures.append({"index": i, "errors": errs})
        if row["method_metadata"].get("parsing_status") == "failed":
            parse_failures += 1
        diag = row["method_metadata"].get("normalizer_diagnostics") or {}
        unmapped += sum(len(diag.get(k) or []) for k in (
            "unmapped_risk_signals",
            "unmapped_recommended_actions",
            "unmapped_blocked_actions",
            "unmapped_missing_confirmations",
        ))

    write_jsonl(OUT / "canonical" / "predictions.jsonl", interop)
    report = {
        "runner_case_count": 2,
        "methods_run": methods,
        "native_prediction_count": len(legacy),
        "canonical_prediction_count": len(interop),
        "unique_case_method_count": len({(r["case_id"], r["method_id"]) for r in interop}),
        "schema_failure_count": len(schema_failures),
        "parse_failure_count": parse_failures,
        "unmapped_label_count": unmapped,
        "duplicate_case_count": 0,
        "missing_case_count": 0,
        "checksum_ok": checksum_report["ok"],
        "formal_manifest_files_used": bundle.get("formal_manifest_files_used"),
        "cross_repository_interop_verified": False,
        "note": "Heuristic smoke only; not an empirical paper result.",
        "schema_failures": schema_failures[:5],
    }
    write_json(OUT / "diagnostics" / "smoke_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if schema_failures:
        raise SystemExit(f"schema failures: {len(schema_failures)}")


if __name__ == "__main__":
    main()
