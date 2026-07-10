#!/usr/bin/env python3
"""Build or validate Dense / Hybrid / E-KELL comparison indexes.

Default mode may call embedding models. Use --validate-only to avoid model loads.
This Cursor implementation phase should prefer --validate-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.checksums import sha256_file  # noqa: E402
from external_baselines.common.experiment_manifest import (  # noqa: E402
    build_method_config,
    enabled_methods,
    load_experiment_manifest,
)
from external_baselines.common.formal_config_validator import _is_placeholder  # noqa: E402
from external_baselines.common.io import write_json  # noqa: E402
from external_baselines.interop.bundle import load_runner_bundle, validate_bundle_checksum  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build/validate comparison indexes.")
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--method-set", choices=["main_table", "comparison_suite"], default="comparison_suite")
    parser.add_argument("--validate-only", action="store_true", help="Do not load embedding models.")
    parser.add_argument(
        "--output",
        default="outputs/diagnostics/comparison_index_report.json",
    )
    args = parser.parse_args(argv)

    experiment = load_experiment_manifest(args.experiment_manifest)
    bundle_path = args.bundle or experiment.get("bundle")
    report: dict = {
        "validate_only": bool(args.validate_only),
        "method_set": args.method_set,
        "bundle": None,
        "indexes": {},
        "errors": [],
    }
    if not bundle_path or _is_placeholder(bundle_path):
        report["errors"].append("bundle_placeholder_or_missing")
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if args.validate_only else 1)

    bundle = load_runner_bundle(bundle_path)
    checksum = validate_bundle_checksum(bundle)
    report["bundle"] = {
        "path": str(bundle_path),
        "checksum_ok": bool(checksum.get("ok")),
        "consumer_computed_bundle_hash": bundle.get("consumer_computed_bundle_hash"),
        "corpus_dir": bundle.get("corpus_dir"),
    }
    methods = enabled_methods(experiment, method_set=args.method_set)
    for entry in methods:
        mid = entry["method_id"]
        cfg = build_method_config(experiment, entry)
        if bundle.get("corpus_dir"):
            cfg.setdefault("paths", {})["corpus_dir"] = bundle["corpus_dir"]
        if mid == "dense_rag":
            dense = cfg.get("dense_rag") or {}
            index_path = dense.get("index_path")
            status = {
                "configured_index_path": index_path,
                "exists": bool(index_path) and Path(str(index_path)).exists(),
                "backend": dense.get("backend"),
                "model_name": dense.get("model_name"),
                "model_version": dense.get("model_version"),
            }
            if args.validate_only:
                if _is_placeholder(dense.get("model_version")):
                    status["error"] = "model_version_placeholder"
                report["indexes"]["dense"] = status
                continue
            # Real build path (not executed in this Cursor phase by default).
            from external_baselines.dense_rag.pipeline import build_dense_index
            from external_baselines.retrieval.embedding_backends import resolve_dimension

            evidence = Path(cfg.get("paths", {}).get("corpus_dir", "data/corpus")) / "evidence_chunks.jsonl"
            index = build_dense_index(
                evidence,
                model_name=str(dense.get("model_name")),
                model_version=str(dense.get("model_version")),
                backend=str(dense.get("backend", "text2vec")),
                dim=resolve_dimension(dense, 1024),
                cache_path=index_path,
                batch_size=int(dense.get("batch_size", 16)),
                normalize_embeddings=bool(dense.get("normalize_embeddings", True)),
                paper_final=bool(cfg.get("paper_final")),
                reject_smoke=bool(dense.get("reject_smoke", True)),
                corpus_checksum=sha256_file(evidence),
            )
            status["checksum"] = index.checksum
            status["built"] = True
            report["indexes"]["dense"] = status
        elif mid == "hybrid_rag":
            dense = cfg.get("dense_rag") or {}
            report["indexes"]["hybrid_dense_dependency"] = {
                "reuses_dense_index_path": dense.get("index_path"),
                "note": "Hybrid reuses Dense evidence index; no separate hybrid index.",
            }
        elif mid == "ekell_style_controlled_shared_llm":
            vector = cfg.get("ekell_vector") or {}
            report["indexes"]["ekell"] = {
                "configured_index_path": vector.get("index_path"),
                "exists": bool(vector.get("index_path")) and Path(str(vector.get("index_path"))).exists(),
                "backend": vector.get("backend"),
                "model_name": vector.get("model_name"),
                "model_version": vector.get("model_version"),
                "note": "E-KELL KG/entity index remains separate from Dense evidence index.",
            }

    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
