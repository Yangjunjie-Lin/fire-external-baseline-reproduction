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
from external_baselines.common.strict_config_types import require_exact_bool  # noqa: E402
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
    corpus_dir = Path(str(bundle.get("corpus_dir") or "data/corpus"))
    methods = enabled_methods(experiment, method_set=args.method_set)
    dense_checksum = None
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
                elif index_path and Path(str(index_path)).is_dir():
                    try:
                        from external_baselines.retrieval.dense_index import load_dense_index

                        payload = load_dense_index(
                            index_path,
                            expected_model_name=str(dense.get("model_name")) if dense.get("model_name") else None,
                            expected_model_version=str(dense.get("model_version"))
                            if dense.get("model_version") and not _is_placeholder(dense.get("model_version"))
                            else None,
                            expected_backend=str(dense.get("backend")) if dense.get("backend") else None,
                        )
                        status["checksum"] = payload.get("checksum")
                        status["validated"] = True
                        dense_checksum = payload.get("checksum")
                    except Exception as exc:  # noqa: BLE001
                        status["error"] = str(exc)
                        report["errors"].append(f"dense_validate:{exc}")
                report["indexes"]["dense"] = status
                continue
            from external_baselines.dense_rag.pipeline import build_dense_index
            from external_baselines.retrieval.embedding_backends import resolve_dimension

            evidence = corpus_dir / "evidence_chunks.jsonl"
            index = build_dense_index(
                evidence,
                model_name=str(dense.get("model_name")),
                model_version=str(dense.get("model_version")),
                backend=str(dense.get("backend", "text2vec")),
                dim=resolve_dimension(dense, 1024),
                cache_path=index_path,
                batch_size=int(dense.get("batch_size", 16)),
                normalize_embeddings=require_exact_bool(
                    dense.get("normalize_embeddings"),
                    field="dense_rag.normalize_embeddings",
                ),
                paper_final=bool(cfg.get("paper_final")),
                reject_smoke=bool(dense.get("reject_smoke", True)),
                corpus_checksum=sha256_file(evidence),
            )
            status["checksum"] = index.checksum
            status["built"] = True
            dense_checksum = index.checksum
            report["indexes"]["dense"] = status
        elif mid == "hybrid_rag":
            dense = cfg.get("dense_rag") or {}
            report["indexes"]["hybrid_dense_dependency"] = {
                "reuses_dense_index_path": dense.get("index_path"),
                "checksum": dense_checksum,
                "note": "Hybrid reuses Dense evidence index; no separate hybrid index.",
            }
        elif mid == "ekell_style_controlled_shared_llm":
            vector = cfg.get("ekell_vector") or {}
            index_path = vector.get("index_path")
            status = {
                "configured_index_path": index_path,
                "exists": bool(index_path) and Path(str(index_path)).exists(),
                "backend": vector.get("backend"),
                "model_name": vector.get("model_name"),
                "model_version": vector.get("model_version"),
                "note": "E-KELL KG/entity index remains separate from Dense evidence index.",
            }
            if args.validate_only:
                if _is_placeholder(vector.get("model_version")):
                    status["error"] = "model_version_placeholder"
                elif index_path and Path(str(index_path)).is_dir():
                    try:
                        from external_baselines.ekell_style.vector_index import VectorIndex

                        loaded = VectorIndex.load_directory(
                            index_path,
                            expected_backend=str(vector.get("backend")) if vector.get("backend") else None,
                            expected_model_name=str(vector.get("model_name")) if vector.get("model_name") else None,
                            expected_model_version=str(vector.get("model_version"))
                            if vector.get("model_version") and not _is_placeholder(vector.get("model_version"))
                            else None,
                            require_real_embedding=True,
                        )
                        status["checksum"] = (loaded.metadata or {}).get("index_checksum")
                        status["validated"] = True
                    except Exception as exc:  # noqa: BLE001
                        status["error"] = str(exc)
                        report["errors"].append(f"ekell_validate:{exc}")
                report["indexes"]["ekell"] = status
                continue

            from external_baselines.ekell_style.kg_loader import load_kg
            from external_baselines.ekell_style.vector_index import VectorIndex
            from external_baselines.retrieval.embedding_backends import (
                create_embedding_backend,
                resolve_dimension,
            )

            kg = load_kg(corpus_dir)
            backend = create_embedding_backend(
                str(vector.get("backend", "text2vec")),
                model_name=str(vector.get("model_name") or ""),
                model_version=str(vector.get("model_version") or "unspecified"),
                dimension=resolve_dimension(vector, 1024),
                paper_final=bool(cfg.get("paper_final")),
                reject_smoke=bool(vector.get("reject_smoke", True)),
                normalize_embeddings=require_exact_bool(
                    vector.get("normalize_embeddings"),
                    field="ekell_vector.normalize_embeddings",
                ),
            )
            index = VectorIndex.from_kg(
                kg,
                backend,
                corpus_checksum=sha256_file(corpus_dir / "evidence_chunks.jsonl")
                if (corpus_dir / "evidence_chunks.jsonl").is_file()
                else None,
                paper_final=bool(cfg.get("paper_final")),
                reject_smoke=bool(vector.get("reject_smoke", True)),
            )
            if not index_path or _is_placeholder(index_path):
                status["error"] = "ekell_index_path_missing"
                report["errors"].append("ekell_index_path_missing")
            else:
                manifest = index.save_directory(index_path)
                reloaded = VectorIndex.load_directory(
                    index_path,
                    expected_backend=str(vector.get("backend")),
                    expected_model_name=str(vector.get("model_name")),
                    expected_model_version=str(vector.get("model_version")),
                    require_real_embedding=True,
                )
                status["checksum"] = manifest.get("index_checksum") or (reloaded.metadata or {}).get(
                    "index_checksum"
                )
                status["built"] = True
                status["reloaded"] = True
            report["indexes"]["ekell"] = status

    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if report["errors"] and not args.validate_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
