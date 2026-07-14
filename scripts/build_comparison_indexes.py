#!/usr/bin/env python3
"""Build or fail-closed validate Dense / Hybrid / E-KELL comparison indexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.experiment_manifest import (  # noqa: E402
    build_method_config,
    enabled_methods,
    load_experiment_manifest,
)
from external_baselines.common.formal_config_validator import (  # noqa: E402
    _is_placeholder,
    validate_experiment_manifest,
    validate_method_config,
)
from external_baselines.common.io import write_json  # noqa: E402
from external_baselines.common.path_resolution import (  # noqa: E402
    PathContext,
    resolve_declared_path,
)
from external_baselines.common.strict_config_types import (  # noqa: E402
    require_exact_bool,
    require_exact_int,
    require_exact_nonempty_string,
)
from external_baselines.ekell_style.kg_loader import (  # noqa: E402
    fire_kg_checksum,
    load_kg_strict,
)
from external_baselines.interop.bundle import (  # noqa: E402
    load_runner_bundle,
    runner_bundle_corpus_aggregate_sha256,
    runner_bundle_evidence_source_checksum,
    validate_bundle_checksum,
    validate_formal_bundle_aggregate_checksum,
)


def _record_error(
    report: dict[str, Any],
    status: dict[str, Any] | None,
    code: str,
) -> None:
    if status is not None:
        status["error"] = code
    if code not in report["errors"]:
        report["errors"].append(code)


def _finalize_report(report: dict[str, Any], output: str | Path) -> None:
    report["ok"] = not report["errors"]
    write_json(output, report)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def _resolved_repository_path(
    value: str,
    *,
    expected_kind: str = "either",
    must_exist: bool = False,
) -> Path:
    return resolve_declared_path(
        value,
        context=PathContext(repository_root=ROOT),
        policy="repository_relative",
        must_exist=must_exist,
        expected_kind=expected_kind,  # type: ignore[arg-type]
    )


def _embedding_settings(
    config: dict[str, Any],
    block: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    paper_final = require_exact_bool(config.get("paper_final"), field="paper_final")
    if paper_final is not True:
        raise ValueError("paper_final_must_be_true")
    settings = {
        "backend": require_exact_nonempty_string(
            block.get("backend"), field=f"{prefix}.backend"
        ),
        "model_name": require_exact_nonempty_string(
            block.get("model_name"), field=f"{prefix}.model_name"
        ),
        "model_version": require_exact_nonempty_string(
            block.get("model_version"), field=f"{prefix}.model_version"
        ),
        "dimension": require_exact_int(
            block.get("dimension"), field=f"{prefix}.dimension", minimum=1
        ),
        "batch_size": require_exact_int(
            block.get("batch_size", 16), field=f"{prefix}.batch_size", minimum=1
        ),
        "reject_smoke": require_exact_bool(
            block.get("reject_smoke"), field=f"{prefix}.reject_smoke"
        ),
        "normalize_embeddings": require_exact_bool(
            block.get("normalize_embeddings"),
            field=f"{prefix}.normalize_embeddings",
        ),
        "index_path": require_exact_nonempty_string(
            block.get("index_path"), field=f"{prefix}.index_path"
        ),
        "paper_final": paper_final,
    }
    return settings


def _require_index_identity(payload: dict[str, Any]) -> None:
    if not payload.get("index_checksum"):
        raise ValueError("index_checksum_missing")
    if not payload.get("index_manifest_sha256"):
        raise ValueError("index_manifest_sha256_missing")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build/validate comparison indexes.")
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument(
        "--method-set",
        choices=["main_table", "comparison_suite"],
        default="comparison_suite",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not load embedding models.",
    )
    parser.add_argument(
        "--output",
        default="outputs/diagnostics/comparison_index_report.json",
    )
    args = parser.parse_args(argv)

    output_path = _resolved_repository_path(args.output)
    report: dict[str, Any] = {
        "ok": False,
        "validate_only": args.validate_only is True,
        "method_set": args.method_set,
        "bundle": {},
        "experiment_validation": {
            "stage": "index_build_candidate",
            "ok": False,
            "errors": [],
        },
        "indexes": {},
        "errors": [],
        "warnings": [],
    }
    try:
        experiment_path = _resolved_repository_path(args.experiment_manifest)
        experiment = load_experiment_manifest(experiment_path)
    except Exception as exc:  # noqa: BLE001
        _record_error(report, None, f"experiment_manifest_load_failed:{exc}")
        _finalize_report(report, output_path)
        return

    bundle_declared = args.bundle or experiment.get("bundle_declared")
    if not bundle_declared or _is_placeholder(bundle_declared):
        _record_error(report, None, "bundle_placeholder_or_missing")
        _finalize_report(report, output_path)
        return
    try:
        bundle_value = require_exact_nonempty_string(bundle_declared, field="bundle")
        bundle_path = (
            _resolved_repository_path(bundle_value)
            if args.bundle
            else Path(str(experiment.get("bundle_resolved") or ""))
        )
        bundle = load_runner_bundle(bundle_path, formal=True)
    except Exception as exc:  # noqa: BLE001
        _record_error(report, None, f"runner_bundle_load_failed:{exc}")
        _finalize_report(report, output_path)
        return

    try:
        validate_formal_bundle_aggregate_checksum(bundle)
    except Exception as exc:  # noqa: BLE001
        _record_error(report, None, f"runner_bundle_aggregate_checksum_failed:{exc}")
        _finalize_report(report, output_path)
        return
    try:
        checksum = validate_bundle_checksum(bundle)
    except Exception as exc:  # noqa: BLE001
        checksum = {"ok": False, "error": str(exc)}
    if checksum.get("ok") is not True:
        report["bundle"] = {
            "path": str(bundle_path),
            "checksum_ok": False,
            "checksum": checksum,
        }
        _record_error(
            report,
            None,
            "runner_bundle_file_checksum_validation_failed",
        )
        _finalize_report(report, output_path)
        return

    try:
        validation = validate_experiment_manifest(
            experiment_path,
            validation_stage="index_build_candidate",
            method_set=args.method_set,
            runtime_bundle_path=bundle_path,
        )
        report["experiment_validation"] = {
            "stage": "index_build_candidate",
            "ok": True,
            "errors": [],
            "resource_paths": validation.get("resource_paths") or {},
        }
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        report["experiment_validation"] = {
            "stage": "index_build_candidate",
            "ok": False,
            "errors": [message],
        }
        _record_error(
            report,
            None,
            f"experiment_index_build_candidate_validation_failed:{message}",
        )
        _finalize_report(report, output_path)
        return

    try:
        bundle_corpus_checksum = runner_bundle_corpus_aggregate_sha256(
            bundle,
            required=True,
        )
    except Exception as exc:  # noqa: BLE001
        _record_error(report, None, f"runner_bundle_corpus_identity_invalid:{exc}")
        _finalize_report(report, output_path)
        return
    try:
        evidence_source_checksum = runner_bundle_evidence_source_checksum(
            bundle,
            required=True,
        )
    except Exception as exc:  # noqa: BLE001
        _record_error(report, None, f"runner_bundle_evidence_source_invalid:{exc}")
        _finalize_report(report, output_path)
        return
    try:
        corpus_dir = Path(
            require_exact_nonempty_string(bundle.get("corpus_dir"), field="bundle.corpus_dir")
        ).resolve(strict=False)
        kg = load_kg_strict(corpus_dir)
        canonical_kg_checksum = fire_kg_checksum(kg)
    except Exception as exc:  # noqa: BLE001
        _record_error(report, None, f"runner_bundle_kg_identity_invalid:{exc}")
        _finalize_report(report, output_path)
        return

    report["bundle"] = {
        "path": str(bundle_path),
        "checksum_ok": True,
        "consumer_computed_bundle_hash": bundle.get("consumer_computed_bundle_hash"),
        "corpus_dir": str(corpus_dir),
        "bundle_corpus_aggregate_sha256": bundle_corpus_checksum,
        "evidence_source_checksum": evidence_source_checksum,
        "kg_checksum": canonical_kg_checksum,
    }

    try:
        methods = enabled_methods(experiment, method_set=args.method_set)
    except Exception as exc:  # noqa: BLE001
        _record_error(report, None, f"method_set_resolution_failed:{exc}")
        _finalize_report(report, output_path)
        return

    dense_status: dict[str, Any] | None = None
    dense_config: dict[str, Any] | None = None
    dense_index_path: Path | None = None
    for entry in methods:
        mid = entry["method_id"]
        try:
            cfg = build_method_config(experiment, entry)
            cfg.setdefault("paths", {})["corpus_dir"] = str(corpus_dir)
        except Exception as exc:  # noqa: BLE001
            _record_error(report, None, f"{mid}_config_merge_failed:{exc}")
            continue

        if mid == "dense_rag":
            dense_config = cfg
            dense = cfg.get("dense_rag") or {}
            status: dict[str, Any] = {
                "configured_index_path": dense.get("index_path"),
                "backend": dense.get("backend"),
                "model_name": dense.get("model_name"),
                "model_version": dense.get("model_version"),
            }
            report["indexes"]["dense"] = status
            dense_status = status
            if _is_placeholder(dense.get("model_version")):
                _record_error(report, status, "dense_model_version_placeholder")
                continue
            try:
                settings = _embedding_settings(cfg, dense, prefix="dense_rag")
                validate_method_config(
                    cfg,
                    method_id=mid,
                    allow_placeholders=False,
                    require_formal=True,
                    validation_stage="index_build_candidate",
                    validate_index_integrity=False,
                )
            except Exception as exc:  # noqa: BLE001
                _record_error(report, status, f"dense_config_invalid:{exc}")
                continue
            if _is_placeholder(settings["index_path"]):
                _record_error(report, status, "dense_index_path_missing")
                continue
            dense_index_path = _resolved_repository_path(settings["index_path"])
            status["resolved_index_path"] = str(dense_index_path)
            status["exists"] = dense_index_path.exists()
            if args.validate_only:
                if not dense_index_path.exists():
                    _record_error(report, status, "dense_index_path_missing")
                    continue
                if not dense_index_path.is_dir():
                    _record_error(report, status, "dense_index_path_not_directory")
                    continue
                try:
                    from external_baselines.retrieval.dense_index import (
                        validate_dense_index_integrity_for_freeze,
                    )

                    identity = validate_dense_index_integrity_for_freeze(
                        dense_index_path,
                        expected_model_name=settings["model_name"],
                        expected_model_version=settings["model_version"],
                        expected_backend=settings["backend"],
                        expected_dimension=settings["dimension"],
                        expected_corpus_checksum=bundle_corpus_checksum,
                        expected_evidence_source_checksum=evidence_source_checksum,
                        expected_normalize_embeddings=settings["normalize_embeddings"],
                    )
                    _require_index_identity(identity)
                    status.update(identity)
                    status["validated"] = True
                except Exception as exc:  # noqa: BLE001
                    _record_error(
                        report,
                        status,
                        f"dense_index_validation_failed:{exc}",
                    )
                continue
            try:
                from external_baselines.dense_rag.pipeline import build_dense_index
                from external_baselines.retrieval.dense_index import (
                    validate_dense_index_integrity_for_freeze,
                )

                build_dense_index(
                    corpus_dir / "evidence_chunks.jsonl",
                    model_name=settings["model_name"],
                    model_version=settings["model_version"],
                    backend=settings["backend"],
                    dim=settings["dimension"],
                    cache_path=dense_index_path,
                    batch_size=settings["batch_size"],
                    normalize_embeddings=settings["normalize_embeddings"],
                    paper_final=settings["paper_final"],
                    reject_smoke=settings["reject_smoke"],
                    corpus_checksum=bundle_corpus_checksum,
                )
                identity = validate_dense_index_integrity_for_freeze(
                    dense_index_path,
                    expected_backend=settings["backend"],
                    expected_model_name=settings["model_name"],
                    expected_model_version=settings["model_version"],
                    expected_dimension=settings["dimension"],
                    expected_corpus_checksum=bundle_corpus_checksum,
                    expected_evidence_source_checksum=evidence_source_checksum,
                    expected_normalize_embeddings=settings["normalize_embeddings"],
                )
                _require_index_identity(identity)
                status.update(identity)
                status.update({"built": True, "validated": True})
            except Exception as exc:  # noqa: BLE001
                _record_error(report, status, f"dense_index_build_failed:{exc}")

        elif mid == "hybrid_rag":
            status = {
                "reuses_dense_index_path": (
                    (cfg.get("dense_rag") or {}).get("index_path")
                    or (cfg.get("hybrid_rag") or {}).get("dense_index_path")
                ),
                "note": "Hybrid reuses Dense evidence index; no separate hybrid index.",
            }
            report["indexes"]["hybrid_dense_dependency"] = status
            try:
                validate_method_config(
                    cfg,
                    method_id=mid,
                    allow_placeholders=False,
                    require_formal=True,
                    validation_stage="index_build_candidate",
                    dense_config=dense_config,
                    validate_index_integrity=False,
                )
            except Exception as exc:  # noqa: BLE001
                _record_error(report, status, f"hybrid_config_invalid:{exc}")
                continue
            if dense_status is None or dense_status.get("validated") is not True:
                _record_error(report, status, "hybrid_dense_dependency_missing")
                continue
            hybrid_declared = require_exact_nonempty_string(
                status["reuses_dense_index_path"],
                field="hybrid_rag.dense_index_path",
            )
            hybrid_path = _resolved_repository_path(hybrid_declared)
            if dense_index_path is None or hybrid_path != dense_index_path:
                _record_error(report, status, "hybrid_dense_dependency_not_validated")
                continue
            status.update(
                {
                    "checksum": dense_status.get("index_checksum"),
                    "index_manifest_sha256": dense_status.get("index_manifest_sha256"),
                    "validated": True,
                }
            )

        elif mid == "ekell_style_controlled_shared_llm":
            vector = cfg.get("ekell_vector") or {}
            status = {
                "configured_index_path": vector.get("index_path"),
                "backend": vector.get("backend"),
                "model_name": vector.get("model_name"),
                "model_version": vector.get("model_version"),
                "note": "E-KELL KG/entity index remains separate from Dense evidence index.",
            }
            report["indexes"]["ekell"] = status
            if _is_placeholder(vector.get("model_version")):
                _record_error(report, status, "ekell_model_version_placeholder")
                continue
            try:
                settings = _embedding_settings(cfg, vector, prefix="ekell_vector")
                validate_method_config(
                    cfg,
                    method_id=mid,
                    allow_placeholders=False,
                    require_formal=True,
                    validation_stage="index_build_candidate",
                    validate_index_integrity=False,
                )
                configured_kg_checksum = cfg.get("kg_checksum")
                if configured_kg_checksum is not None:
                    configured_kg_checksum = require_exact_nonempty_string(
                        configured_kg_checksum,
                        field="kg_checksum",
                    )
                    if configured_kg_checksum != canonical_kg_checksum:
                        raise ValueError("ekell_config_kg_checksum_mismatch")
            except Exception as exc:  # noqa: BLE001
                code = (
                    "ekell_config_kg_checksum_mismatch"
                    if str(exc) == "ekell_config_kg_checksum_mismatch"
                    else f"ekell_config_invalid:{exc}"
                )
                _record_error(report, status, code)
                continue
            if _is_placeholder(settings["index_path"]):
                _record_error(report, status, "ekell_index_path_missing")
                continue
            index_path = _resolved_repository_path(settings["index_path"])
            status["resolved_index_path"] = str(index_path)
            status["exists"] = index_path.exists()
            if args.validate_only:
                if not index_path.exists():
                    _record_error(report, status, "ekell_index_path_missing")
                    continue
                if not index_path.is_dir():
                    _record_error(report, status, "ekell_index_path_not_directory")
                    continue
                try:
                    from external_baselines.ekell_style.vector_index import VectorIndex

                    identity = VectorIndex.validate_directory_for_freeze(
                        index_path,
                        expected_backend=settings["backend"],
                        expected_model_name=settings["model_name"],
                        expected_model_version=settings["model_version"],
                        expected_dimension=settings["dimension"],
                        expected_corpus_checksum=bundle_corpus_checksum,
                        expected_kg_checksum=canonical_kg_checksum,
                        expected_normalize_embeddings=settings["normalize_embeddings"],
                    )
                    _require_index_identity(identity)
                    status.update(identity)
                    status["validated"] = True
                except Exception as exc:  # noqa: BLE001
                    _record_error(
                        report,
                        status,
                        f"ekell_index_validation_failed:{exc}",
                    )
                continue
            try:
                from external_baselines.ekell_style.vector_index import VectorIndex
                from external_baselines.retrieval.embedding_backends import (
                    create_embedding_backend,
                )

                backend = create_embedding_backend(
                    settings["backend"],
                    model_name=settings["model_name"],
                    model_version=settings["model_version"],
                    dimension=settings["dimension"],
                    paper_final=settings["paper_final"],
                    reject_smoke=settings["reject_smoke"],
                )
                index = VectorIndex.from_kg(
                    kg,
                    backend,
                    corpus_checksum=bundle_corpus_checksum,
                    kg_checksum=canonical_kg_checksum,
                    paper_final=settings["paper_final"],
                    reject_smoke=settings["reject_smoke"],
                    normalize_embeddings=settings["normalize_embeddings"],
                )
                index.save_directory(index_path)
                identity = VectorIndex.validate_directory_for_freeze(
                    index_path,
                    expected_backend=settings["backend"],
                    expected_model_name=settings["model_name"],
                    expected_model_version=settings["model_version"],
                    expected_dimension=settings["dimension"],
                    expected_corpus_checksum=bundle_corpus_checksum,
                    expected_kg_checksum=canonical_kg_checksum,
                    expected_normalize_embeddings=settings["normalize_embeddings"],
                )
                _require_index_identity(identity)
                status.update(identity)
                status.update({"built": True, "validated": True})
            except Exception as exc:  # noqa: BLE001
                _record_error(report, status, f"ekell_index_build_failed:{exc}")

    _finalize_report(report, output_path)


if __name__ == "__main__":
    main()
