#!/usr/bin/env python3
"""Create a freeze manifest from configs + selected DEV evidence.

Use --draft to allow incomplete fields. Without --draft, all formal fields are required.
"""

from __future__ import annotations

import argparse
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
from external_baselines.common.formal_config_validator import (  # noqa: E402
    FormalConfigError,
    _is_placeholder,
    validate_experiment_manifest,
)
from external_baselines.common.freeze_manifest import (  # noqa: E402
    build_freeze_manifest_payload,
    validate_freeze_manifest,
)
from external_baselines.common.io import read_json, write_json  # noqa: E402
from external_baselines.common.path_resolution import (  # noqa: E402
    PathContext,
    resolve_declared_path,
)
from external_baselines.common.strict_config_types import require_exact_bool  # noqa: E402
from external_baselines.ekell_style.kg_loader import (  # noqa: E402
    fire_kg_checksum,
    load_kg_strict,
)
from external_baselines.ekell_style.vector_index import VectorIndex  # noqa: E402
from external_baselines.interop.bundle import (  # noqa: E402
    load_runner_bundle,
    runner_bundle_corpus_aggregate_sha256,
    runner_bundle_evidence_source_checksum,
    validate_bundle_checksum,
    validate_formal_bundle_aggregate_checksum,
)
from external_baselines.method_registry import comparison_suite_methods  # noqa: E402
from external_baselines.retrieval.dense_index import validate_dense_index_integrity_for_freeze  # noqa: E402


def _load_index_block(index_dir: str | Path | None, *, kind: str) -> dict:
    if not index_dir or _is_placeholder(index_dir):
        return {}
    path = Path(str(index_dir))
    manifest_path = path / "index_manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        return {}
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        return {}
    block = dict(manifest)
    block.update({
        "index_checksum": manifest.get("index_checksum"),
        "index_manifest_sha256": sha256_file(manifest_path),
        "corpus_checksum": manifest.get("corpus_checksum"),
        "model_version": manifest.get("model_version"),
    })
    if kind == "ekell":
        block["kg_checksum"] = manifest.get("kg_checksum")
    return block


def _embedding_normalize_value(block: dict, *, field: str) -> bool:
    try:
        return require_exact_bool(block.get("normalize_embeddings"), field=field)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _repo_path(value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_declared_path(
        value,
        context=PathContext(repository_root=ROOT),
        policy="repository_relative",
        must_exist=must_exist,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create freeze manifest (draft or complete).")
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--selected-dev-run", required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Allow incomplete fields (do not claim a complete freeze).",
    )
    parser.add_argument(
        "--include-legacy-compat-fields",
        action="store_true",
        help="Also write legacy top-level runner_bundle_checksum and corpus/schema checksum fields.",
    )
    args = parser.parse_args(argv)

    evidence = _repo_path(args.selected_dev_run)
    if not evidence.is_file() or evidence.stat().st_size <= 0:
        raise SystemExit(f"selected DEV evidence missing or empty: {evidence}")

    experiment_manifest_path = _repo_path(args.experiment_manifest)
    experiment = load_experiment_manifest(experiment_manifest_path)
    raw = experiment.get("raw") or {}
    method_paths = {
        str(e.get("method_id")): str(e.get("config"))
        for e in (raw.get("methods") or [])
        if isinstance(e, dict) and e.get("method_id") and e.get("config")
    }
    for mid in comparison_suite_methods():
        method_paths.setdefault(mid, method_paths.get(mid) or "")

    bundle_checksum = None
    producer_declared_checksum = None
    consumer_computed_hash = None
    input_cases_sha256 = None
    corpus_checksum = None
    evidence_source_checksum = None
    bundle_kg_checksum = None
    schema_checksum = None
    bundle_path = args.bundle or experiment.get("bundle")
    bundle = None
    if bundle_path and not _is_placeholder(bundle_path):
        try:
            bundle_path = _repo_path(bundle_path)
            bundle = load_runner_bundle(bundle_path, formal=not args.draft)
            if not args.draft:
                validate_formal_bundle_aggregate_checksum(bundle)
                checksum_report = validate_bundle_checksum(bundle)
                if checksum_report.get("ok") is not True:
                    raise ValueError(
                        "runner_bundle_file_checksum_validation_failed"
                    )
            producer_declared_checksum = bundle.get("producer_declared_checksum")
            consumer_computed_hash = bundle.get("consumer_computed_bundle_hash")
            bundle_checksum = consumer_computed_hash
            schema_checksum = bundle.get("prediction_schema_sha256")
            scenarios_path = bundle.get("scenarios_path")
            if scenarios_path:
                input_cases_sha256 = bundle.get("input_cases_sha256") or sha256_file(scenarios_path)
            corpus_checksum = runner_bundle_corpus_aggregate_sha256(
                bundle,
                required=not args.draft,
            )
            evidence_source_checksum = runner_bundle_evidence_source_checksum(
                bundle,
                required=not args.draft,
            )
            corpus_dir = bundle.get("corpus_dir")
            if corpus_dir:
                bundle_kg_checksum = fire_kg_checksum(
                    load_kg_strict(corpus_dir)
                )
        except Exception as exc:  # noqa: BLE001
            if not args.draft:
                raise SystemExit(f"Failed to load Runner Bundle for freeze: {exc}") from exc
            print(f"WARNING: could not load bundle for checksums: {exc}", file=sys.stderr)
    elif not args.draft:
        raise SystemExit("Complete freeze requires a non-placeholder Runner Bundle path.")

    if not args.draft:
        try:
            validate_experiment_manifest(
                experiment_manifest_path,
                validation_stage="freeze_candidate",
                method_set="comparison_suite",
                runtime_bundle_path=bundle_path,
            )
        except FormalConfigError as exc:
            raise SystemExit(f"Freeze-candidate validation failed: {exc}") from exc

    methods = enabled_methods(experiment, method_set="comparison_suite")
    dense_index_path = None
    ekell_index_path = None
    embedding_candidates: dict[str, dict] = {}
    ekell_expected_kg_checksum = bundle_kg_checksum
    for entry in methods:
        cfg = build_method_config(experiment, entry)
        mid = entry["method_id"]
        if mid == "dense_rag":
            dense = cfg.get("dense_rag") or {}
            dense_index_path = dense.get("index_path")
            embedding_candidates[mid] = {
                "backend": dense.get("backend"),
                "model_name": dense.get("model_name"),
                "model_version": dense.get("model_version"),
                "dimension": dense.get("dimension"),
                "normalize_embeddings": _embedding_normalize_value(
                    dense,
                    field="dense_rag.normalize_embeddings",
                ),
            }
        elif mid == "hybrid_rag":
            dense = cfg.get("dense_rag") or {}
            hybrid = cfg.get("hybrid_rag") or {}
            if dense or hybrid:
                embedding_candidates[mid] = {
                    "backend": dense.get("backend") or hybrid.get("dense_method"),
                    "model_name": dense.get("model_name") or hybrid.get("dense_model_name"),
                    "model_version": dense.get("model_version")
                    or hybrid.get("dense_model_version"),
                    "dimension": dense.get("dimension") or hybrid.get("dimension"),
                    "normalize_embeddings": _embedding_normalize_value(
                        dense if "normalize_embeddings" in dense else hybrid,
                        field="hybrid_rag.normalize_embeddings",
                    ),
                }
        elif mid == "ekell_style_controlled_shared_llm":
            vector = cfg.get("ekell_vector") or {}
            ekell_index_path = vector.get("index_path")
            configured_kg_checksum = cfg.get("kg_checksum")
            if (
                configured_kg_checksum is not None
                and bundle_kg_checksum is not None
                and configured_kg_checksum != bundle_kg_checksum
            ):
                raise SystemExit("ekell_config_kg_checksum_mismatch")
            embedding_candidates[mid] = {
                "backend": vector.get("backend"),
                "model_name": vector.get("model_name"),
                "model_version": vector.get("model_version"),
                "dimension": vector.get("dimension"),
                "normalize_embeddings": _embedding_normalize_value(
                    vector,
                    field="ekell_vector.normalize_embeddings",
                ),
            }

    embedding = embedding_candidates.get("dense_rag") or next(
        iter(embedding_candidates.values()),
        {},
    )
    for mid, candidate in embedding_candidates.items():
        for field in (
            "backend",
            "model_name",
            "model_version",
            "dimension",
            "normalize_embeddings",
        ):
            if candidate.get(field) != embedding.get(field):
                if field == "normalize_embeddings":
                    raise SystemExit("cross_method_normalize_embeddings_mismatch")
                raise SystemExit(f"cross_method_embedding_identity_mismatch:{mid}:{field}")

    if not args.draft and embedding.get("model_version") and _is_placeholder(embedding.get("model_version")):
        raise SystemExit("embedding.model_version is still a placeholder; refuse non-draft freeze.")

    if args.draft:
        dense_block = _load_index_block(dense_index_path, kind="dense")
        ekell_block = _load_index_block(ekell_index_path, kind="ekell")
    else:
        if not dense_index_path or _is_placeholder(dense_index_path):
            raise SystemExit("Complete freeze requires non-placeholder Dense index path.")
        if not ekell_index_path or _is_placeholder(ekell_index_path):
            raise SystemExit("Complete freeze requires non-placeholder E-KELL index path.")
        dense_block = validate_dense_index_integrity_for_freeze(
            dense_index_path,
            expected_backend=embedding.get("backend"),
            expected_model_name=embedding.get("model_name"),
            expected_model_version=embedding.get("model_version"),
            expected_dimension=embedding.get("dimension"),
            expected_corpus_checksum=corpus_checksum,
            expected_evidence_source_checksum=evidence_source_checksum,
            expected_normalize_embeddings=embedding.get("normalize_embeddings"),
        )
        ekell_block = VectorIndex.validate_directory_for_freeze(
            ekell_index_path,
            expected_backend=embedding.get("backend"),
            expected_model_name=embedding.get("model_name"),
            expected_model_version=embedding.get("model_version"),
            expected_dimension=embedding.get("dimension"),
            expected_kg_checksum=str(ekell_expected_kg_checksum or "") or None,
            expected_corpus_checksum=corpus_checksum,
            expected_normalize_embeddings=embedding.get("normalize_embeddings"),
        )
    indexes = {
        "dense": dense_block,
        "hybrid_dense_dependency": {
            "index_checksum": dense_block.get("index_checksum"),
            "index_manifest_sha256": dense_block.get("index_manifest_sha256"),
        },
        "ekell": ekell_block,
    }

    payload = build_freeze_manifest_payload(
        experiment_manifest_path=experiment_manifest_path,
        experiment_raw=raw,
        selected_dev_run=evidence,
        producer_declared_checksum=producer_declared_checksum,
        consumer_computed_hash=consumer_computed_hash or bundle_checksum,
        input_cases_sha256=input_cases_sha256,
        corpus_checksum=corpus_checksum,
        schema_checksum=schema_checksum,
        method_config_paths=method_paths,
        indexes=indexes,
        embedding=embedding or None,
        producer_checksum_available=producer_declared_checksum is not None,
        include_legacy_compat_fields=bool(args.include_legacy_compat_fields),
    )
    provenance = payload.setdefault("path_provenance", {})
    for label, declared in (
        ("runner_bundle", bundle_path),
        ("dense_index", dense_index_path),
        ("ekell_index", ekell_index_path),
        ("freeze_manifest", args.output),
    ):
        if declared:
            resolved = _repo_path(declared)
            provenance[label] = {
                "declared_path": str(declared).replace("\\", "/"),
                "resolved_path": str(resolved),
                "path_policy": (
                    "absolute" if Path(declared).is_absolute() else "repository_relative"
                ),
            }
    if args.draft:
        payload["freeze_status"] = "draft"
        payload["draft"] = True
    else:
        # Ensure hybrid matches dense
        dense_cs = (payload.get("indexes") or {}).get("dense", {}).get("index_checksum")
        hybrid_cs = (payload.get("indexes") or {}).get("hybrid_dense_dependency", {}).get("index_checksum")
        if dense_cs and hybrid_cs and str(dense_cs) != str(hybrid_cs):
            raise SystemExit("Hybrid dense dependency checksum must equal Dense index checksum.")
        output_path = _repo_path(args.output)
        temp_path = output_path.with_name(f"{output_path.name}.tmp")
        try:
            write_json(temp_path, payload)
            validate_freeze_manifest(
                temp_path,
                experiment_manifest_path=experiment_manifest_path,
                experiment_raw=raw,
                require_complete=True,
                expected_runner_bundle_checksum=consumer_computed_hash or bundle_checksum,
                expected_corpus_checksum=corpus_checksum,
                expected_prediction_schema_checksum=schema_checksum,
                loaded_index_manifests={
                    "dense": dense_block,
                    "hybrid_dense_dependency": {
                        "index_checksum": dense_block.get("index_checksum"),
                        "index_manifest_sha256": dense_block.get("index_manifest_sha256"),
                    },
                    "ekell": ekell_block,
                },
                method_config_paths=method_paths,
            )
            temp_path.replace(output_path)
        except Exception as exc:  # noqa: BLE001
            temp_path.unlink(missing_ok=True)
            if isinstance(exc, FormalConfigError):
                raise SystemExit(f"Incomplete freeze manifest (use --draft to allow): {exc}") from exc
            raise SystemExit(f"Complete freeze manifest generation failed: {exc}") from exc
        print(f"Wrote complete freeze manifest to {output_path}")
        print("Confirm freeze_status=frozen in the experiment manifest only after human review.")
        return

    output_path = _repo_path(args.output)
    write_json(output_path, payload)
    print(f"Wrote freeze manifest draft to {output_path}")
    print("Manual confirmation still required before setting freeze_status=frozen.")


if __name__ == "__main__":
    main()
