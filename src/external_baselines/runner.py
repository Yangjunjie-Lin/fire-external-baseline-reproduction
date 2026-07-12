from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from external_baselines.common.guards import assert_paper_final_allowed
from external_baselines.common.io import (
    assert_no_gold_in_prediction_input,
    load_config,
    load_expected_by_id,
    load_scenarios,
    to_prediction_input,
    write_jsonl,
)
from external_baselines.common.llm_client import TokenUsage, UsageTrackingLLMClient, build_llm_client
from external_baselines.common.manifest import build_run_manifest, write_run_manifest
from external_baselines.common.method_runtime import (
    close_method_runtime,
    pipeline_accepts_runtime,
    prepare_method_runtime,
    runtime_index_checksum,
)
from external_baselines.ekell_style.kg_loader import audit_corpus
from external_baselines.evaluation.metrics import aggregate_metrics, score_output
from external_baselines.evaluation.report import build_report, write_metrics_csv
from external_baselines.interop.schema import canonicalize_method_id


def get_pipeline(method: str) -> Callable[..., dict[str, Any]]:
    from external_baselines.method_registry import resolve_pipeline

    return resolve_pipeline(method)


def _data_counts(corpus_dir: str | Path) -> dict[str, Any]:
    audit = audit_corpus(corpus_dir)
    return {
        "entities": audit["entity_count"],
        "relations": audit["relation_count"],
        "triples": audit["triple_count"],
        "evidence_chunks": audit["evidence_chunk_count"],
        "missing_files": audit["missing_files"],
        "schema_warning_count": len(audit["schema_warnings"]),
    }


def generate_predictions(
    *,
    methods: list[str],
    dataset: str | Path,
    config_paths: list[str | Path] | None = None,
    limit: int | None = None,
    output_path: str | Path = "outputs/baseline_outputs.jsonl",
    manifest_path: str | Path | None = "outputs/run_manifest.json",
    config: dict[str, Any] | None = None,
    method_configs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate predictions with gold-isolated inputs only.

    If ``method_configs`` is provided, each method uses its own merged config
    (shared model + method-specific). Otherwise a single ``config`` is used.
    """
    default_config = config or load_config("configs/default.yaml", *(config_paths or []))
    assert_paper_final_allowed(default_config)
    default_config.setdefault("paths", {})["scenario_file"] = str(dataset)
    scenarios = load_scenarios(dataset, limit=limit)
    corpus_dir = default_config.get("paths", {}).get("corpus_dir", "data/corpus")

    outputs: list[dict[str, Any]] = []
    run_usage_by_method: dict[str, dict[str, Any]] = {}
    runtime_metadata_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_id = canonicalize_method_id(method)
        method_config = (method_configs or {}).get(method_id) or default_config
        assert_paper_final_allowed(method_config)
        llm = build_llm_client(method_config)
        pipeline = get_pipeline(method_id)
        runtime = prepare_method_runtime(method_id, method_config)
        accepts_runtime = pipeline_accepts_runtime(pipeline)
        try:
            for scenario in scenarios:
                usage_before = (
                    llm.usage_snapshot()
                    if isinstance(llm, UsageTrackingLLMClient)
                    else TokenUsage()
                )
                prediction_input = to_prediction_input(scenario, config=method_config)
                assert_no_gold_in_prediction_input(prediction_input)
                if accepts_runtime and runtime is not None:
                    out = pipeline(prediction_input, config=method_config, llm=llm, runtime=runtime)
                else:
                    out = pipeline(prediction_input, config=method_config, llm=llm)
                case_usage = (
                    llm.usage_delta(usage_before)
                    if isinstance(llm, UsageTrackingLLMClient)
                    else TokenUsage()
                )
                runtime_block = out.setdefault("method_specific", {}).setdefault("runtime", {})
                runtime_block.update({
                    "llm_calls": case_usage.llm_calls,
                    "token_usage": case_usage.to_dict(),
                    "case_llm_calls": case_usage.llm_calls,
                    "case_prompt_tokens": case_usage.prompt_tokens,
                    "case_completion_tokens": case_usage.completion_tokens,
                    "case_total_tokens": case_usage.total_tokens,
                })
                out["method"] = canonicalize_method_id(str(out.get("method") or method_id))
                out["scenario_id"] = prediction_input["scenario_id"]
                outputs.append(out)
        finally:
            close_method_runtime(runtime)

        if runtime is not None and getattr(runtime, "audit", None) is not None:
            audit = runtime.audit.to_dict()
            audit["index_checksum"] = runtime_index_checksum(runtime)
            runtime_metadata_by_method[method_id] = audit
        if isinstance(llm, UsageTrackingLLMClient):
            run_usage_by_method[method_id] = llm.usage.snapshot().to_dict()

    write_jsonl(output_path, outputs)
    if manifest_path:
        manifest = build_run_manifest(
            methods=[canonicalize_method_id(m) for m in methods],
            dataset=dataset,
            limit=limit,
            corpus_dir=corpus_dir,
            config=default_config,
            data_counts=_data_counts(corpus_dir),
        )
        manifest["run_usage_by_method"] = run_usage_by_method
        manifest["run_usage_total"] = {
            key: sum(int(value.get(key, 0)) for value in run_usage_by_method.values())
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls")
        }
        manifest["runtime_reuse_by_method"] = runtime_metadata_by_method
        write_run_manifest(manifest_path, manifest)
    return outputs

def evaluate_predictions(
    *,
    predictions_path: str | Path,
    dataset: str | Path,
    metrics_path: str | Path = "outputs/baseline_metrics.csv",
    report_path: str | Path = "outputs/baseline_report.md",
    manifest_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate previously generated predictions (proxy diagnostics only)."""
    from external_baselines.common.io import read_jsonl

    outputs = read_jsonl(predictions_path)
    expected_by_id = load_expected_by_id(dataset, limit=limit)
    scored = [score_output(out, expected_by_id.get(str(out.get("scenario_id")), {})) for out in outputs]
    aggregated = aggregate_metrics(scored)
    write_metrics_csv(metrics_path, aggregated)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    manifest = None
    if manifest_path and Path(manifest_path).exists():
        from external_baselines.common.io import read_json
        manifest = read_json(manifest_path)
    Path(report_path).write_text(build_report(outputs, aggregated, manifest=manifest), encoding="utf-8")
    return {"aggregated": aggregated, "n": len(outputs)}


def run_methods(
    *,
    methods: list[str],
    dataset: str | Path,
    config_paths: list[str | Path] | None = None,
    limit: int | None = None,
    output_path: str | Path = "outputs/baseline_outputs.jsonl",
    metrics_path: str | Path = "outputs/baseline_metrics.csv",
    report_path: str | Path = "outputs/baseline_report.md",
    manifest_path: str | Path = "outputs/run_manifest.json",
    evaluate: bool = True,
) -> list[dict[str, Any]]:
    """Backward-compatible entry: generate then optionally evaluate (proxy metrics)."""
    outputs = generate_predictions(
        methods=methods,
        dataset=dataset,
        config_paths=config_paths,
        limit=limit,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    if evaluate:
        evaluate_predictions(
            predictions_path=output_path,
            dataset=dataset,
            metrics_path=metrics_path,
            report_path=report_path,
            manifest_path=manifest_path,
            limit=limit,
        )
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run external baseline pipelines.")
    parser.add_argument(
        "--methods",
        default="direct_llm,bm25_rag,ekell_style_controlled_shared_llm",
    )
    parser.add_argument("--method", default=None, help="Single method alias; overrides --methods when provided.")
    parser.add_argument("--dataset", default="data/scenarios/scenario_matrix_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--output", default="outputs/baseline_outputs.jsonl")
    parser.add_argument("--metrics", default="outputs/baseline_metrics.csv")
    parser.add_argument("--report", default="outputs/baseline_report.md")
    parser.add_argument("--manifest", default="outputs/run_manifest.json")
    parser.add_argument("--generate-only", action="store_true", help="Skip proxy evaluation.")
    args = parser.parse_args(argv)

    methods = [args.method] if args.method else [m.strip() for m in args.methods.split(",") if m.strip()]
    outputs = run_methods(
        methods=methods,
        dataset=args.dataset,
        config_paths=args.config,
        limit=args.limit,
        output_path=args.output,
        metrics_path=args.metrics,
        report_path=args.report,
        manifest_path=args.manifest,
        evaluate=not args.generate_only,
    )
    print(f"Wrote {len(outputs)} baseline outputs to {args.output}")
    if not args.generate_only:
        print(f"Wrote metrics to {args.metrics}")
        print(f"Wrote report to {args.report}")
    print(f"Wrote run manifest to {args.manifest}")


if __name__ == "__main__":
    main()
