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
from external_baselines.common.llm_client import build_llm_client
from external_baselines.common.manifest import build_run_manifest, write_run_manifest
from external_baselines.ekell_style.kg_loader import audit_corpus
from external_baselines.evaluation.metrics import aggregate_metrics, score_output
from external_baselines.evaluation.report import build_report, write_metrics_csv
from external_baselines.interop.schema import baseline_row_to_interop, canonicalize_method_id


def get_pipeline(method: str) -> Callable[..., dict[str, Any]]:
    method = method.lower().strip()
    method = canonicalize_method_id(method)
    if method == "direct_llm":
        from external_baselines.direct_llm.pipeline import run_scenario
        return run_scenario
    if method in {"bm25_rag", "vanilla_rag"}:
        from external_baselines.vanilla_rag.pipeline import run_scenario
        return run_scenario
    if method == "dense_rag":
        from external_baselines.dense_rag.pipeline import run_scenario
        return run_scenario
    if method == "hybrid_rag":
        from external_baselines.hybrid_rag.pipeline import run_scenario
        return run_scenario
    if method in {"ekell_style_faithful", "ekell_style", "e-kell-style", "ekell"}:
        from external_baselines.ekell_style.pipeline import run_scenario_faithful
        return run_scenario_faithful
    if method == "ekell_style_enhanced":
        from external_baselines.ekell_style.enhanced_pipeline import run_scenario_enhanced
        return run_scenario_enhanced
    if method == "lightrag":
        from external_baselines.graphrag_adapter.lightrag_adapter import run_scenario
        return run_scenario
    if method in {"microsoft_graphrag", "graphrag"}:
        from external_baselines.graphrag_adapter.microsoft_graphrag_adapter import run_scenario
        return run_scenario
    if method == "fallback_graph_retrieval":
        from external_baselines.graphrag_adapter.fallback_graph_retrieval import run_scenario
        return run_scenario
    raise ValueError(f"Unknown baseline method: {method}")


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
    for method in methods:
        method_id = canonicalize_method_id(method)
        method_config = (method_configs or {}).get(method_id) or default_config
        assert_paper_final_allowed(method_config)
        llm = build_llm_client(method_config)
        pipeline = get_pipeline(method_id)
        for scenario in scenarios:
            prediction_input = to_prediction_input(scenario, config=method_config)
            assert_no_gold_in_prediction_input(prediction_input)
            out = pipeline(prediction_input, config=method_config, llm=llm)
            out["method"] = canonicalize_method_id(str(out.get("method") or method_id))
            out["scenario_id"] = prediction_input["scenario_id"]
            outputs.append(out)

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
    parser.add_argument("--methods", default="direct_llm,bm25_rag,ekell_style_faithful")
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
