from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from external_baselines.common.io import load_config, load_scenarios, write_jsonl
from external_baselines.common.llm_client import build_llm_client
from external_baselines.evaluation.metrics import aggregate_metrics, score_output
from external_baselines.evaluation.report import build_report, write_metrics_csv


def get_pipeline(method: str) -> Callable[..., dict[str, Any]]:
    method = method.lower().strip()
    if method == "direct_llm":
        from external_baselines.direct_llm.pipeline import run_scenario
        return run_scenario
    if method == "vanilla_rag":
        from external_baselines.vanilla_rag.pipeline import run_scenario
        return run_scenario
    if method in {"ekell_style", "e-kell-style", "ekell"}:
        from external_baselines.ekell_style.pipeline import run_scenario
        return run_scenario
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


def run_methods(
    *,
    methods: list[str],
    dataset: str | Path,
    config_paths: list[str | Path] | None = None,
    limit: int | None = None,
    output_path: str | Path = "outputs/baseline_outputs.jsonl",
    metrics_path: str | Path = "outputs/baseline_metrics.csv",
    report_path: str | Path = "outputs/baseline_report.md",
) -> list[dict[str, Any]]:
    config = load_config("configs/default.yaml", *(config_paths or []))
    # Ensure dataset path is reflected in config for report/debugging.
    config.setdefault("paths", {})["scenario_file"] = str(dataset)
    scenarios = load_scenarios(dataset, limit=limit)
    llm = build_llm_client(config)

    outputs: list[dict[str, Any]] = []
    for method in methods:
        pipeline = get_pipeline(method)
        for scenario in scenarios:
            out = pipeline(scenario, config=config, llm=llm)
            outputs.append(out)

    write_jsonl(output_path, outputs)
    scored = []
    expected_by_id = {s["scenario_id"]: s.get("expected", {}) for s in scenarios}
    for out in outputs:
        scored.append(score_output(out, expected_by_id.get(str(out.get("scenario_id")), {})))
    aggregated = aggregate_metrics(scored)
    write_metrics_csv(metrics_path, aggregated)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(build_report(outputs, aggregated), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run external baseline pipelines.")
    parser.add_argument("--methods", default="direct_llm,vanilla_rag,ekell_style")
    parser.add_argument("--method", default=None, help="Single method alias; overrides --methods when provided.")
    parser.add_argument("--dataset", default="data/scenarios/scenario_matrix_v2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--output", default="outputs/baseline_outputs.jsonl")
    parser.add_argument("--metrics", default="outputs/baseline_metrics.csv")
    parser.add_argument("--report", default="outputs/baseline_report.md")
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
    )
    print(f"Wrote {len(outputs)} baseline outputs to {args.output}")
    print(f"Wrote metrics to {args.metrics}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
