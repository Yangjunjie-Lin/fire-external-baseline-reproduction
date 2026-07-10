# Comparison Protocol

## Goal

Fair **system-level** comparison between independent external baselines and the main project via `firebench-interop-v1`, using the same scenarios, corpus/KG snapshot, LLM config, token budget, and shared evaluator.

This repository must not import or call `fire-agent-demo` code. It may only consume Runner Bundle inputs and (separately) exported normalized target predictions for side-by-side review.

## Track A — System-Level Outcome (shared)

Compare common outcomes only:

- risk recognition / risk signals
- action recommendation
- unsafe / blocked actions
- missing confirmations
- evidence support & citation integrity
- human review requirement
- final decision gate
- final response safety
- latency / tokens / cost

Do **not** compare SAFE-Router internal module routing fields that baselines do not have.

### Sub-tracks

- **A1** text-only (`direct_llm`)
- **A2** text + same corpus/KG (RAG + E-KELL-style)
- **A3** text + corpus + dynamic snapshots (only methods that support snapshots; faithful E-KELL-style as implemented does not — use a separately named enhanced state-aware method if needed)

See `docs/resource_access_matrix.md`.

## Preferred interop path

1. Main project provides Runner Bundle + Evaluator Bundle (baselines read Runner only).
2. Controlled track: `python scripts/run_interop_baselines.py --experiment-manifest configs/experiments/paper_main_table_v1.yaml --bundle ...`
3. Paper-fidelity track is a **separate** experiment (`ekell_style_paper_fidelity` / ChatGLM-6B); do not merge rows with controlled FireBench results.
4. Main-project neutral evaluator scores paired predictions (bootstrap CI, Wilcoxon/permutation, McNemar, Holm, breakdowns).

See `docs/paper_fidelity_vs_controlled_comparison.md`.

## Legacy side-by-side (optional)

```bash
python scripts/compare_with_target_outputs.py \
  --baseline outputs/baseline_outputs.jsonl \
  --target path/to/safe_outputs_normalized.jsonl \
  --output outputs/side_by_side_comparison.md
```

## Proxy metrics

Local automatic metrics in this repo are **diagnostics only** and must not replace the shared paper evaluator or expert correctness.

## Metric categories

### A. Automatic proxy metrics

- `risk_signal_detection_rate`
- `evidence_support_rate`
- `citation_coverage`
- `unsafe_suggestion_rate`
- `unsupported_recommendation_rate`
- `actionability_score`
- `hallucination_flag`
- `decision_correctness_proxy`

### B. Text-inferred safety metrics

- `blocked_action_recall`
- `missing_confirmation_detection_rate`
- `decision_gate_accuracy`
- `operator_boundary_violation_rate`

These fields may be inferred from text for external baselines. They are useful for coarse comparison but should not be overclaimed.

### C. Manual / expert rubric template

Use `docs/manual_evaluation_rubric.md` to score:

- correctness
- evidence support
- safety compliance
- completeness
- actionability
- conciseness
- comprehensibility / instructiveness

## Limitations

- Automatic metrics are prototypes and proxies.
- Text-inferred safety metrics can misread outputs.
- The original E-KELL expert evaluation is not reproduced unless qualified human evaluators are used.
- The target system's internal control modules must not be called from this repo.
