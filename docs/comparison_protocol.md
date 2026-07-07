# Comparison Protocol

## Goal

Compare outputs from independent external baselines against exported SAFE Fire Agent outputs from `fire-agent-demo` using the same scenario matrix and a compatible schema.

This repository must not import or call `fire-agent-demo` code. It may only consume exported, normalized target outputs.

## Required inputs

1. Baseline outputs from this repo: `outputs/baseline_outputs.jsonl`
2. Target SAFE Fire Agent outputs exported elsewhere: `safe_outputs_normalized.jsonl`
3. Shared scenario IDs from `scenario_matrix_v2.json`

## Command

```bash
python scripts/compare_with_target_outputs.py \
  --baseline outputs/baseline_outputs.jsonl \
  --target path/to/safe_outputs_normalized.jsonl \
  --output outputs/side_by_side_comparison.md
```

## Compared fields

- `key_risks`
- `recommended_actions`
- `blocked_or_unsafe_actions`
- `missing_confirmations`
- `supporting_evidence`
- `final_decision_gate`

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
