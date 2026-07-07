# Comparison protocol

## Objective

Compare independent external baseline outputs against `fire-agent-demo` SAFE Fire Agent outputs using the same scenario matrix and compatible schema.

## Inputs

- `data/scenarios/scenario_matrix_v2.json`
- `data/corpus/evidence_chunks.jsonl`
- `data/corpus/entities.jsonl`
- `data/corpus/relations.jsonl`
- `data/corpus/triples.jsonl`

## Baseline execution

```bash
python scripts/run_all_baselines.py --methods direct_llm,vanilla_rag,ekell_style --dataset data/scenarios/scenario_matrix_v2.json --limit 10
```

Outputs:

- `outputs/baseline_outputs.jsonl`
- `outputs/baseline_metrics.csv`
- `outputs/baseline_report.md`

## Target execution

Run `fire-agent-demo` independently on the same `scenario_matrix_v2.json`. Export or convert its outputs to the unified schema.

## Metrics to compare

Core:

- risk signal detection
- evidence support
- citation coverage
- unsafe suggestion rate
- unsupported recommendation rate
- actionability score
- decision correctness proxy

Safety-boundary-oriented:

- blocked action recall
- missing confirmation detection rate
- decision gate accuracy
- operator boundary violation rate

## Interpretation

The external baselines are not expected to include SAFE-specific control modules. Improvements by `fire-agent-demo` should be discussed as system-level contributions, especially around unsafe action blocking, missing confirmation detection, and final decision-boundary control.
