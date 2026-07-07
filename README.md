# fire-external-baseline-reproduction

Independent external baseline reproduction project for fire emergency decision-support comparison.

This repository is intentionally separate from [`fire-agent-demo`](https://github.com/Yangjunjie-Lin/fire-agent-demo). It does **not** contain, import, or call SAFE-Router, Safety Checker, Dynamic REG, HITL Gate, or any internal control module from the target project.

The goal is to reproduce or adapt external baseline pipelines as faithfully and transparently as possible, then run them on the same fire emergency scenario matrix and copied evidence corpus for system-level comparison.

> Research-only warning: this project does not provide real-world emergency advice and must not be used for live emergency operations.

## Implemented first milestone

| Method | Status | Description |
|---|---:|---|
| `direct_llm` | runnable | B0 no-retrieval LLM baseline |
| `vanilla_rag` | runnable | B1 lexical BM25/TF-IDF-style text retrieval over `evidence_chunks.jsonl` |
| `ekell_style` | runnable | B2 E-KELL-style KG + LLM prompt-chain reimplementation |
| `lightrag` | adapter/fallback | Optional external LightRAG adapter; falls back to local graph/text retrieval when package is unavailable |
| `microsoft_graphrag` | adapter/fallback | Optional Microsoft GraphRAG adapter placeholder with clear fallback behavior |

The `ekell_style` method should be cited as:

**E-KELL-style paper-faithful reimplementation**

It is **not** an official E-KELL reproduction unless official authors' code is used.

## External methods mapped

- E-KELL: *Enhancing Emergency Decision-making with Knowledge Graphs and Large Language Models*, arXiv:2311.08732.
- Microsoft GraphRAG: <https://github.com/microsoft/graphrag>
- LightRAG: <https://github.com/HKUDS/LightRAG>
- KG2RAG: <https://github.com/nju-websoft/KG2RAG> *(documented as second-stage optional)*
- PathRAG: <https://github.com/BUPT-GAMMA/PathRAG> *(documented as second-stage optional)*

## Install

```bash
pip install -r requirements.txt
```

For local development:

```bash
pip install -e .
```

LLM use is optional. If no API key/client is configured, the project uses a deterministic local heuristic client so that the baseline runner remains reproducible and testable.

## Prepare data

Copy only data files from the target project. Do not import target project code.

```bash
python scripts/prepare_data.py --source ../fire-agent-demo --target data/
```

Expected copied files:

```text
data/corpus/evidence_chunks.jsonl
data/corpus/entities.jsonl
data/corpus/relations.jsonl
data/corpus/triples.jsonl
data/scenarios/scenario_matrix_v2.json
```

The script searches common target-project locations, especially:

```text
../fire-agent-demo/data/processed/4B_fire_kg_graphrag/
../fire-agent-demo/data/examples/scenario_matrix_v2.json
```

## Run one baseline

```bash
python scripts/run_baseline.py --method ekell_style --dataset data/scenarios/scenario_matrix_v2.json --limit 10
```

## Run all first-milestone baselines

```bash
python scripts/run_all_baselines.py --methods direct_llm,vanilla_rag,ekell_style --dataset data/scenarios/scenario_matrix_v2.json --limit 10
```

## Export report

```bash
python scripts/export_report.py --input outputs/baseline_outputs.jsonl --output outputs/baseline_report.md
```

## Unified output schema

Every baseline writes JSONL records with this schema:

```json
{
  "scenario_id": "...",
  "method": "...",
  "situation_summary": "...",
  "key_risks": [],
  "recommended_actions": [],
  "blocked_or_unsafe_actions": [],
  "missing_confirmations": [],
  "supporting_evidence": [],
  "citations": [],
  "final_decision_gate": "...",
  "retrieved_contexts": [],
  "latency_sec": 0.0,
  "raw_output": {},
  "method_specific": {}
}
```

For external baselines, structured safety fields may be inferred from generated text because the original methods may not natively expose fields such as `blocked_or_unsafe_actions`, `missing_confirmations`, or `final_decision_gate`. Such outputs are marked with:

```json
{
  "method_specific": {
    "structured_safety_fields": "inferred_from_text"
  }
}
```

## Evaluation scope

The evaluation utilities are lightweight and prototype-level. They are designed for relative system comparison, not certified emergency-response validation.

Implemented/estimated metrics include:

- `risk_signal_detection_rate`
- `evidence_support_rate`
- `citation_coverage`
- `unsafe_suggestion_rate`
- `unsupported_recommendation_rate`
- `actionability_score`
- `hallucination_flag`
- `decision_correctness_proxy`
- `blocked_action_recall`
- `missing_confirmation_detection_rate`
- `decision_gate_accuracy`
- `operator_boundary_violation_rate`

Safety-related metrics are marked as inferred from text.

## Project boundaries

This repository may reuse copied input data from `fire-agent-demo`, but it must not reuse implementation modules from that project.

Allowed interaction:

1. Use the same input fire scenarios.
2. Use the same or copied fire corpus/evidence files.
3. Output comparable schema for later comparison.

Not allowed:

1. Import target project modules.
2. Call SAFE-Router, Safety Checker, Dynamic REG, or HITL Gate.
3. Add SAFE-like improvements to external baselines.
4. Claim external baseline methods are improved or certified.

## Repository map

```text
configs/                   Baseline and LLM configuration examples
data/                      Copied input data only
src/external_baselines/     Baseline pipeline implementations
scripts/                   CLI wrappers
docs/                      Reproduction notes and protocol
outputs/                   Generated output files
```

## Comparison objective

After running this project and the target `fire-agent-demo` on the same scenario matrix, compare outputs on:

- safety compliance
- risk recognition
- evidence support
- unsafe action blocking
- missing confirmation detection
- decision boundary control

Research question:

> Does the SAFE Fire Agent system improve over external KG/RAG/GraphRAG/LLM baseline systems in safety compliance, risk recognition, evidence support, unsafe action blocking, missing confirmation detection, and decision boundary control?
