# fire-external-baseline-reproduction

Independent external baseline reproduction project for system-level comparison with [`fire-agent-demo`](https://github.com/Yangjunjie-Lin/fire-agent-demo).

This repository is for **research comparison only**. It does **not** provide real-world emergency advice.

## Current reproduction status

Primary reproduction target:

- **E-KELL: Enhancing Emergency Decision-making with Knowledge Graphs and Large Language Models**
- Paper: https://arxiv.org/abs/2311.08732

Current label:

> **E-KELL-style paper-faithful pipeline-level reimplementation, not official reproduction.**

Supported claim:

> This project implements an independent, pipeline-level, paper-faithful E-KELL-style KG + LLM prompt-chain baseline for fire emergency decision-support comparison. It is not an official E-KELL reproduction, but it maximizes reproduction fidelity using available public paper-level details, copied fire emergency KG/evidence inputs, transparent deviations, and reproducible execution protocols.

Unsupported claim:

> This project fully reproduces the official E-KELL results.

## Fidelity level

Current achieved level: **Level 3 — data-compatible pipeline-level reproduction**, when copied KG/evidence/scenario inputs are present.

See [`docs/reproduction_fidelity_audit.md`](docs/reproduction_fidelity_audit.md).

## Independence boundary

This repository must remain completely independent from `fire-agent-demo`.

Allowed:

1. Copy input fire scenario files.
2. Copy fire corpus / evidence files.
3. Output results in a compatible schema.

Not allowed:

- importing `fire_agent_demo`
- calling SAFE-Router
- calling Safety Checker
- calling Dynamic REG
- calling HITL Gate
- using target-project risk scoring
- using target-project final gate logic
- silently adding target-project policy routing or safety modules

## What is reproduced

- B0 Direct LLM baseline
- B1 Vanilla lexical RAG baseline
- B2 E-KELL-style KG + LLM prompt-chain baseline
- B3 LightRAG / Microsoft GraphRAG adapter stubs with explicit fallback status
- unified output schema
- run manifest
- corpus/data validation
- lightweight proxy metrics
- manual evaluation rubric template
- side-by-side comparison script

E-KELL-style pipeline:

```text
Scenario Input
→ Scenario Understanding / Parsing
→ KG Entity Matching
→ KG Subgraph / Fact Retrieval
→ Evidence Context Construction
→ Prompt Chain Reasoning
→ Final Emergency Decision Support Output
→ Unified Output Normalization
```

## What is not reproduced

- official E-KELL KG
- official E-KELL code
- official E-KELL data preprocessing pipeline
- official exact prompt templates if not public
- official expert evaluation with emergency commanders / firefighters
- official exact results

See [`docs/official_code_data_search.md`](docs/official_code_data_search.md).

## Install

```bash
pip install -e .
pip install -r requirements.txt
```

For tests:

```bash
python -m pytest -q
```

## Data preparation

Copy data from a local `fire-agent-demo` checkout:

```bash
python scripts/prepare_data.py --source ../fire-agent-demo --target data/
```

This copies data only. It does not copy or import code.

Expected input files:

```text
data/corpus/evidence_chunks.jsonl
data/corpus/entities.jsonl
data/corpus/relations.jsonl
data/corpus/triples.jsonl
data/scenarios/scenario_matrix_v2.json
```

A `data/data_manifest.json` file is generated with source paths, sizes, checksums, and copy timestamp.

## Validate data

```bash
python scripts/validate_data.py
python scripts/audit_corpus.py --corpus data/corpus
```

## Heuristic smoke test warning

The default config uses:

```yaml
llm:
  provider: heuristic
```

This is only for smoke tests and reproducibility checks. It is **not** final experimental output.

For final comparison, use a real LLM provider and record:

- provider
- model
- temperature
- max tokens
- run date
- dataset version / checksum

Every output includes:

```json
"method_specific": {
  "llm_config_summary": {
    "provider": "...",
    "model": "...",
    "temperature": 0.0,
    "heuristic_fallback": true
  }
}
```

## Real LLM config examples

OpenAI-compatible:

```bash
cp configs/llm_openai_compatible.yaml.example configs/llm_local.yaml
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # optional for compatible endpoints
python scripts/run_all_baselines.py \
  --config configs/llm_local.yaml \
  --methods direct_llm,vanilla_rag,ekell_style \
  --dataset data/scenarios/scenario_matrix_v2.json \
  --limit 3
```

DeepSeek / Qwen example configs are provided as:

- `configs/llm_deepseek.yaml.example`
- `configs/llm_qwen.yaml.example`

They use OpenAI-compatible client wiring through environment variables.

## Run baselines

Run one baseline:

```bash
python scripts/run_baseline.py \
  --method ekell_style \
  --dataset data/scenarios/scenario_matrix_v2.json \
  --limit 10
```

Run all first-milestone baselines:

```bash
python scripts/run_all_baselines.py \
  --methods direct_llm,vanilla_rag,ekell_style \
  --dataset data/scenarios/scenario_matrix_v2.json \
  --limit 10
```

Expected outputs:

```text
outputs/baseline_outputs.jsonl
outputs/baseline_metrics.csv
outputs/baseline_report.md
outputs/run_manifest.json
```

Export report from existing outputs:

```bash
python scripts/export_report.py \
  --input outputs/baseline_outputs.jsonl \
  --output outputs/baseline_report.md
```

## Evaluation

Metric categories:

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

### C. Manual / expert rubric template

See [`docs/manual_evaluation_rubric.md`](docs/manual_evaluation_rubric.md).

The automatic metrics are prototype proxies. They do not reproduce the original E-KELL expert evaluation unless qualified human evaluators are used.

## Compare with fire-agent-demo outputs

First export normalized SAFE Fire Agent outputs from `fire-agent-demo` externally. Then run:

```bash
python scripts/compare_with_target_outputs.py \
  --baseline outputs/baseline_outputs.jsonl \
  --target path/to/safe_outputs_normalized.jsonl \
  --output outputs/side_by_side_comparison.md
```

This script matches by `scenario_id` and compares:

- `key_risks`
- `recommended_actions`
- `blocked_or_unsafe_actions`
- `missing_confirmations`
- `supporting_evidence`
- `final_decision_gate`

It does not import target-project code.

## GraphRAG / LightRAG transparency

The `lightrag` and `microsoft_graphrag` methods are currently transparent adapters. Unless their actual external packages, indexing, and query pipelines are configured, they fall back to local graph/text retrieval and mark:

- `actual_external_package_used: false`
- `fallback_retrieval_used: true`
- `indexing_performed: false`
- external repository URL
- deviation from official system

Do not claim complete LightRAG or Microsoft GraphRAG reproduction unless actual indexing and query integration is implemented.

## Development checks

```bash
python scripts/validate_data.py
python scripts/run_all_baselines.py --methods direct_llm,vanilla_rag,ekell_style --dataset data/scenarios/scenario_matrix_v2.json --limit 3
python scripts/export_report.py --input outputs/baseline_outputs.jsonl --output outputs/baseline_report.md
python -m compileall src scripts tests
python -m pytest -q
```

## Limitations

- Not official E-KELL reproduction.
- Not certified emergency-response advice.
- Default heuristic mode is smoke-test only.
- Original E-KELL expert evaluation is not reproduced.
- Copied input data may not match official E-KELL data.
- GraphRAG / LightRAG adapters are not complete official reproductions unless explicitly configured.
