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

Minimum formal method set (strong, independent baselines):

| ID | method_id | Class |
|---|---|---|
| B0 | `direct_llm` | baseline |
| B1 | `bm25_rag` (`vanilla_rag` alias) | baseline |
| B2 | `dense_rag` | enhanced / smoke until real embeddings |
| B3 | `hybrid_rag` | enhanced / smoke until real dense |
| B4 | `ekell_style_faithful` | faithful (not official E-KELL) |
| B5 | `ekell_style_enhanced` | enhanced (separate paper row) |
| B6 | `lightrag` | actual only if indexing+query; else fallback_only |
| B7 | `microsoft_graphrag` | actual only if indexing+query; else fallback_only |

Also: explicit `fallback_graph_retrieval` (never enters actual GraphRAG leaderboard).

Additional package features:

- `firebench-interop-v1` Runner Bundle integration
- gold-isolated prediction generation
- frozen DEV-selected configs under `configs/frozen/`
- unified legacy schema + canonical interop predictions
- run manifest / checksums
- proxy diagnostics (not a substitute for the shared paper evaluator)
- manual evaluation rubric templates

E-KELL-style pipeline:

```text
Scenario Input
→ Situation Understanding / Parsing
→ Entity Matching
→ KG Subgraph Retrieval
→ Evidence Context Construction
→ Prompt Chain Reasoning
→ Final Response
→ Output Normalization
```

Label:

> **E-KELL-style paper-faithful pipeline-level reimplementation, not official E-KELL reproduction.**

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

## firebench-interop-v1

```bash
python scripts/run_interop_baselines.py \
  --bundle path/to/runner_bundle \
  --methods direct_llm,bm25_rag,dense_rag,hybrid_rag,ekell_style_faithful \
  --config configs/shared_real_model.yaml \
  --output outputs/firebench_interop_v1_predictions.jsonl
```

See [`docs/firebench_interop_v1_integration.md`](docs/firebench_interop_v1_integration.md).

Gold-isolated split workflow:

```bash
python scripts/generate_predictions.py --methods direct_llm,bm25_rag,ekell_style_faithful --config configs/deterministic_heuristic_smoke.yaml
python scripts/evaluate_predictions.py --predictions outputs/predictions.jsonl   # proxy only
python scripts/build_report.py --predictions outputs/predictions.jsonl
```

## Heuristic smoke test warning

The default config uses:

```yaml
llm:
  provider: heuristic
```

This is only for smoke tests and reproducibility checks. It is **not** final experimental output.
`paper_final: true` rejects heuristic providers.

For final comparison, use a real shared LLM config (`configs/shared_real_model.yaml.example`) and record provider, model, model_version, temperature, top_p, max tokens, seed, dataset/bundle checksums.

## Real LLM config examples

```bash
cp configs/shared_real_model.yaml.example configs/shared_real_model.yaml
# set OPENAI_API_KEY / OPENAI_BASE_URL
# DO NOT auto-run paid APIs from CI/agents
```

Also: `configs/paper_main_run.yaml.example`, `configs/paper_robustness_run.yaml.example`, `configs/frozen/*.yaml`.

## Run baselines

```bash
python scripts/run_baseline.py \
  --method ekell_style_faithful \
  --dataset data/scenarios/scenario_matrix_v2.json \
  --limit 10

python scripts/run_all_baselines.py \
  --methods direct_llm,bm25_rag,ekell_style_faithful \
  --dataset data/scenarios/scenario_matrix_v2.json \
  --limit 10
```

Expected outputs:

```text
outputs/baseline_outputs.jsonl
outputs/firebench_interop_v1_predictions.jsonl
outputs/baseline_metrics.csv
outputs/baseline_report.md
outputs/run_manifest.json
```

## Key docs

- [`docs/firebench_interop_v1_integration.md`](docs/firebench_interop_v1_integration.md)
- [`docs/baseline_tuning_protocol.md`](docs/baseline_tuning_protocol.md)
- [`docs/baseline_method_cards.md`](docs/baseline_method_cards.md)
- [`docs/resource_access_matrix.md`](docs/resource_access_matrix.md)
- [`docs/final_experiment_commands.md`](docs/final_experiment_commands.md)
- [`docs/data_license_audit.md`](docs/data_license_audit.md)
- [`docs/no_overclaim_policy.md`](docs/no_overclaim_policy.md)
- [`docs/comparison_protocol.md`](docs/comparison_protocol.md)

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


## Top-tier paper readiness

This repository is structurally prepared as a top-tier-paper external baseline package scaffold, but it is **not** a completed top-tier experimental result package. Final paper-level validity requires:

- real LLM runs with recorded provider/model/version/temperature/date
- actual LightRAG / Microsoft GraphRAG indexing and query integration if those baselines are claimed
- a larger and frozen scenario matrix
- exported SAFE Fire Agent normalized outputs from `fire-agent-demo`
- blind expert/manual evaluation
- inter-annotator agreement reporting
- statistical analysis over paired scenario-level results
- final artifact packaging with checksums and filled data/model/run cards

Use these documents before final experiments and paper submission:

- `docs/top_tier_readiness_audit.md`
- `docs/paper_experiment_protocol.md`
- `docs/data_card_template.md`
- `docs/scenario_matrix_card_template.md`
- `docs/model_run_card_template.md`
- `docs/statistical_analysis_plan.md`
- `docs/paper_appendix_artifact_checklist.md`
- `docs/no_overclaim_policy.md`

Diagnostic commands:

```bash
python scripts/doctor.py
python scripts/validate_outputs.py --input outputs/baseline_outputs.jsonl
python scripts/analyze_manual_scores.py --input evaluation_forms/manual_evaluation_results.csv
```

## Limitations

- Not official E-KELL reproduction.
- Not certified emergency-response advice.
- Default heuristic mode is smoke-test only.
- Original E-KELL expert evaluation is not reproduced.
- Copied input data may not match official E-KELL data.
- GraphRAG / LightRAG adapters are not complete official reproductions unless explicitly configured.
