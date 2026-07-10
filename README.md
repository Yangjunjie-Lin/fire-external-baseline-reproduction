# fire-external-baseline-reproduction

Independent external baseline package for system-level comparison with [`fire-agent-demo`](https://github.com/Yangjunjie-Lin/fire-agent-demo).

Research comparison only — **not** real-world emergency advice.

## 1. Project purpose

Provide **external** baselines (Direct LLM, BM25-RAG, E-KELL-style, optional GraphRAG adapters) that:

- consume a main-project **Runner Bundle** (`input_cases.jsonl`)
- emit **firebench-interop-v1** canonical predictions
- never import `fire_agent_demo` or call SAFE-Router / Safety Checker / Dynamic REG / HITL

## 2. Current status

```text
five-method comparison implementation ready
real resources not yet installed
real indexes not yet built
real dry run not yet executed
formal experiment not yet executed
```

Controlled comparison code is complete for:

- `main_table` (3 methods)
- `comparison_suite` (5 methods: Direct / BM25 / Dense / Hybrid / E-KELL controlled)

Dense and Hybrid are **controlled supplemental** baselines. They do **not** enter E-KELL paper-fidelity and do **not** change E-KELL controlled paper structure.

Formal freeze happens only after DEV selection. Do not claim experiment complete / paper ready / empirically validated.

Readiness gates:

| Flag | Value |
|---|---|
| configuration_prepared | true |
| comparison_suite_code_ready | true |
| api_environment_available | present_or_unknown |
| real_model_calls_executed | false |
| embedding_index_built | false |
| main_project_v1_ready | false |
| cross_repository_real_dry_run | false |
| formal_experiment_started | false |

**Model authority:** Formal model identity is frozen in YAML configuration. Environment variables provide credentials and endpoint settings only. `SILICONFLOW_MODEL` does not silently override formal YAML model identity.

Stage plan: [`docs/experiments/staged_execution_plan.md`](docs/experiments/staged_execution_plan.md)

Details: [`docs/status/current_project_status.md`](docs/status/current_project_status.md)

## 3. Method table

| method_id | Layer | Implementation | Empirical |
|---|---|---|---|
| `direct_llm` | formal main table | implemented | heuristic smoke only |
| `bm25_rag` | formal main table | implemented (package: `vanilla_rag/`) | deterministic sparse OK; shared-LLM pending |
| `ekell_style_controlled_shared_llm` | formal main table | Level 3 pipeline-level reimplementation | shared-LLM pending |
| `dense_rag` | comparison_suite supplemental | real text2vec index build/load/query | real index not built |
| `hybrid_rag` | comparison_suite supplemental | BM25 + Dense + RRF (reuses Dense index) | real index not built |
| `ekell_style_paper_fidelity` | paper-fidelity track | interface ready | ChatGLM-6B pending |
| `ekell_style_enhanced` | supplemental ablation | implemented | not in comparison_suite |
| `lightrag` / `microsoft_graphrag` / `fallback_graph_retrieval` | fallback_only | adapters + local fallback | actual indexing pending |
| `ekell_style_legacy_bm25` | legacy diagnostic | BM25+3-stage scaffold | not main table |

**Method sets:**

```bash
--method-set main_table          # default: direct_llm, bm25_rag, ekell_style_controlled_shared_llm
--method-set comparison_suite    # five-method fair comparison (recommended for system contrast)
```

`--include-supplemental` is deprecated; prefer `--method-set comparison_suite`.

**Single registry:** `src/external_baselines/method_registry.py`  
Aliases (e.g. `vanilla_rag` → `bm25_rag`, `ekell_style_faithful` → controlled) are derived from that registry only.

## 4. Formal interop workflow

**Only formal entrypoint.** `.example` files are templates only — copy before running.

```bash
pip install -e ".[llm,embeddings]"
# or: pip install -r requirements.txt && pip install -r requirements-optional-embeddings.txt

# Copy templates before formal runs (local files are gitignored):
cp configs/experiments/controlled_main_table_v1.yaml.example configs/experiments/controlled_main_table_v1.yaml
cp configs/models/shared_real_model.yaml.example configs/models/shared_real_model.yaml

# Preparation checks (no API calls):
python scripts/check_main_project_readiness.py --resources configs/local/experiment_resources.yaml
python scripts/check_comparison_readiness.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --resources configs/local/experiment_resources.yaml \
  --method-set comparison_suite
python scripts/show_experiment_state.py

python scripts/validate_formal_config.py \
  --validation-stage dry_run \
  --method-set comparison_suite \
  --config configs/experiments/controlled_main_table_v1.yaml

python scripts/run_interop_baselines.py \
  --execution-stage dry_run \
  --method-set comparison_suite \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle path/to/runner_bundle \
  --limit 3 \
  --output outputs/dry_run/comparison_suite_v1/predictions.jsonl \
  --manifest outputs/dry_run/comparison_suite_v1/run_manifest.json
```

After DEV freeze:

```bash
python scripts/create_freeze_manifest.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --selected-dev-run outputs/tuning/selected_dev_run.json \
  --bundle path/to/runner_bundle \
  --output configs/freeze/comparison_freeze_manifest_v1.json

python scripts/validate_formal_config.py \
  --validation-stage formal \
  --config configs/experiments/controlled_main_table_v1.yaml

python scripts/run_interop_baselines.py \
  --execution-stage formal \
  --method-set comparison_suite \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle path/to/frozen_runner_bundle \
  --output outputs/interop/comparison_suite_v1/predictions.jsonl \
  --manifest outputs/interop/comparison_suite_v1/run_manifest.json
```

Paper-fidelity track (separate experiment):

```bash
cp configs/experiments/ekell_paper_fidelity_v1.yaml.example configs/experiments/ekell_paper_fidelity_v1.yaml
cp configs/models/chatglm6b_local.yaml.example configs/models/chatglm6b_local.yaml
cp configs/ekell_paper_fidelity_chatglm6b.yaml.example configs/ekell_paper_fidelity_chatglm6b.yaml

python scripts/validate_formal_config.py \
  --config configs/experiments/ekell_paper_fidelity_v1.yaml
```

Template structure check (not a formal run):

```bash
python scripts/validate_formal_config.py \
  --config configs/experiments/controlled_main_table_v1.yaml.example \
  --allow-placeholders
```

| Contract | Value |
|---|---|
| Formal input | Runner Bundle → `manifest.files.input_cases` → `input_cases.jsonl` |
| Formal output | firebench-interop-v1 JSONL |
| Schema authority | Bundle `prediction_schema.json` (+ checksum) |
| Scoring authority | `fire-agent-demo` shared evaluator |

Heuristic smoke (no paid API):

```bash
python scripts/smoke_interop.py
# or: python scripts/smoke_main_runner_bundle.py
```

## 5. Repository structure

```text
src/external_baselines/   # methods, interop, evaluation, method_registry
scripts/                  # formal: run_interop_baselines.py; see scripts/legacy/
configs/                  # experiments/, frozen/, models/, prompts/, smoke
schemas/                  # local schema copies (dev/tests only)
docs/status|methods|fidelity|...
data/                     # local copies only (not formal primary input)
outputs/                  # runtime artifacts (gitignored)
```

## 6. Development checks

```bash
python -m compileall src scripts tests
python -m pytest -q
```

Local data copy (legacy/dev; not formal primary path):

```bash
python scripts/prepare_data.py --source ../fire-agent-demo --target data/
python scripts/validate_data.py
```

## 7. Limitations

- Not official E-KELL reproduction; not certified emergency advice.
- Default heuristic LLM is smoke-only; `paper_final: true` rejects it.
- LightRAG / Microsoft GraphRAG remain `fallback_only` until actual index+query.
- Local `evaluate_predictions.py` is **proxy diagnostics only** — not the paper evaluator.
- Formal experiments (shared LLM, ChatGLM-6B, expert eval, statistics) are **pending**.

## 8. Documentation index

| Topic | Doc |
|---|---|
| Status | [`docs/status/current_project_status.md`](docs/status/current_project_status.md) |
| Registry | [`docs/methods/method_registry.md`](docs/methods/method_registry.md) |
| Fidelity | [`docs/fidelity/method_fidelity_matrix.md`](docs/fidelity/method_fidelity_matrix.md) |
| Interop | [`docs/firebench_interop_v1_integration.md`](docs/firebench_interop_v1_integration.md) |
| Tracks | [`docs/paper_fidelity_vs_controlled_comparison.md`](docs/paper_fidelity_vs_controlled_comparison.md) |
| No overclaim | [`docs/no_overclaim_policy.md`](docs/no_overclaim_policy.md) |
| Legacy scripts | [`scripts/legacy/README.md`](scripts/legacy/README.md) |
| Doc archive note | [`docs/archive/README.md`](docs/archive/README.md) |

### Development and legacy commands

See [`scripts/legacy/README.md`](scripts/legacy/README.md). Examples (not paper-final):

```bash
python scripts/generate_predictions.py --methods direct_llm,bm25_rag,ekell_style_controlled_shared_llm --config configs/deterministic_heuristic_smoke.yaml
python scripts/evaluate_predictions.py --predictions outputs/predictions.jsonl   # LOCAL PROXY — NOT SHARED PAPER EVALUATOR
python scripts/run_baseline.py --method bm25_rag --dataset data/scenarios/scenario_matrix_v2.json --limit 10
```
