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
Engineering-complete external baseline scaffold
+ code-level E-KELL-style reproduction
+ firebench-interop-v1 interop
≠ formal experiments completed
≠ official E-KELL reproduction
≠ empirically validated paper results
```

Details: [`docs/status/current_project_status.md`](docs/status/current_project_status.md)

## 3. Method table

| method_id | Layer | Implementation | Empirical |
|---|---|---|---|
| `direct_llm` | formal main table | implemented | heuristic smoke only |
| `bm25_rag` | formal main table | implemented (package: `vanilla_rag/`) | deterministic sparse OK; shared-LLM pending |
| `ekell_style_controlled_shared_llm` | formal main table | Level 3 pipeline-level reimplementation | shared-LLM pending |
| `ekell_style_paper_fidelity` | paper-fidelity track | interface ready | ChatGLM-6B pending |
| `dense_rag` / `hybrid_rag` / `ekell_style_enhanced` | supplemental | implemented | formal only with real dense / as extension |
| `lightrag` / `microsoft_graphrag` / `fallback_graph_retrieval` | fallback_only | adapters + local fallback | actual indexing pending |
| `ekell_style_legacy_bm25` | legacy diagnostic | BM25+3-stage scaffold | not main table |

**Single registry:** `src/external_baselines/method_registry.py`  
Aliases (e.g. `vanilla_rag` → `bm25_rag`, `ekell_style_faithful` → controlled) are derived from that registry only.

## 4. Formal interop workflow

**Only formal entrypoint.** `.example` files are templates only — copy before running.

```bash
pip install -e .
pip install -r requirements.txt

# Controlled main table
cp configs/experiments/controlled_main_table_v1.yaml.example configs/experiments/controlled_main_table_v1.yaml
cp configs/models/shared_real_model.yaml.example configs/models/shared_real_model.yaml
# Fill all placeholders (model, ekell_vector, bundle path, etc.)

python scripts/validate_formal_config.py \
  --config configs/experiments/controlled_main_table_v1.yaml

python scripts/run_interop_baselines.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle path/to/runner_bundle \
  --output outputs/interop/predictions.jsonl \
  --manifest outputs/interop/run_manifest.json
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
