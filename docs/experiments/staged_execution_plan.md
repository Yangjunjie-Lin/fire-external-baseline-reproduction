# Staged execution plan (deferred)

Baseline engineering and formal configuration are prepared. **No stage below is executed automatically.**
Real cross-repository runs remain locked until the main project publishes its first stable model and formal Runner Bundle.

```text
engineering complete
configuration prepared
execution safely deferred
waiting for main project v1
```

## Stage 0 — Current (comparison code ready; resources pending)

**Goal:** static validation, readiness checks, no paid API, no real model download.

```text
five-method comparison implementation ready
real resources not yet installed
real indexes not yet built
real dry run not yet executed
formal experiment not yet executed
```

**Allowed now:**

```bash
python scripts/check_main_project_readiness.py \
  --resources configs/local/experiment_resources.yaml

python scripts/check_comparison_readiness.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml.example \
  --method-set comparison_suite

python scripts/show_experiment_state.py

python scripts/validate_formal_config.py \
  --validation-stage template \
  --config configs/experiments/controlled_main_table_v1.yaml.example

python scripts/build_comparison_indexes.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml.example \
  --bundle <optional> \
  --method-set comparison_suite \
  --validate-only
```

**Not allowed:** real LLM calls, embedding download, full index build, cross-repo dry run, formal experiment.

---

## Stage 1 — Real dry run (after main project v1)

**Goal:** verify API path, embedding path, schema, token/latency, parser, case completeness. **Not paper results.**

Requires:

- `main_project_v1_ready == true` (structure + approval; manual status alone cannot bypass validation)
- `allow_real_model_calls: true`
- `allow_cross_repo_test: true`
- `--limit` in 1–10
- output under `outputs/dry_run/`
- `freeze_status` may remain `provisional`

Does **not** require `allow_formal_evaluation`, `configs_frozen`, or `real_dry_run_completed`.

**Future commands (do not run until readiness gates open):**

```bash
python scripts/validate_formal_config.py \
  --validation-stage dry_run \
  --config configs/experiments/controlled_main_table_v1.yaml

python scripts/build_comparison_indexes.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <runner_bundle> \
  --method-set comparison_suite

python scripts/run_interop_baselines.py \
  --execution-stage dry_run \
  --method-set comparison_suite \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <runner_bundle> \
  --limit 3 \
  --output outputs/dry_run/comparison_suite_v1/predictions.jsonl \
  --manifest outputs/dry_run/comparison_suite_v1/run_manifest.json
```

Update `configs/local/experiment_resources.yaml`:

- set `main_project.runner_bundle_path` (explicit; discovered candidates are informational only)
- set `execution.allow_real_model_calls: true` (only for controlled dry run)
- set `execution.allow_cross_repo_test: true` only after main-project approval

---

## Stage 2 — DEV tuning

Tune on DEV only:

- BM25 parameters
- Dense top-k
- Hybrid RRF weights
- E-KELL vector top-k
- Neighborhood hop / context budget

Outputs stay **provisional** (`freeze_status: provisional`).

Dense/Hybrid are controlled supplemental baselines in `comparison_suite`. They remain out of `main_table` selection unless `--method-set comparison_suite` is used. Real embedding indexes are built after resources are installed.

---

## Stage 3 — Configuration freeze

Requirements before TEST:

- DEV selection complete
- manifest + method configs updated to `freeze_status: frozen` (human decision)
- config checksums recorded
- prompt hash fixed
- LLM `model` / `model_version` frozen in YAML (`model_source=yaml_config`)
- embedding `model_version` fixed (replace `REQUIRED_BEFORE_REAL_INDEX_BUILD`)

```bash
python scripts/create_freeze_manifest.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --selected-dev-run outputs/tuning/selected_dev_run.json \
  --bundle <runner_bundle> \
  --output configs/freeze/comparison_freeze_manifest_v1.json

# Human: set freeze_status=frozen and freeze_manifest path in experiment manifest.

python scripts/validate_formal_config.py \
  --validation-stage formal \
  --config configs/experiments/controlled_main_table_v1.yaml
```

Formal model identity is frozen in YAML configuration. Environment variables provide credentials and endpoint settings only. `SILICONFLOW_MODEL` does not silently override formal YAML model identity.

---

## Stage 4 — Formal TEST run

One-shot run with frozen configs on the TEST split / formal Runner Bundle.

Requires:

- `allow_formal_evaluation: true`
- `configs_frozen: true`
- `real_dry_run_completed: true`
- **no** `--limit`
- **no** `--allow-partial`
- output under `outputs/interop/` (or formal directory)

```bash
python scripts/run_interop_baselines.py \
  --execution-stage formal \
  --method-set comparison_suite \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <frozen_runner_bundle>
```

Scoring uses **fire-agent-demo shared evaluator** (external to this repo).

---

## Stage 5 — Statistics and reporting

After evaluator outputs:

- main table + supplemental table
- confidence intervals / significance (if applicable)
- cost and latency summaries
- error analysis

---

## Embedding backend note

Dense, Hybrid, and E-KELL controlled share the same embedding factory (`src/external_baselines/retrieval/embedding_backends.py`) with backend id **`text2vec`** (`Text2VecEmbeddingBackend`), which loads models such as **`BAAI/bge-m3`** via `text2vec.SentenceModel` (lazy load; never at import).

Dense uses an evidence-chunk index; E-KELL uses a separate KG/entity index. Hybrid reuses the Dense evidence index. Indexes are not yet built in this repository state.

---

## Readiness relationship

```text
main_project_v1_ready  →  enables Stage 1 dry run (with execution flags)
real_dry_run_passed    →  enables Stage 2 DEV tuning
configs_frozen         →  enables Stage 4 TEST
```

`--override-readiness-lock` exists for manual debugging only. **CI and automation must not use it.** Override is recorded and does not make a run paper-valid.
