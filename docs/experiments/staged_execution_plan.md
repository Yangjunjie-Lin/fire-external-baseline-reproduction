# Staged execution plan (deferred)

Baseline engineering and formal configuration are prepared. **No stage below is executed automatically.**
Real cross-repository runs remain locked until the main project publishes its first stable model and formal Runner Bundle.

```text
engineering complete
configuration prepared
execution safely deferred
waiting for main project v1
```

## Stage 0 — Current (configuration prepared)

**Goal:** static validation, readiness checks, no paid API, no index build.

**Allowed now:**

```bash
python scripts/check_main_project_readiness.py \
  --resources configs/local/experiment_resources.yaml

python scripts/show_experiment_state.py

python scripts/validate_formal_config.py \
  --config configs/experiments/controlled_main_table_v1.yaml.example \
  --allow-placeholders
```

**Not allowed:** real LLM calls, embedding download, index build, cross-repo dry run, formal experiment.

---

## Stage 1 — Real dry run (after main project v1)

**Goal:** verify API path, embedding path, schema, token/latency, parser, case completeness. **Not paper results.**

Requires:

- `main_project_v1_ready == true` (structure + approval; manual status alone cannot bypass validation)
- `allow_real_model_calls: true`
- `allow_cross_repo_test: true`
- `--limit` in 1–10
- output under `outputs/dry_run/`

Does **not** require `allow_formal_evaluation`, `configs_frozen`, or `real_dry_run_completed`.

**Future commands (do not run until readiness gates open):**

```bash
python scripts/validate_formal_config.py \
  --config configs/experiments/controlled_main_table_v1.yaml

python scripts/run_interop_baselines.py \
  --execution-stage dry_run \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <runner_bundle> \
  --limit 3 \
  --output outputs/dry_run/controlled_v1/predictions.jsonl \
  --manifest outputs/dry_run/controlled_v1/run_manifest.json
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

Dense/Hybrid configs are prepared but real embedding index/query wiring is pending. They remain disabled supplemental methods.

---

## Stage 3 — Configuration freeze

Requirements before TEST:

- DEV selection complete
- manifest + method configs updated to `freeze_status: frozen` (human decision)
- config checksums recorded
- prompt hash fixed
- LLM `model` / `model_version` frozen in YAML (`model_source=yaml_config`)
- embedding `model_version` fixed (replace `REQUIRED_BEFORE_REAL_INDEX_BUILD`)

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

E-KELL vector retrieval uses backend id **`text2vec`** (`Text2VecEmbeddingBackend`), which loads models such as **`BAAI/bge-m3`** via `text2vec.SentenceModel`.

Dense/Hybrid RAG frozen configs reference the same candidate model for fairness. The dense index builder still requires a non-smoke adapter before real index construction — config is prepared; implementation wiring is deferred.

---

## Readiness relationship

```text
main_project_v1_ready  →  enables Stage 1 dry run (with execution flags)
real_dry_run_passed    →  enables Stage 2 DEV tuning
configs_frozen         →  enables Stage 4 TEST
```

`--override-readiness-lock` exists for manual debugging only. **CI and automation must not use it.** Override is recorded and does not make a run paper-valid.
