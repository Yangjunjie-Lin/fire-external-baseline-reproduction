# Final Experiment Commands

> **Do not auto-run paid APIs.** Commands are for the user after license review + credentials.
>
> `.example` files are templates only. Copy to non-`.example` paths before formal runs.

## A. Five-method comparison suite (recommended system contrast)

```bash
cp .env.example .env
# edit .env → set SILICONFLOW_API_KEY=...

cp configs/models/shared_real_model.yaml.example configs/models/shared_real_model.yaml
cp configs/experiments/controlled_main_table_v1.yaml.example configs/experiments/controlled_main_table_v1.yaml
# edit controlled_main_table_v1.yaml → set bundle + embedding model_version

# 1) Resource check
python scripts/check_comparison_readiness.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --resources configs/local/experiment_resources.yaml \
  --method-set comparison_suite

# 2) Index build (after embedding model is available)
python scripts/build_comparison_indexes.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <runner_bundle> \
  --method-set comparison_suite

# validate-only (no model load):
python scripts/build_comparison_indexes.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <runner_bundle> \
  --method-set comparison_suite \
  --validate-only

# 3) Dry-run validation (provisional freeze OK)
python scripts/validate_formal_config.py \
  --validation-stage dry_run \
  --config configs/experiments/controlled_main_table_v1.yaml

# 4) Five-method dry run
python scripts/run_interop_baselines.py \
  --execution-stage dry_run \
  --method-set comparison_suite \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <runner_bundle> \
  --limit 3 \
  --output outputs/dry_run/comparison_suite_v1/predictions.jsonl \
  --manifest outputs/dry_run/comparison_suite_v1/run_manifest.json

# 5) After DEV selection — create freeze manifest (human confirms freeze_status)
python scripts/create_freeze_manifest.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --selected-dev-run outputs/tuning/selected_dev_run.json \
  --bundle <runner_bundle> \
  --output configs/freeze/comparison_freeze_manifest_v1.json

# 6) Formal validation (requires freeze_status=frozen + freeze_manifest)
python scripts/validate_formal_config.py \
  --validation-stage formal \
  --config configs/experiments/controlled_main_table_v1.yaml

# 7) Formal comparison
python scripts/run_interop_baselines.py \
  --execution-stage formal \
  --method-set comparison_suite \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <frozen_runner_bundle> \
  --output outputs/interop/comparison_suite_v1/predictions.jsonl \
  --manifest outputs/interop/comparison_suite_v1/run_manifest.json
```

Formal stage forbids `--limit`, `--allow-partial`, and `--override-readiness-lock`.

## B. Main table only (3 methods)

```bash
python scripts/run_interop_baselines.py \
  --execution-stage dry_run \
  --method-set main_table \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <runner_bundle> \
  --limit 3 \
  --output outputs/dry_run/main_table_v1/predictions.jsonl \
  --manifest outputs/dry_run/main_table_v1/run_manifest.json
```

## C. Paper fidelity (ChatGLM-6B; separate experiment)

```bash
cp configs/experiments/ekell_paper_fidelity_v1.yaml.example configs/experiments/ekell_paper_fidelity_v1.yaml
cp configs/models/chatglm6b_local.yaml.example configs/models/chatglm6b_local.yaml
# Configure local ChatGLM-6B + text2vec; do not merge with controlled FireBench rows.

python scripts/validate_formal_config.py \
  --validation-stage formal \
  --config configs/experiments/ekell_paper_fidelity_v1.yaml
```

## Template validation (structure only)

```bash
python scripts/validate_formal_config.py \
  --validation-stage template \
  --config configs/experiments/controlled_main_table_v1.yaml.example
```

`--allow-placeholders` still works but is deprecated (maps to `template`).

## Notes

- Dense/Hybrid are controlled supplemental baselines; not E-KELL paper-fidelity.
- `--include-supplemental` is deprecated; use `--method-set comparison_suite`.
- Formal model identity is frozen in YAML; env vars supply credentials only.
