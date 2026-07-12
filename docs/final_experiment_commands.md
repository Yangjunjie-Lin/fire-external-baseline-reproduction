# Final Experiment Commands

> **Do not auto-run paid APIs.** Commands are for the user after license review + credentials.
>
> `.example` files are templates only. Copy to non-`.example` paths before formal runs.

## A. Five-method comparison suite (recommended system contrast)

```bash
cp .env.example .env
# edit .env → set SILICONFLOW_API_KEY=...

pip install -e ".[llm,embeddings]"
# First text2vec encode may download BAAI/bge-m3; pre-cache or mount the model before formal runs.

cp configs/models/shared_real_model.yaml.example configs/models/shared_real_model.yaml
cp configs/experiments/controlled_main_table_v1.yaml.example configs/experiments/controlled_main_table_v1.yaml
# edit controlled_main_table_v1.yaml → set bundle + embedding model_version

# 1) Resource check
python scripts/check_comparison_readiness.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --resources configs/local/experiment_resources.yaml \
  --bundle <runner_bundle> \
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
  --method-set comparison_suite \
  --config configs/experiments/controlled_main_table_v1.yaml

# Pre-formal contract checks (read-only main-project reference)
python scripts/check_firebench_contract_snapshot.py --main-repo ../fire-agent-demo
python scripts/check_firebench_taxonomy_snapshot.py --main-repo ../fire-agent-demo
python scripts/check_comparison_readiness.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <frozen_runner_bundle> \
  --method-set comparison_suite

# 4) Five-method dry run (unified decision I/O — preferred for evaluator handoff)
python scripts/run_decision_comparison_suite.py \
  --runner-bundle <runner_bundle> \
  --method-set comparison_suite \
  --execution-stage dry_run \
  --limit 3 \
  --prediction-dir outputs/interop/dry_run/predictions \
  --decision-dir outputs/decision_runs/dry_run \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml

# Optional: legacy combined JSONL runner
python scripts/run_interop_baselines.py \
  --execution-stage dry_run \
  --method-set comparison_suite \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <runner_bundle> \
  --limit 3 \
  --output outputs/dry_run/comparison_suite_v1/predictions.jsonl \
  --manifest outputs/dry_run/comparison_suite_v1/run_manifest.json

# 5) After DEV selection — freeze order (do not reverse):
#    a) set freeze_status=frozen and freeze_manifest path in the experiment manifest
#    b) save the manifest (checksum includes freeze_manifest path)
#    c) create freeze manifest (hashes the saved manifest)
python scripts/create_freeze_manifest.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --selected-dev-run outputs/tuning/selected_dev_run.json \
  --bundle <runner_bundle> \
  --output configs/freeze/comparison_freeze_manifest_v1.json
# Use --draft only while fields are incomplete; non-draft requires complete checksums.

# 6) Formal validation (requires freeze_status=frozen + freeze_manifest)
python scripts/validate_formal_config.py \
  --validation-stage formal \
  --method-set comparison_suite \
  --config configs/experiments/controlled_main_table_v1.yaml

# 7) Formal comparison (per-method prediction JSONL for main-project evaluator)
python scripts/run_decision_comparison_suite.py \
  --runner-bundle <frozen_runner_bundle> \
  --method-set comparison_suite \
  --execution-stage formal \
  --formal-run-root outputs/formal/test_public \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml

# Legacy layout (prediction_dir and decision_dir must share one formal run root):
#   <formal_run_root>/predictions
#   <formal_run_root>/decisions
# python scripts/run_decision_comparison_suite.py \
#   --prediction-dir outputs/formal/test_public/predictions \
#   --decision-dir outputs/formal/test_public/decisions \
#   ...

# 8) Taxonomy contract check before handing predictions to the main-project evaluator
python scripts/check_output_taxonomy.py \
  --prediction-dir outputs/interop/test_public/predictions

python scripts/check_firebench_contract_snapshot.py \
  --main-repo ../fire-agent-demo

python scripts/check_firebench_taxonomy_snapshot.py \
  --main-repo ../fire-agent-demo

# Optional: legacy combined JSONL
python scripts/run_interop_baselines.py \
  --execution-stage formal \
  --method-set comparison_suite \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <frozen_runner_bundle> \
  --output outputs/interop/comparison_suite_v1/predictions.jsonl \
  --manifest outputs/interop/comparison_suite_v1/run_manifest.json
```

Formal stage forbids `--limit`, `--allow-partial`, `--override-readiness-lock`, and `--enable-dev-aliases`. The decision comparison suite resolves manifest method entries before config merge, uses a two-phase formal compliance state machine (`pre_publish_compliance_passed` → transactional publish → `formal_result`), **exits nonzero on any formal failure**, validates fail-closed Runner Bundle integrity with **separate producer-declared checksum and consumer-computed hash**, enforces one shared generation-model identity across all five methods (including temperature/top_p/max_tokens/seed/enable_thinking at runtime), requires persisted directory indexes for Dense and E-KELL with explicit `actual_embedding_used=true` and `smoke_fallback_used=false`, unified preflight of all five methods before any LLM initialization, rollback-safe transactional publish of predictions, decisions, and **`suite_summary.json`**, and derives `formal_compliance.formal_result` from runtime evidence rather than stage labels. Dry-run method summaries always report `formal_result=false` but may exit zero.

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
- Formal control/diagnostics directory is outside the immutable run root:

```text
outputs/formal/
├── test_public/
│   ├── predictions/
│   ├── decisions/
│   ├── suite_summary.json
│   ├── run_manifest.json
│   └── diagnostics/
│       └── decision_suite_preflight.json
└── .test_public.control/
    ├── runs/<run_id>/
    │   ├── preflight.json
    │   ├── failure_summary.json
    │   └── publish_receipt.json
    ├── FORMAL_RUN_FAILED.json
    └── FORMAL_PUBLISH_CLEANUP_WARNING.json
```

- Final suite summary and run manifest are staged before commit; the committed run root is never rewritten after rename.
- Post-commit cleanup failures are warnings only (written to control root).
- All Formal CLI validation failures emit structured JSON on stderr and exit 1.
