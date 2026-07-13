# Final Experiment Commands

> **Do not auto-run paid APIs.** Commands are for the user after license review + credentials.
>
> `.example` files are templates only. Copy to non-`.example` paths before formal runs.

## A. Five-method comparison suite (recommended system contrast)

**Artifact layouts (do not mix):**

| Workflow | Entrypoint | Evaluator handoff |
|----------|------------|-------------------|
| Formal unified suite | `run_decision_comparison_suite.py` + `--formal-run-root` | `outputs/formal/test_public/predictions/` |
| Dry-run diagnostics | `run_decision_comparison_suite.py` + `--execution-stage dry_run` | non-formal; `outputs/interop/dry_run/predictions/` + `outputs/decision_runs/dry_run/` |
| Legacy combined JSONL | `run_interop_baselines.py` | `outputs/interop/comparison_suite_v1/predictions.jsonl` (not the unified Formal layout) |

Formal unified run root:

```text
outputs/formal/test_public/
├── predictions/
├── decisions/
├── suite_summary.json
├── run_manifest.json
└── diagnostics/
```

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

# 5) After DEV selection — complete freeze generation:
#    create_freeze_manifest.py runs freeze-candidate validation first, then loads
#    the Runner Bundle with Formal authority and writes the freeze atomically.
python scripts/create_freeze_manifest.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --selected-dev-run outputs/tuning/selected_dev_run.json \
  --bundle <runner_bundle> \
  --output configs/freeze/comparison_freeze_manifest_v1.json
# Use --draft only while fields are incomplete; draft output is not a complete
# Formal identity. Non-draft requires complete checksums and real identities.

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
  --prediction-dir outputs/formal/test_public/predictions

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

Formal stage forbids `--limit`, `--allow-partial`, `--override-readiness-lock`, and `--enable-dev-aliases`. The decision comparison suite resolves manifest method entries before config merge, uses a two-phase formal compliance state machine (`pre_publish_compliance_passed` → runtime cleanup → staged validation → transactional publish → `formal_result`), **exits nonzero on any formal failure**, validates fail-closed Runner Bundle integrity with **separate producer-declared checksum and consumer-computed hash** plus a manifest-declared in-bundle prediction schema checksum, enforces one shared generation-model identity across all five methods (including finite temperature/top_p/max_tokens/seed/enable_thinking at runtime), requires persisted directory indexes for Dense and E-KELL with explicit `actual_embedding_used=true` and `smoke_fallback_used=false`, unified preflight of all five methods before any LLM initialization, rollback-safe transactional publish of predictions, decisions, and **`suite_summary.json`**, and derives `formal_compliance.formal_result` from runtime evidence rather than stage labels. Dry-run method summaries always report `formal_result=false` but may exit zero.

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
- `comparison_suite_methods` is the sole ordered five-method authority. The `methods` array is an unordered configuration registry and may contain disabled non-comparison entries, but no non-comparison entry may be enabled for a Formal comparison-suite run.
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
- Immutable suite summary records commit success but does not pre-declare backup cleanup success; actual cleanup status is in `publish_receipt.json`.
- Staged validation reparses predictions against the frozen Runner Bundle prediction schema and schema SHA; verifies exact case IDs, method IDs, summaries, supplemental decision artifacts, and all run-manifest hashes.
- The immutable run manifest records both input-cases and prediction-schema provenance, and staged validation checks those values against preflight.
- Formal verifies the actual runtime embedding backend against both method configuration and persisted index metadata.
- Runtime caches are scoped to one comparison-suite invocation and cannot leak across runs.
- Embedding backend injection is invoked only for Dense, Hybrid, and E-KELL.
- Run manifests hash predictions, method summaries, decisions, responses, and unmapped-taxonomy artifacts.
- Manifest artifact paths are validated with both POSIX and Windows path semantics (drive-qualified, root-relative, UNC, device namespace, traversal, symlink escape) and must resolve inside the staged run root.
- Formal execution requires `manifest.files.prediction_schema` to identify a schema file located inside the frozen Runner Bundle. The Bundle manifest must declare the prediction schema SHA-256, and the consumer-computed hash must match both the Bundle declaration and frozen experiment identity.
- The frozen prediction schema is parsed, checksum-validated, and verified as a Draft 2020-12 JSON Schema once before staged record validation; invalid schemas fail closed through `FormalSuiteExecutionError`. Meta-schema and record validation use the same no-network `$ref` policy: under the current single-schema Bundle protocol, only internal fragments, the primary schema `$id`, and the primary schema filename may be referenced.
- Formal `input_cases.jsonl` is strict: every non-empty line must be a JSON object with an exact non-empty string `case_id`; invalid or non-object lines are rejected rather than silently skipped.
- Formal execution never falls back to repository-local schemas. Local schemas are development snapshots only and are not registered as Formal JSON Schema resources.
- The no-network schema registry is input-driven and registers only the primary Bundle schema plus explicitly checksum-verified Bundle resources. Its behavior does not depend on source checkout, current working directory, editable installation, or wheel installation.
- Formal embedding identity validation requires exact JSON boolean flags and positive JSON integer dimensions in persisted index metadata.
- Formal safety-critical numeric parameters require exact finite YAML/JSON numeric types; string-to-number, boolean-to-number, NaN, and Infinity coercion are rejected.
- Formal model, backend, version, environment-variable, prompt, index, and manifest identity fields require exact non-empty YAML strings without implicit string coercion.
- Formal experiment identity fields must be explicitly declared as exact non-empty YAML strings. Explicit `null` values are rejected rather than replaced with defaults.
- Malformed nested prediction/runtime values produce structured schema errors without leaking raw `AttributeError` or `TypeError`.
- Missing optional boolean fields may use documented defaults, but explicitly setting those fields to `null` is invalid.
- Engineering release-readiness gates are structural repository checks; behavioral correctness is established by pytest, staged tamper tests, offline Formal E2E, and externally observed CI results.
- Formal 推荐使用 `--formal-run-root`；提供 formal run root 时无需额外 `--prediction-dir` / `--decision-dir`。
- Legacy 双目录模式仅允许共享同一 run root；`jsonschema` 为核心运行依赖；Hybrid wrapper 不拥有共享 Dense runtime 的关闭权。
- Dense/E-KELL are closed by the suite cache scope before staged validation and transactional publish; Hybrid wrappers close at method end. Runtime cleanup failure does not mask a primary suite error, but with no primary error it prevents commit.
- Release readiness 区分 engineering / empirical；engineering gate 失败退出非零；empirical pending 不阻断工程 CI。
- Formal YAML 安全字段要求精确 boolean 与正整数 dimension，拒绝 string/float/bool 隐式转换。
- Engineering readiness 与 empirical/paper readiness 分离；CI 不声称远程 workflow 已通过。
- Offline Formal E2E injects only LLM transport and embedding-compute boundaries.
- Post-commit warning failures are printed and returned, but never mutate or invalidate the committed run.
