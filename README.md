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
unified decision I/O ready for five-method comparison
FireBench taxonomy contract ready
real resources not yet installed
real indexes not yet built
real dry run not yet executed
formal experiment not yet executed
```

Structured decision IDs use a frozen FireBench taxonomy snapshot. Natural-language text remains free-form in `action.text` / `final_response.text`. Character-level normalization and exact aliases only — no semantic inference.

External baselines preserve native retrieval/reasoning and emit:

1. structured decision JSON
2. natural-language response
3. firebench-interop-v1 prediction JSONL (per method)

This repository **only generates predictions**. Formal scoring remains owned by `fire-agent-demo`.

Controlled comparison code is complete for:

- `main_table` (3 methods)
- `comparison_suite` (5 methods: Direct / BM25 / Dense / Hybrid / E-KELL controlled)

Dense and Hybrid are **controlled supplemental** baselines. They do **not** enter E-KELL paper-fidelity and do **not** change E-KELL controlled paper structure.

Formal freeze happens only after DEV selection. Do not claim experiment complete / paper ready / empirically validated.

Official offline validation is fail-closed. `build_comparison_indexes.py
--validate-only` returns exit code `0` only when the Runner Bundle and every
required Dense, Hybrid, and E-KELL identity validate; any recorded error writes
`ok=false` and returns exit code `1`. The builder shares Formal validation's
exact string/integer/boolean contract and never repairs invalid configuration
through truthiness or type coercion. Complete freezes bind the actual merged
E-KELL `prompt_dir`, the SHA-256 of every required prompt, and the complete
prompt-tree SHA-256. Official build, freeze, and Formal preflight share the same
strict FireKG loader: all four JSONL files are required and non-empty, every
record must be an object, and errors retain the original filename and line
number. Repository, experiment, Bundle, index, evidence, and prompt paths use
explicit resolution policies rather than the process working directory.

Experiment-manifest resources are resolved once by the manifest loader. Formal
validation and preflight consume that resolved path contract and do not
reinterpret raw paths. Manifest-relative paths use the experiment directory,
repository-relative paths use the repository root, and Bundle-relative paths
use the Bundle root; no Formal resource path depends on the process current
working directory.

Selected DEV evidence is frozen as a portable canonical path plus SHA-256.
Machine-specific resolved absolute paths are diagnostic only and are not
authoritative. Complete freezes containing the legacy string-form selected DEV
evidence path must be regenerated. Before loading an embedding model or writing
a persisted index, the official index builder runs full experiment-level
`index_build_candidate` validation. Strict FireKG validation permits exact
strings or protocol-approved integers for identifier fields (never booleans or
floats), while evidence text, relation labels, citations, URLs, and source paths
must be exact non-empty strings and are never accepted by numeric coercion.

### Formal path and strict FireKG closure contract

- An absolute path whose resolved target is inside the declared repository,
  experiment-manifest directory, or Runner Bundle root is stored under the
  corresponding relative policy with a POSIX canonical path. Classification
  is CWD-independent, and a symlink cannot escape an internal root.
- A complete freeze rejects genuinely external base/shared/method configs,
  Runner Bundle, selected DEV evidence, E-KELL prompts, Dense/E-KELL indexes,
  and freeze manifest with
  `complete_freeze_external_resource_not_portable:<resource>`. Draft freezes
  may diagnose them only as `external=true`, `portable=false`, and
  `resolved_path_authoritative=false`.
- Selected DEV authority is the canonical repository-relative path plus
  SHA-256. Legacy string-only complete freezes must be regenerated.
- Strict FireKG identifiers are exact: strings are never trimmed; surrounding
  whitespace and control characters fail; exact integers are canonicalized to
  decimal identity; booleans, floats, lists, and objects fail.
- Alias list elements and triple `evidence`/`description`/`text`/`content`
  fields that enter retrieval text must be exact strings. The documented
  comma/semicolon scalar alias compatibility form remains string-only.
- Explicit `triple_id`/`edge_id` values are unique. Without one, duplicate
  identity includes the fact and all declared provenance references, including
  source/chunk/citation aliases: same provenance fails, different provenance is
  allowed.
- `index_build_candidate` completes before embedding initialization or index
  writes. The offline fake-embedding lifecycle and local tests are structural
  evidence only, not real Formal results or proof that remote GitHub Actions
  passed.

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

For the Formal comparison suite, `comparison_suite_methods` is the sole ordered method-set authority. The `methods` array is an unordered configuration registry: it must define the five required method configs exactly once and may retain disabled non-comparison entries such as `ekell_style_enhanced`, but no non-comparison entry may be enabled for a Formal comparison-suite run.

## 4. Formal interop workflow

**Only formal entrypoint.** `.example` files are templates only — copy before running.

### Unified decision comparison suite (preferred for evaluator handoff)

All five methods share the same Runner Bundle input and independently emit structured decision + natural-language response + per-method `firebench-interop-v1` JSONL. Natural language is for human review; decision fields are the primary comparison object for the main-project evaluator.

**Execution modes:** Dry run allows `--limit`, heuristic LLM, smoke embedding fixtures, and optional temporary index rebuild for wiring tests. **`formal_result` is always false** in dry run (technical diagnostics may still pass). DEV may use real config and optional `--enable-dev-aliases` on subsets; outputs remain non-formal. Freeze-candidate validation applies paper-facing model, method, index, Bundle, checksum, and exact-type requirements while allowing the freeze file currently being created to be absent. Formal requires a frozen non-`.example` experiment manifest with explicit exact non-empty string identity fields, **validates the frozen Runner Bundle identity** (per-file checksums, input cases, manifest-declared in-bundle prediction schema, corpus), **forbids `--limit`**, processes the **complete** Runner Bundle case set, enforces **one shared generation-model identity** across all five methods (provider, model, version, temperature, top_p, max_tokens, seed, enable_thinking), requires persisted **directory** indexes for Dense and E-KELL with **explicit** `actual_embedding_used=true` and `smoke_fallback_used=false`, runs **five-method resource preflight** (including all E-KELL prompt files and logical components) before any LLM call, enforces strict JSON **array** types in decision parsing, records separate **index checksum** vs **manifest-file SHA**, and publishes predictions **transactionally** (temp dir → atomic publish only after all methods pass). `formal_result=true` requires runtime evidence plus successful transactional publish.

Complete freeze generation verifies the Runner Bundle aggregate identity in addition to per-file input and schema checksums. A producer-declared Bundle checksum is optional, but when present it must be a lowercase SHA-256 and exactly match the consumer-computed Bundle hash. Freeze-candidate validation accepts only `freeze_status=provisional`; mark the experiment manifest `frozen` only after the complete freeze file has been generated and manually reviewed.

Freeze-candidate validation performs full persisted-index integrity checks for Dense, Hybrid, and E-KELL resources without rebuilding indexes. It verifies `documents.jsonl`, `embeddings.npy`, semantic document checksums, file checksums, final index checksums, manifest SHA-256, model identity, dimensions, normalization policy, corpus/KG identity where applicable, `actual_embedding_used=true`, and `smoke_fallback_used=false`. Complete freezes contain full Dense and E-KELL file-level identity (`documents_checksum`, `documents_file_checksum`, `embeddings_checksum`, `corpus_checksum`, plus `kg_checksum` for E-KELL), not only top-level index checksums. Hybrid explicitly inherits Dense `index_checksum` and `index_manifest_sha256`.

Formal preflight revalidates the live Dense, Hybrid, and E-KELL persisted indexes before any LLM client build or prediction generation and requires every frozen identity field to match the live identity exactly. Replacing a frozen index with a different internally valid index is rejected. `normalize_embeddings` is a real build parameter: it is executed during index construction, recorded in manifests, included in canonical index checksums, checked against method configuration, frozen, and rechecked at runtime. Persisted embeddings must be finite, nonzero, and unit norm within tolerance when normalization is enabled. Indexes built before `normalize_embeddings` entered the canonical checksum must be rebuilt; they are not silently migrated.

The complete-freeze fail-fast order is Runner Bundle load and aggregate checksum, freeze-candidate validation, strict persisted-index hashing, payload construction, complete validation, then atomic replace. At execution time, runtime evidence records the index checksum, manifest SHA-256, documents-file checksum, embeddings checksum, and normalization policy; the runtime identity is compared with preflight again before its LLM client is created, so a manifest change after preflight is rejected.

Complete-freeze output is written transactionally: the temporary output is removed on any failure, including validation/runtime/OS/replace errors, and any existing final freeze file is preserved. Formal input JSONL diagnostics keep original source line numbers after blank lines. Formal `case_id` values must already be canonical exact strings: non-empty, no surrounding whitespace, no control characters, and no duplicate canonical IDs.

**Pre-formal contract checks (read-only main-project reference):**

```bash
python scripts/check_firebench_contract_snapshot.py --main-repo ../fire-agent-demo
python scripts/check_firebench_taxonomy_snapshot.py --main-repo ../fire-agent-demo
python scripts/check_comparison_readiness.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle path/to/frozen_runner_bundle \
  --method-set comparison_suite
python scripts/check_output_taxonomy.py --prediction-dir outputs/formal/test_public/predictions
```

```bash
# Dry run (heuristic/smoke OK for local wiring checks)
python scripts/run_decision_comparison_suite.py \
  --runner-bundle path/to/runner_bundle \
  --method-set comparison_suite \
  --execution-stage dry_run \
  --limit 3 \
  --prediction-dir outputs/interop/dry_run/predictions \
  --decision-dir outputs/decision_runs/dry_run

# Formal (after real resources + freeze; uses Bundle prediction_schema.json)
python scripts/run_decision_comparison_suite.py \
  --runner-bundle path/to/frozen_runner_bundle \
  --method-set comparison_suite \
  --execution-stage formal \
  --formal-run-root outputs/formal/test_public \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml
```

Formal run root layout:

```text
outputs/formal/test_public/
├── predictions/
├── decisions/
├── suite_summary.json
├── run_manifest.json
└── diagnostics/
```

Formal comparison-suite artifacts (recommended unified entrypoint: `run_decision_comparison_suite.py`):

```text
outputs/formal/test_public/
├── predictions/          ← hand to fire-agent-demo evaluator
├── decisions/
├── suite_summary.json
├── run_manifest.json
└── diagnostics/
```

Evaluator handoff path:

```text
outputs/formal/test_public/predictions/
```

Legacy Formal layout (same parent required; equivalent to `--formal-run-root` above):

```bash
python scripts/run_decision_comparison_suite.py \
  --runner-bundle path/to/frozen_runner_bundle \
  --method-set comparison_suite \
  --execution-stage formal \
  --prediction-dir outputs/formal/test_public/predictions \
  --decision-dir outputs/formal/test_public/decisions \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml
```

Dry-run diagnostic artifacts (non-formal wiring checks; `formal_result` is always false):

```text
outputs/interop/dry_run/predictions/
outputs/decision_runs/dry_run/
```

### Legacy combined interop runner

Separate entrypoint (`run_interop_baselines.py`); **not** the recommended Formal artifact layout for the five-method comparison suite:

```text
outputs/interop/comparison_suite_v1/predictions.jsonl
outputs/dry_run/comparison_suite_v1/predictions.jsonl
outputs/dry_run/comparison_suite_v1/run_manifest.json
```

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
| Schema authority | Bundle `manifest.files.prediction_schema` file inside the Runner Bundle, with matching SHA-256 in `manifest.checksums` and freeze identity |
| Scoring authority | `fire-agent-demo` shared evaluator |

**Formal comparison suite compliance (offline-tested):**

- Manifest method entries are resolved before config merge (`get_method_entry` → `build_method_config`).
- Two-phase compliance: `pre_publish_compliance_passed` (no publish required) → method/cache runtime close → staged final summary/manifest in temp root → transactional publish commit → `formal_result=true` already present at first rename. Runtime cleanup failures stop before staged validation and commit.
- Formal temp artifacts are created **only after** static validation and five-method preflight pass.
- Formal diagnostics and failure records live in an external **control root** (`.<run-root-name>.control/`) and never mutate the published run root before commit.
- Formal publication uses a **single same-filesystem run root** (`--formal-run-root`); commit is one directory rename. **No core formal artifact is rewritten after commit.**
- `--formal-run-root` is the recommended Formal output interface and automatically derives `predictions/` and `decisions/` beneath one immutable run root. Legacy `--prediction-dir` / `--decision-dir` remain supported only when both paths share that same Formal root.
- `jsonschema` is a core runtime dependency because frozen Draft 2020-12 schema validation is mandatory for Formal execution.
- Dense and E-KELL runtimes are cache-owned and close before staged validation and transactional publish. Hybrid wrappers are method-owned (`owns_dense_runtime=False`) and do not close their shared Dense dependency. Runtime cleanup failures cannot mask the original suite failure and prevent commit when there is no primary suite failure.
- Release-readiness distinguishes engineering readiness from empirical readiness. Engineering gate failures return a non-zero exit code; pending real experiments do not fail engineering CI.
- Formal numeric parameters require finite exact YAML numbers; NaN and positive or negative infinity are forbidden. Formal model, backend, version, environment-variable, prompt, index, and manifest identity fields require exact non-empty YAML strings without implicit string coercion.
- Publish phases: **PREPARE** (backup) → **STAGED PACKAGE** (final `suite_summary.json`, `run_manifest.json`, preflight copy) → **COMMIT** (rename temp run root) → **CLEANUP** (best-effort backup removal; failures write warnings/receipts to control root only).
- Post-commit cleanup or receipt failures are non-destructive warnings; committed runs are never rolled back.
- Immutable `suite_summary.json` records staged compliance and atomic publication success but does **not** pre-declare backup cleanup outcome (`transactional_cleanup_complete: null`). Actual cleanup status lives in external `publish_receipt.json`.
- Returned Python summary may include `transactional_publish_runtime` and `post_commit_warnings` without rewriting the committed run root.
- Staged-package validation reparses every prediction JSONL against the frozen Runner Bundle prediction schema and schema SHA, checks exact case/method identity, validates summaries and supplemental decision artifacts, and recomputes all run-manifest hashes before rename.
- Immutable Formal run manifests record both `input_cases_provenance` and `prediction_schema_provenance`; staged validation checks those values against preflight before publish.
- Formal verifies the actual runtime embedding backend and `normalize_embeddings` policy against both method configuration and persisted index metadata.
- Runtime caches are scoped to one comparison-suite invocation and cannot leak across runs.
- Embedding backend injection is invoked only for Dense, Hybrid, and E-KELL (`embedding_backend_factory`); Direct LLM and BM25 never request an embedding backend.
- Run manifests hash predictions, method summaries, decisions, responses, and unmapped-taxonomy artifacts.
- Manifest artifact paths are validated with both POSIX and Windows path semantics (including drive-qualified, root-relative, UNC, and device-namespace forms) and must resolve inside the staged run root.
- The frozen prediction schema is parsed, checksum-validated, and verified as a Draft 2020-12 JSON Schema once before staged record validation. Formal execution requires `manifest.files.prediction_schema` to identify a schema file inside the frozen Runner Bundle; the Bundle manifest must declare the schema SHA-256, and the consumer-computed hash must match both the Bundle declaration and frozen experiment identity.
- Schema meta-validation, runtime validation, and staged prediction validation preserve the same primary schema filename and schema `$id`; under the single-schema Bundle protocol, the primary filename and primary `$id` may self-reference.
- Formal execution never falls back to repository-local schemas. Local schemas are development snapshots only and are not registered as Formal JSON Schema resources.
- The no-network schema registry is input-driven and registers only the primary Bundle schema plus explicitly checksum-verified Bundle resources. Its behavior does not depend on source checkout, current working directory, editable installation, or wheel installation.
- Under the current single-schema Bundle protocol, Formal schema references are limited to internal fragments, the primary schema `$id`, and the primary schema filename.
- Formal embedding identity validation requires exact JSON boolean flags, exact `normalize_embeddings`, and positive JSON integer dimensions in persisted index metadata.
- Formal safety-critical numeric parameters require exact finite YAML/JSON numeric types; string-to-number, boolean-to-number, NaN, and Infinity coercion are rejected.
- Formal experiment identity fields must be explicitly declared as exact non-empty YAML strings. Explicit `null` values are rejected rather than replaced with defaults.
- Formal `input_cases.jsonl` is strict: every non-empty line must be a JSON object with an exact non-empty canonical string `case_id`; invalid JSON, non-object rows, missing/empty/non-string/duplicate `case_id`, surrounding whitespace, control characters, and empty files are rejected with original source line numbers rather than silently skipped.
- Malformed nested prediction/runtime values produce structured schema errors and do not leak raw `AttributeError` or `TypeError`.
- Missing optional boolean fields may use documented defaults, but explicitly setting those fields to `null` is invalid.
- Release-readiness engineering gates are structural repository checks; behavioral correctness is established by pytest, staged tamper tests, offline Formal E2E, and externally observed CI results.
- Runtime caches are scoped through a context-local suite cache with explicit close ownership; concurrent comparison suites in the same process do not share or clear each other's runtime objects.
- GitHub Actions covers Python 3.10–3.12 with offline compile, lint, tests, hygiene, formal-config validation, E-KELL fidelity audit, package build, reproducibility dry-run, and release-readiness checks.
- Offline full E2E exercises real guard/freeze/preflight/runtime/pipeline while injecting only external LLM transport and embedding-compute boundaries via `embedding_backend_factory`.
- New freeze manifests use explicit `runner_bundle` identity fields; legacy `runner_bundle_checksum` is opt-in via `--include-legacy-compat-fields`.
- Generated files under `outputs/` are never tracked.

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
