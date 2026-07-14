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
unified decision I/O ready for five-method comparison
FireBench taxonomy contract ready
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

python scripts/check_firebench_contract_snapshot.py \
  --main-repo ../fire-agent-demo

python scripts/check_firebench_taxonomy_snapshot.py \
  --main-repo ../fire-agent-demo

# Offline decision-suite + taxonomy wiring (heuristic LLM; temporary fixtures)
python -m pytest tests/test_decision_comparison_suite.py tests/test_firebench_taxonomy.py -q
```

**Not allowed:** real LLM calls, embedding download, full index build, cross-repo dry run, formal experiment.

`build_comparison_indexes.py --validate-only` is a hard gate even in Stage 0:
any Bundle, exact-type configuration, Dense, Hybrid dependency, E-KELL, KG, or
persisted-index failure produces `ok=false` and exit code `1`; only a completely
valid report exits `0`. Relative resource paths are resolved from their declared
repository, experiment-manifest, or Bundle policy and never from an arbitrary
current working directory.

The experiment-manifest loader is the single authority for base, shared,
method, Runner Bundle, and freeze-manifest resource paths. Formal validation
and preflight consume its resolved contract without raw-path fallback.
Machine-local absolute paths are diagnostic only. Selected DEV evidence is
frozen with a canonical portable path and SHA-256; a complete legacy freeze
whose selected DEV evidence is a string must be regenerated.

Before any embedding backend initialization or persisted-index write, the
official builder runs full experiment-level `index_build_candidate` validation.
Strict FireKG validation separates identifier fields from semantic text:
protocol-approved integer IDs are allowed, but booleans and floats are not;
evidence text, relation labels, citations, URLs, and source paths must be exact
non-empty strings without numeric coercion.

Taxonomy note: structured decision IDs must match the FireBench taxonomy snapshot. Formal aliases mirror main-project `taxonomy.py` (commit `f228867480eb369c2b55cde3185af548965a23a5`). DEV-only aliases require explicit enable and are forbidden in formal runs. Final prediction JSONL must contain canonical IDs only; parser requires all decision/response/action fields to be explicitly present in formal mode. Unknown IDs fail formal validation. Freeze taxonomy before TEST.

After DEV selection, the complete freeze binds the prompt directory selected by
the final merged E-KELL configuration, all required prompt-file SHA-256 values,
and the full prompt-tree SHA-256. Official build, freeze, and Formal preflight
also require non-empty `entities.jsonl`, `relations.jsonl`, `triples.jsonl`, and
`evidence_chunks.jsonl`; every non-empty line must decode to a JSON object.

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
- manifest + method configs updated with real paper-facing values
- freeze-candidate validation passes before creating the complete freeze file
- experiment manifest remains `freeze_status: provisional` during freeze-candidate validation
- Runner Bundle aggregate identity verified: consumer hash required; producer checksum optional but, when present, must be valid SHA-256 and match the consumer hash
- Dense, Hybrid, and E-KELL persisted indexes pass full integrity validation without rebuild, including file-level checksums, manifest SHA-256, normalization policy, finite/nonzero embedding values, and unit-norm checks when normalization is enabled
- config checksums recorded
- prompt hash fixed
- LLM `model` / `model_version` frozen in YAML (`model_source=yaml_config`)
- embedding `model_version` fixed (replace `REQUIRED_BEFORE_REAL_INDEX_BUILD`)

Freeze order:

1. Finish DEV selection
2. Save a non-`.example` experiment manifest with real model, method, Bundle, and index identities
3. Run `create_freeze_manifest.py`; non-draft mode loads and validates the Formal Runner Bundle aggregate first, then performs `freeze_candidate` validation and strict Dense/E-KELL persisted-index validation before writing the freeze atomically
4. Treat `--draft` output as development-only and incomplete
5. Manually review the complete freeze identity
6. Mark the experiment manifest `freeze_status: frozen` only after review, then use final Formal validation with a reference to the existing complete freeze file

Install for real Dense/Hybrid/E-KELL embeddings: `pip install -e ".[llm,embeddings]"` (or `requirements-optional-embeddings.txt`). First `text2vec` encode may download the model; pre-cache before formal runs.

```bash
python scripts/create_freeze_manifest.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --selected-dev-run outputs/tuning/selected_dev_run.json \
  --bundle <runner_bundle> \
  --output configs/freeze/comparison_freeze_manifest_v1.json

python scripts/validate_formal_config.py \
  --validation-stage freeze_candidate \
  --method-set comparison_suite \
  --config configs/experiments/controlled_main_table_v1.yaml

python scripts/validate_formal_config.py \
  --validation-stage formal \
  --method-set comparison_suite \
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
- **no** `--limit` (complete Runner Bundle case coverage enforced)
- **no** `--allow-partial`
- **no** `--enable-dev-aliases`
- frozen Runner Bundle identity validated against freeze manifest (fail-closed; complete `runner_bundle` block with bundle/input/schema/corpus SHA256; `manifest.files.prediction_schema` must point to an in-bundle schema with a matching `manifest.checksums` SHA-256)
- producer-declared Bundle checksum is optional, but when present must exactly match the consumer-computed Bundle hash
- manifest method entries resolved before per-method config merge
- `comparison_suite_methods` is the sole ordered five-method authority; `methods` is an unordered configuration registry and may retain disabled non-comparison entries only
- two-phase formal compliance: pre-publish checks (no publish required) → method/cache runtime close → staged final summary/manifest in temp root → transactional publish commit → `formal_result=true` at first rename; runtime cleanup failure stops before staged validation and commit
- formal temp root created only after static validation and five-method preflight; preflight/failure records written to external `.control/` directory (never mutates published run root before commit)
- single same-filesystem `--formal-run-root` publication (one directory rename); **no core formal artifact rewritten after commit**
- new freeze manifests use explicit `runner_bundle` block (legacy top-level checksum fields opt-in only)
- one shared generation-model identity across all five comparison methods
- persisted Dense/E-KELL **directory** indexes (built via `build_comparison_indexes.py`; no legacy JSON or runtime rebuild; manifest must explicitly record real embedding)
- complete freeze contains full Dense and E-KELL file-level identity (`documents_checksum`, `documents_file_checksum`, `embeddings_checksum`, `corpus_checksum`, plus E-KELL `kg_checksum`) and semantic identity (`backend`, `model_name`, `model_version`, `dimension`, `normalize_embeddings`, `actual_embedding_used`, `smoke_fallback_used`); Hybrid inherits Dense `index_checksum` and `index_manifest_sha256`
- indexes built before `normalize_embeddings` entered the canonical checksum must be rebuilt; they are not silently migrated or grandfathered into Formal runs
- five-method **preflight** passes before any LLM call (external control root + copy under `diagnostics/` in staged run root; includes E-KELL prompt files)
- **transactional** publish: runtime cleanup → staged package validation (reparsed predictions against frozen Runner Bundle schema + supplemental artifact hash checks) → PREPARE → COMMIT → CLEANUP (backup cleanup failures are control-root warnings only; immutable summary does not pre-declare cleanup success)
- Formal preflight revalidates the live Dense, Hybrid, and E-KELL persisted indexes before any LLM client build or prediction generation, rejects internally valid replacement indexes whose identity differs from the freeze, and verifies the actual runtime embedding backend plus `normalize_embeddings` policy against both method configuration and persisted index metadata
- prepared runtime evidence records file-level identity and is compared with preflight before that method's LLM client is initialized; a manifest change after preflight fails with `runtime_index_identity_changed_after_preflight`
- runtime caches are scoped to one comparison-suite invocation and cannot leak across runs
- embedding backend injection is invoked only for Dense, Hybrid, and E-KELL
- run manifests hash predictions, method summaries, decisions, responses, and unmapped-taxonomy artifacts
- run manifests directly record input-cases provenance and prediction-schema provenance, both checked against preflight during staged validation
- manifest artifact paths are validated with both POSIX and Windows path semantics and must resolve inside the staged run root
- the frozen prediction schema is parsed, checksum-validated, and verified as a Draft 2020-12 JSON Schema once before staged record validation; meta-schema and record validation share one no-network `$ref` policy limited to internal fragments, the primary schema `$id`, and the primary schema filename under the current single-schema Bundle protocol
- formal `input_cases.jsonl` is strict and fail-closed: every non-empty line must be a JSON object with an exact non-empty canonical string `case_id`; diagnostics preserve original source line numbers and reject surrounding whitespace or control characters
- formal execution never falls back to repository-local schemas; local schema snapshots are development/diagnostic resources only and are not registered as Formal JSON Schema resources
- the no-network schema registry is input-driven and independent of source checkout, current working directory, editable installation, or wheel installation
- formal embedding identity validation requires exact JSON boolean flags, exact `normalize_embeddings`, and positive JSON integer dimensions in persisted index metadata
- formal numeric parameters require exact finite YAML/JSON numeric types; NaN and Infinity are rejected
- formal model/backend/version/environment-variable/prompt/index/manifest identity fields require exact non-empty YAML strings without implicit coercion; explicit `null` is rejected instead of replaced with defaults
- the frozen Runner Bundle is the sole Formal schema authority
- malformed nested prediction/runtime values produce structured schema errors without raw type exceptions
- runtime caches are scoped through a context-local suite cache with explicit close ownership; concurrent comparison suites in the same process do not share or clear each other's runtime objects
- GitHub Actions covers Python 3.10–3.12 with offline compile, lint, tests, hygiene, formal-config validation, E-KELL fidelity audit, package build, reproducibility dry-run, and release-readiness checks
- offline Formal E2E injects only LLM transport and embedding-compute boundaries
- producer-declared checksum and consumer-computed hash are frozen and validated separately (legacy ambiguous `bundle_checksum` rejected in formal)
- output under `outputs/interop/` (or formal directory)
- dry-run artifacts must never report `formal_result=true`

```bash
python scripts/check_firebench_contract_snapshot.py --main-repo ../fire-agent-demo
python scripts/check_firebench_taxonomy_snapshot.py --main-repo ../fire-agent-demo

python scripts/check_comparison_readiness.py \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle <frozen_runner_bundle> \
  --method-set comparison_suite

python scripts/run_decision_comparison_suite.py \
  --runner-bundle <frozen_runner_bundle> \
  --method-set comparison_suite \
  --execution-stage formal \
  --formal-run-root outputs/formal/test_public \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml
```

`--formal-run-root` alone is sufficient for Formal output layout (`predictions/` and `decisions/` are derived). Legacy `--prediction-dir` / `--decision-dir` must share the same Formal root when used.

Legacy combined runner (also no `--limit` in formal):

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
