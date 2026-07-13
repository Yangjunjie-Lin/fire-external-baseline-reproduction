# Current Project Status

## Current phase

```text
unified decision I/O ready for five-method comparison
FireBench taxonomy contract ready
real resources not yet installed
real indexes not yet built
real dry run not yet executed
formal experiment not yet executed
```

Five methods share one Runner Bundle input protocol and independently emit taxonomy-compliant structured decision JSON, natural-language response, and per-method `firebench-interop-v1` prediction JSONL. Native retrieval/reasoning designs are preserved. Formal evaluation remains owned by `fire-agent-demo`.

Structured IDs use the FireBench taxonomy snapshot (`configs/contracts/firebench_taxonomy_v1.json`). Formal aliases mirror main-project `taxonomy.py` at commit `f228867480eb369c2b55cde3185af548965a23a5`. Development-only aliases live in `configs/contracts/firebench_taxonomy_dev_aliases_v1.json` and are disabled in formal runs. Character-level normalization and exact aliases only; final prediction JSONL must contain canonical IDs only. Unknown/unmapped IDs fail formal validation.

## Execution modes

| Mode | Heuristic/smoke | Dev aliases | `--limit` | Index build | Manifest | `formal_result` |
|---|---|---|---|---|---|---|
| Dry run | allowed (fixtures) | optional (`--enable-dev-aliases`) | allowed | allowed (smoke/rebuild) | optional | **always false** |
| DEV | real or experimental config | explicit enable only | allowed (subset debug) | recommended persisted dirs | recommended | false |
| Formal | forbidden | forbidden | **forbidden** | **load-only** persisted dirs | required (non-`.example`, frozen) | runtime evidence + transactional publish |

Formal enforcement (decision suite): immutable suite summary does not pre-declare backup cleanup success (`transactional_cleanup_complete: null`); publish receipt is the cleanup authority; staged validator reparses predictions against the frozen Runner Bundle schema (parsed once, checksum-validated, Draft 2020-12 meta-schema verified, no-network `$ref` policy shared with record validation) and verifies manifest hashes including cross-platform artifact path containment; Formal requires `manifest.files.prediction_schema` to identify a schema file inside the Runner Bundle, requires the schema SHA-256 in `manifest.checksums`, and requires the live hash to match both the Bundle declaration and frozen identity; the no-network schema registry is input-driven and registers only the primary Bundle schema plus explicitly checksum-verified Bundle resources, so local schema snapshots are development diagnostics only and Formal behavior is independent of checkout/CWD/editable/wheel installation; under the current single-schema Bundle protocol, only internal fragments, the primary schema `$id`, and the primary schema filename may be referenced; Formal embedding identity validation requires exact JSON boolean flags and positive JSON integer dimensions in persisted index metadata; Formal safety-critical numeric parameters require exact finite YAML/JSON numeric types without string-to-number, boolean-to-number, NaN, or Infinity coercion; Formal model/backend/version/environment-variable/prompt/index/manifest identity fields require exact non-empty YAML strings and explicit `null` is rejected instead of replaced with defaults; malformed nested prediction/runtime values produce structured schema errors without raw type exceptions; `--formal-run-root` is the recommended Formal CLI output interface (legacy prediction/decision dirs remain compatible only when they share that root); `jsonschema` is a core runtime dependency; runtime caches are scoped through a context-local suite cache with explicit close ownership and close before staged validation/publish (Hybrid wrapper per method with `owns_dense_runtime=False`); runtime cleanup failures cannot mask the original suite error and prevent commit when there is no primary error; `embedding_backend_factory` is invoked only for Dense, Hybrid, and E-KELL; Formal CLI recommends `--formal-run-root` as the sole output path parameter; GitHub Actions covers Python 3.10-3.12 with offline compile/lint/tests/hygiene plus formal-config validation, E-KELL fidelity audit, package build, core-install smoke, reproducibility dry-run, and release-readiness audit; engineering release-readiness gates are structural repository checks that fail CI on incomplete engineering requirements while empirical/paper readiness remain honestly incomplete; post-commit warning failures are visible via stderr/return summary but cannot modify the committed run.

Current Formal identity rules: `comparison_suite_methods` is the sole ordered five-method authority, while `methods` is an unordered configuration registry that may retain disabled non-comparison entries only. Complete freeze generation runs `freeze_candidate` validation, Formal Runner Bundle aggregate checksum validation before persisted-index hashing, and strict Dense/E-KELL persisted-index integrity validation before atomic output; draft freeze output is development-only. A producer-declared Bundle checksum is optional, but when present it must be a valid lowercase SHA-256 and exactly match the consumer-computed Bundle hash. Complete freezes contain full Dense and E-KELL file-level identity (`documents_checksum`, `documents_file_checksum`, `embeddings_checksum`, `corpus_checksum`, plus E-KELL `kg_checksum`) and semantic identity (`backend`, `model_name`, `model_version`, `dimension`, `normalize_embeddings`, `actual_embedding_used`, `smoke_fallback_used`), not only top-level checksums. Hybrid inherits Dense `index_checksum` and `index_manifest_sha256`. Formal preflight revalidates the live Dense, Hybrid, and E-KELL persisted indexes before any LLM client build or prediction generation and rejects internally valid replacement indexes whose identity differs from the freeze. Persisted embeddings must be finite and nonzero, and normalized indexes must satisfy unit-norm tolerance. Indexes built before `normalize_embeddings` entered the canonical checksum must be rebuilt; they are not silently migrated. `freeze_candidate` accepts only `freeze_status=provisional`; `frozen` is reserved for the reviewed manifest after complete freeze generation. Complete-freeze temporary output is removed on any failure while preserving any existing final freeze file. Schema meta-validation, runtime validation, and staged validation share the same primary schema filename and schema `$id`. Immutable run manifests record both input-cases and prediction-schema provenance. Formal `input_cases.jsonl` is strict and fail-closed: each non-empty line must be a JSON object with an exact non-empty canonical string `case_id`; diagnostics preserve original source line numbers and reject surrounding whitespace or control characters.

Formal pre-checks (read-only against main project):

```bash
python scripts/check_firebench_contract_snapshot.py --main-repo ../fire-agent-demo
python scripts/check_firebench_taxonomy_snapshot.py --main-repo ../fire-agent-demo
python scripts/check_output_taxonomy.py --prediction-dir outputs/interop/test_public/predictions
```

## Valid claim

The repository is **ready to receive real scenarios, corpora, indexes and model resources** for unified decision comparison. **No formal experiment has been started.**

It is **not** paper-ready, **not** empirically validated, and **not** an official E-KELL reproduction.

## Preparation complete (this phase)

| Item | Status |
|---|---|
| Shared real LLM config | prepared (env vars only; gitignored) |
| `main_table` + `comparison_suite` method sets | implemented |
| Unified `DecisionOutput` + strict formal parser | implemented |
| FireBench taxonomy snapshot + formal aliases | `configs/contracts/firebench_taxonomy_v1.json`, `firebench_taxonomy_aliases_v1.json` |
| DEV-only taxonomy aliases | `configs/contracts/firebench_taxonomy_dev_aliases_v1.json` |
| Taxonomy snapshot parity checker | `scripts/check_firebench_taxonomy_snapshot.py` |
| Formal decision suite guard | `src/external_baselines/common/decision_suite_guard.py` |
| Unified five-method preflight | `src/external_baselines/common/decision_suite_preflight.py` |
| Runtime evidence / formal compliance | `src/external_baselines/common/runtime_evidence.py` |
| Runner Bundle integrity (formal) | `src/external_baselines/common/bundle_integrity.py` |
| Shared generation identity (formal) | `src/external_baselines/common/generation_identity.py` |
| Transactional formal publish | `scripts/run_decision_comparison_suite.py` (`--keep-failed-temp-artifacts` debug only) |
| Taxonomy normalizer (character-level only) | `src/external_baselines/common/taxonomy_normalizer.py` |
| Output taxonomy checker | `scripts/check_output_taxonomy.py` |
| Schema snapshot checker | `scripts/check_firebench_contract_snapshot.py` |
| Per-method decision suite runner | `scripts/run_decision_comparison_suite.py` |
| Dense real text2vec index build/load/query | implemented (fake-model tests only) |
| Hybrid BM25 + Dense + RRF | implemented; reuses Dense index |
| Shared embedding backend factory | `src/external_baselines/retrieval/embedding_backends.py` |
| Comparison readiness checker | `scripts/check_comparison_readiness.py` |
| Index build entry (`--validate-only`) | `scripts/build_comparison_indexes.py` |
| Stage-aware formal validator | template / dry_run / freeze_candidate / formal |
| Freeze manifest helper | `scripts/create_freeze_manifest.py` (freeze-candidate + Formal Bundle aggregate authority before index hashing + strict full persisted-index identity for complete freezes) |
| Staged execution plan | `docs/experiments/staged_execution_plan.md` |

## Still pending (deferred)

- Main-project v1 Runner Bundle + scenarios/corpus
- Embedding model revision mount / download
- Real Dense + E-KELL index builds
- 1–3 case dry run with shared SiliconFlow LLM
- DEV parameter selection + human freeze
- Formal TEST + main-project evaluator scoring

## Method layers

| Layer | Methods |
|---|---|
| Formal main table | `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm` |
| Comparison suite | main table + `dense_rag` + `hybrid_rag` |
| Paper-fidelity (separate) | `ekell_style_paper_fidelity` |
| Supplemental ablation | `ekell_style_enhanced` |
| Fallback / legacy | `lightrag`, `microsoft_graphrag`, `fallback_graph_retrieval`, `ekell_style_legacy_bm25` |

Dense/Hybrid never modify E-KELL controlled paper structure (`dense_entity_retrieval` / hybrid subgraph / reranker / self-consistency / structured verification remain false).

## Authority split

| Authority | Owner |
|---|---|
| Scenarios / gold / evaluator | main project (`fire-agent-demo`) |
| External baselines / predictions | this repository |
| Formal freeze | human after DEV evidence |
