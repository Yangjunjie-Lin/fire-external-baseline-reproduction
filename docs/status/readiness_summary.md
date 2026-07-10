# Readiness Summary

| Gate | Value | Notes |
|---|---|---|
| configuration_prepared | true | formal local configs + frozen provisional methods |
| comparison_suite_code_ready | true | five-method suite + shared embedding + Dense/Hybrid indexes |
| api_environment_available | present_or_unknown | check via `scripts/show_experiment_state.py` (no values printed) |
| real_llm_config_ready | true | `configs/models/shared_real_model.yaml` (gitignored) |
| real_embedding_config_ready | true | `text2vec` + BAAI/bge-m3 candidate; index not built |
| main_project_v1_ready | false | `scripts/check_main_project_readiness.py` |
| runner_bundle_ready | false | manifest `bundle` placeholder |
| dense_index_built | false | build via `scripts/build_comparison_indexes.py` after resources |
| ekell_index_built | false | separate KG/entity index path |
| real_model_calls_executed | false | execution lock active |
| real_shared_llm_run | false | no paid API in preparation phase |
| real_chatglm_run | false | paper-fidelity deferred |
| embedding_index_built | false | `index_status: not_built` |
| cross_repository_real_dry_run | false | deferred |
| formal_experiment_started | false | deferred |
| freeze_status_machine_consistent | true | template/dry_run/formal stages |
| method_registry_converged | true | `src/external_baselines/method_registry.py` |
| formal_interop_entrypoint | true | `scripts/run_interop_baselines.py` (+ readiness lock) |
| formal_config_validator_ready | true | `--validation-stage template\|dry_run\|formal` |
| execution_intentionally_deferred | true | see `docs/experiments/staged_execution_plan.md` |
| dry_run_formal_stages_separated | true | `--execution-stage dry_run\|formal` |
| yaml_model_authority | true | `SILICONFLOW_MODEL` does not silently override formal YAML |
| dist_build_artifacts_ignored | true | `dist/` / `build/` gitignored; hygiene rejects tracked wheels |
| cross_repository_interop_verified | false | formal shared-LLM + main evaluator still pending |
| cross_repository_contract_tool_ready | true | `scripts/verify_cross_repo_contract.py` |
| cross_repository_contract_verified | false | heuristic contract smoke only; not formal experiment |
| paper_ready | false | **must remain false** |

**Do not claim:** experiment complete, paper ready, empirically validated, official E-KELL reproduction.

Status line:

```text
five-method comparison implementation ready
real resources not yet installed
real indexes not yet built
real dry run not yet executed
formal experiment not yet executed
```

See also: `docs/experiments/staged_execution_plan.md`, `docs/fidelity/method_fidelity_matrix.md`, `docs/final_experiment_commands.md`.
