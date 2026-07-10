# Readiness Summary

| Gate | Value | Notes |
|---|---|---|
| configuration_prepared | true | formal local configs + frozen provisional methods |
| api_environment_available | present_or_unknown | check via `scripts/show_experiment_state.py` (no values printed) |
| real_llm_config_ready | true | `configs/models/shared_real_model.yaml` (gitignored) |
| real_embedding_config_ready | true | `text2vec` + BAAI/bge-m3 candidate; index not built |
| main_project_v1_ready | false | `scripts/check_main_project_readiness.py` |
| runner_bundle_ready | false | manifest `bundle` placeholder |
| real_model_calls_executed | false | execution lock active |
| real_shared_llm_run | false | no paid API in preparation phase |
| real_chatglm_run | false | paper-fidelity deferred |
| embedding_index_built | false | `index_status: not_built` |
| cross_repository_real_dry_run | false | deferred |
| formal_experiment_started | false | deferred |
| method_registry_converged | true | `src/external_baselines/method_registry.py` |
| formal_interop_entrypoint | true | `scripts/run_interop_baselines.py` (+ readiness lock) |
| formal_config_validator_ready | true | template + formal modes |
| execution_intentionally_deferred | true | see `docs/experiments/staged_execution_plan.md` |
| cross_repository_interop_verified | false | formal shared-LLM + main evaluator still pending |
| cross_repository_contract_tool_ready | true | `scripts/verify_cross_repo_contract.py` |
| cross_repository_contract_verified | false | heuristic contract smoke only; not formal experiment |
| paper_ready | false | **must remain false** |

**Do not claim:** experiment ready, paper ready, model validated, formal run verified.

Baseline 工程和正式配置已准备完成。真实联调被主动推迟，直到主项目形成第一版稳定模型和正式 Runner Bundle。

Evidence (engineering only): `outputs/diagnostics/validation_evidence.json` (may be stale; re-run collectors after changes).

See also: `docs/experiments/staged_execution_plan.md`, `docs/fidelity/method_fidelity_matrix.md`.
