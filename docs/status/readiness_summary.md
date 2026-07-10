# Readiness Summary

| Gate | Value | Notes |
|---|---|---|
| method_registry_converged | true | `src/external_baselines/method_registry.py` |
| formal_interop_entrypoint | true | `scripts/run_interop_baselines.py` |
| formal_config_validator_ready | true | template + formal modes |
| ekell_controlled_code_ready | true | `full_pipeline.run_controlled_shared_llm` |
| ekell_paper_fidelity_interface_ready | true | config/adapter only; full formal validation enabled |
| cross_repository_interop_verified | false | formal shared-LLM + main evaluator still pending |
| cross_repository_contract_tool_ready | true | `scripts/verify_cross_repo_contract.py` |
| cross_repository_contract_verified | true | local evidence: `outputs/diagnostics/validation_evidence.json` (heuristic contract smoke only) |
| local_ci_equivalent_passed | see evidence | `scripts/collect_validation_evidence.py` |
| ci_workflow_configured | true | `.github/workflows/ci.yml` |
| ci_remote_status_known | false | remote GitHub Actions not queried here |
| real_shared_llm_run | false | user credentials / paid API |
| real_chatglm_run | false | user server |
| actual_lightrag | false | fallback_only |
| actual_microsoft_graphrag | false | fallback_only |
| expert_evaluation_complete | false | protocol templates only |
| paper_ready | false | experiments incomplete |

**Contract tool ready ≠ contract verified.** Template validation (`--allow-placeholders`) ≠ formal validation.

Evidence: `outputs/diagnostics/validation_evidence.json`

See also: `docs/top_tier_readiness_audit.md`, `docs/fidelity/method_fidelity_matrix.md`.
