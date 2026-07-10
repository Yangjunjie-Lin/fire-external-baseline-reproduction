# Readiness Summary

| Gate | Value | Notes |
|---|---|---|
| method_registry_converged | true | `src/external_baselines/method_registry.py` |
| formal_interop_entrypoint | true | `scripts/run_interop_baselines.py` |
| ekell_controlled_code_ready | true | `full_pipeline.run_controlled_shared_llm` |
| ekell_paper_fidelity_interface_ready | true | config/adapter only |
| cross_repository_interop_verified | false | await formal shared-LLM + main evaluator |
| real_shared_llm_run | false | user credentials / paid API |
| real_chatglm_run | false | user server |
| actual_lightrag | false | fallback_only |
| actual_microsoft_graphrag | false | fallback_only |
| expert_evaluation_complete | false | protocol templates only |
| paper_ready | false | experiments incomplete |

See also: `docs/top_tier_readiness_audit.md`, `docs/fidelity/method_fidelity_matrix.md`.
