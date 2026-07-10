# Top-tier Paper Readiness Audit (honest gates)

Current label:

> E-KELL-style paper-faithful pipeline-level reimplementation, not official reproduction.

**Rule:** “接口已实现” ≠ “实验已验证”.

## Readiness gates (corrected)

| Gate | Ready? | Evidence | Blocker |
|---|---|---|---|
| paper_spec_complete | **true** | `docs/ekell_reproduction_spec.md`, `docs/ekell_paper_to_code_matrix.md` | — |
| kg_construction_ready | **true (scaffolding)** | `ekell_style/kg_construction/` | Official 2264-triple KG unavailable; local = substituted |
| fol_reasoning_ready | **true** | FOL executor + tests | — |
| vector_retrieval_ready | **true (interface + smoke)** | E-KELL-native vector modules; smoke rejected under paper_final | Real text2vec/index checksums on user machine |
| prompt_chain_ready | **true** | stepwise prompts + chain | Not official verbatim Appendix A |
| faithful_implementation_ready | **true (code)** | `full_pipeline.py` dual tracks | Empirically unvalidated |
| controlled_interop_ready | **true (adapter)** | interop + experiment manifest | Formal Runner Bundle pending |
| cross_repository_verified | **false** | Checklist only | Await formal Runner Bundle + hash verify + neutral evaluator |
| real_chatglm_run | **false** | Config/adapter only | User server ChatGLM-6B run |
| real_shared_llm_run | **false** | SiliconFlow config ready | User paid API run |
| expert_evaluation_complete | **false** | Protocol templates empty | Experts + IAA |
| paper_ready | **false** | Scaffold only | Experiments incomplete |

Legacy aliases still used in older docs:

- `strong_baseline_implementation_ready=true` / `strong_baseline_empirically_validated=false`
- `interop_adapter_implemented=true` / `cross_repository_interop_verified=false`
- `paper_fidelity_empirically_validated=false` / `controlled_comparison_empirically_validated=false`

## Paper main table vs supplemental

**Main table (controlled):** `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm`  
**Paper fidelity (separate):** `ekell_style_paper_fidelity`  
**Supplemental:** `dense_rag`, `hybrid_rag`, `ekell_style_enhanced`

## Freeze status

`configs/frozen/*` and `freeze_manifest.json` are **`provisional`** until complete shared-LLM DEV trial logs, selection criteria application, and selected-run evidence exist under `outputs/tuning/`.

## Formal command (do not auto-run paid API)

See `docs/final_experiment_commands.md` and `configs/experiments/paper_main_table_v1.yaml.example`.
