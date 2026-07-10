# Top-tier Paper Readiness Audit (honest gates)

Current label:

> E-KELL-style paper-faithful pipeline-level reimplementation, not official reproduction.

**Rule:** “接口已实现” ≠ “实验已验证”.

## Readiness gates (corrected)

| Gate | Ready? | Evidence | Blocker |
|---|---|---|---|
| baseline_independence_ready | **true** | No `fire_agent_demo` imports; gold isolation tests | Keep CI green |
| interop_adapter_implemented | **true** | `scripts/run_interop_baselines.py`, schema adapter, experiment manifest | — |
| cross_repository_interop_verified | **false** | Checklist only; no formal main-project Runner Bundle consumed yet | Await formal Runner Bundle + schema/scenario/corpus hash verify + neutral evaluator |
| strong_baseline_implementation_ready | **true** | Main-table methods implemented & smoke-tested; BM25/E-KELL strengthened; no safety injection | — |
| strong_baseline_empirically_validated | **false** | No shared-LLM DEV trials / selected-run evidence / paper metrics | User must run real SiliconFlow shared model + evaluator |
| faithful_reproduction_ready | **true (Level 3 code)** | `ekell_style_faithful` + paper-to-code audit + dependency audit | Not Level 5; not empirically validated on paper data |
| actual_graphrag_ready | **false** | fallback_only | Real index+query absent |
| real_llm_config_ready | **true (config)** | SiliconFlow env names aligned with fire-agent-demo; `.env.example` | Credentials are user-supplied; agents must not auto-call paid APIs |
| real_llm_empirically_run | **false** | No paid API executed in this session | User run required |
| final_comparison_ready | **false** | Protocol + manifest ready | Bundle + real LLM + evaluator + license + expert scores |
| paper_ready | **false** | Scaffold only | Experiments incomplete |

## Paper main table vs supplemental

**Main table (only):** `direct_llm`, `bm25_rag`, `ekell_style_faithful`  
**Supplemental/extended (must not replace faithful):** `dense_rag`, `hybrid_rag`, `ekell_style_enhanced`

## Freeze status

`configs/frozen/*` and `freeze_manifest.json` are **`provisional`** until complete shared-LLM DEV trial logs, selection criteria application, and selected-run evidence exist under `outputs/tuning/`.

## Formal command (do not auto-run paid API)

See `docs/final_experiment_commands.md` and `configs/experiments/paper_main_table_v1.yaml.example`.
