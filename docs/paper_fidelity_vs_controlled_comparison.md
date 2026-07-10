# Paper Fidelity vs Controlled Comparison

These tracks must never be merged into one “reproduction result.”

| Dimension | Paper Fidelity | Controlled Comparison |
|---|---|---|
| `method_id` | `ekell_style_paper_fidelity` | `ekell_style_controlled_shared_llm` |
| Goal | Closest to E-KELL paper architecture + ChatGLM-6B interface | Same E-KELL architecture under shared FireBench resources |
| LLM | ChatGLM-6B local adapter (`configs/models/chatglm6b_local.yaml.example`) | Same SiliconFlow model/settings as fire-agent-demo |
| Output | Paper-style decision support + KG facts + trace | Shared FireBench outcome schema |
| Flags | `paper_original_output_format=true` | `controlled_output_format=true` |
| SAFE fields | Not forced | May map to shared schema; adapter must not invent safety |
| KG | Prefer paper-domain construction; local fire KG = `substituted_fire_domain_kg` | Same corpus/KG snapshot as main project |
| Empirically validated? | `false` until real ChatGLM run | `false` until real shared-LLM + evaluator |

## Enhanced (supplemental only)

`ekell_style_enhanced` may use dense/hybrid hooks. It must not replace either track above and must not be labeled paper-faithful.

## Legacy alias

`ekell_style_faithful` canonicalizes to `ekell_style_controlled_shared_llm` (complete controlled pipeline), not the old BM25-only scaffold.
