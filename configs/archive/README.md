# Config archive notes

Prefer these as current sources of truth:

| Purpose | Path |
|---|---|
| Formal main-table experiment | `configs/experiments/paper_main_table_v1.yaml.example` |
| E-KELL paper fidelity | `configs/experiments/ekell_paper_fidelity.yaml.example` |
| Shared SiliconFlow model | `configs/shared_real_model.yaml.example` / `configs/models/siliconflow_shared.yaml.example` |
| ChatGLM-6B | `configs/models/chatglm6b_local.yaml.example` |
| Heuristic smoke | `configs/deterministic_heuristic_smoke.yaml` |
| Frozen method configs | `configs/frozen/` (provisional until DEV evidence) |

Older root-level LLM examples (`llm_*.yaml.example`, `paper_main_run.yaml.example`, `vanilla_rag.yaml`) are retained for compatibility but should not be the README primary path.

Physical package for BM25 remains `src/external_baselines/vanilla_rag/`; config may still say `vanilla_rag` historically while method_id is `bm25_rag`.
