# Current Project Status

## Current phase

```text
Configuration prepared — execution intentionally deferred
waiting for main project first_model_v1_ready + formal Runner Bundle
```

Baseline engineering and formal configuration are prepared. Real cross-repository dry runs and formal experiments remain **locked** until the main project publishes its first stable model version.

## Valid claim

The repository is an **engineering-complete external baseline scaffold** with formal configs, readiness gates, and deferred execution locks. **No formal experiment has been started.**

It is **not** paper-ready, **not** experiment-ready, and **not** empirically validated on shared real LLMs in a cross-repo setting.

## Preparation complete (this phase)

| Item | Status |
|---|---|
| Shared real LLM config (`configs/models/shared_real_model.yaml`) | prepared (env vars only; gitignored) |
| Controlled main-table manifest | prepared (`bundle` placeholder) |
| E-KELL vector embedding config (`text2vec` + BAAI/bge-m3 candidate) | prepared; index not built |
| Dense/Hybrid supplemental configs | prepared; disabled for main table |
| Main-project readiness checker | `scripts/check_main_project_readiness.py` |
| Experiment state + execution lock | `configs/local/experiment_state.yaml` + `run_interop_baselines.py` |
| Staged execution plan | `docs/experiments/staged_execution_plan.md` |

## Still pending (deferred)

- Shared real SiliconFlow LLM **runs** (no API calls in this phase)
- Real embedding index build / download
- Cross-repository real dry run (1–3 cases)
- DEV tuning and config freeze
- Formal TEST run and main-project evaluator scoring

## Implemented (code)

| Area | Status |
|---|---|
| Method registry (single source of truth) | implemented |
| `direct_llm` | implemented; heuristic smoke only |
| `bm25_rag` | implemented; deterministic sparse retrieval |
| E-KELL controlled full pipeline | Level 3 data-compatible pipeline-level reimplementation |
| E-KELL paper-fidelity interface | ChatGLM-6B config/adapter ready; no empirical run |
| Supplemental dense/hybrid/enhanced | implemented; smoke dense ≠ formal |
| Runner Bundle input (`input_cases.jsonl`) | implemented |
| Canonical firebench-interop-v1 JSONL output | implemented |
| Formal config validator (template + formal modes) | implemented |
| Paper-fidelity formal validation (same LLM/vector guards as controlled) | implemented |
| Gold isolation / schema / checksum validation | implemented |

Formal runs must use copied configs (not `.example`). Template validation (`--allow-placeholders`) does not authorize a paper run.

- Shared real SiliconFlow LLM runs
- ChatGLM-6B paper-fidelity run
- Real dense embedding evaluation
- Actual LightRAG indexing + query
- Actual Microsoft GraphRAG workspace + indexing + query
- Expert evaluation / IAA
- Formal statistics / paper tables

## Method layers

| Layer | Methods |
|---|---|
| Formal main table | `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm` |
| Paper-fidelity (separate) | `ekell_style_paper_fidelity` |
| Supplemental | `dense_rag`, `hybrid_rag`, `ekell_style_enhanced` |
| Fallback / legacy | `lightrag`, `microsoft_graphrag`, `fallback_graph_retrieval`, `ekell_style_legacy_bm25` |

## Authority split

| Authority | Owner |
|---|---|
| Prediction generation | this repository |
| Benchmark / scoring | `fire-agent-demo` shared evaluator |

Local proxy metrics are **diagnostic only**.

## Forbidden upgrades of this status

Do not claim: fully completed · experimentally proven · official reproduction · paper-final · top-tier ready results.
