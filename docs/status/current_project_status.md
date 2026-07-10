# Current Project Status

## Current phase

```text
Engineering convergence before formal experiments
```

## Valid claim

The repository is an **engineering-complete external baseline scaffold** with code-level E-KELL-style reproduction and `firebench-interop-v1` interop support. **Formal experiments remain pending.**

It is **not** an official E-KELL reproduction and is **not** empirically validated on shared real LLMs.

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
| Gold isolation / schema / checksum validation | implemented |

## Not empirically completed

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
