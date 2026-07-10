# Resource Access Matrix

| Method | Scenario | Corpus | KG | Dynamic Snapshot | LLM | Target Modules |
|---|---|---|---|---|---|---|
| direct_llm | Yes | No | No | No | Shared | **No** |
| bm25_rag | Yes | Yes | No | No | Shared | **No** |
| dense_rag | Yes | Yes | No | No | Shared | **No** |
| hybrid_rag | Yes | Yes | No | No | Shared | **No** |
| ekell_style_faithful | Yes | Yes | Yes | No* | Shared | **No** |
| ekell_style_enhanced | Yes | Yes | Yes | Optional via separate state-aware id | Shared | **No** |
| lightrag (actual) | Yes | Yes | Via package index | Per package | Shared | **No** |
| microsoft_graphrag (actual) | Yes | Yes | Via package index | Per package | Shared | **No** |
| fallback_graph_retrieval | Yes | Yes | Local fallback graph | No | Shared | **No** |

\* Faithful E-KELL-style architecture as implemented here does not consume dynamic state snapshots. For Track A3, use a separately named enhanced state-aware method — never relabel as faithful.

## Fairness tracks

- **A1** text-only: `direct_llm`
- **A2** text + same corpus/KG: RAG + E-KELL-style
- **A3** text + corpus + dynamic snapshots: only methods that architecturally support snapshots

## Shared constraints

Same scenarios, case IDs, corpus/KG snapshot, model/provider, temperature/top_p, max tokens (or recorded equivalent budget), language requirement, evaluator, gold, and test freeze.
