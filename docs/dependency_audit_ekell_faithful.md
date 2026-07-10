# Dependency Audit: Complete E-KELL Pipelines

**Methods covered**

- Main table controlled: `ekell_style_controlled_shared_llm` (`full_pipeline.py`)
- Paper fidelity: `ekell_style_paper_fidelity` (`full_pipeline.py`)
- Legacy alias: `ekell_style_faithful` → controlled full pipeline
- Legacy BM25 scaffold only: `ekell_style_legacy_bm25` (`pipeline.py`) — diagnostics, not main table
- Supplemental: `ekell_style_enhanced` (`enhanced_pipeline.py`)

## Forbidden for complete / faithful tracks

| Dependency | Rule |
|---|---|
| `fire_agent_demo` / SAFE-Router / Safety Checker / Dynamic REG / HITL / target risk scoring | Forbidden |
| `external_baselines.dense_rag` | Forbidden (generic baseline) |
| `external_baselines.hybrid_rag` | Forbidden |
| `ekell_style.enhanced_pipeline` | Forbidden import into full/faithful path |

## Allowed and required for paper-faithful architecture

| Module | Role |
|---|---|
| `ekell_style.vector_retriever` / `vector_index` / `embedding_backends` | Paper Sec 4 vector KG retrieval |
| `ekell_style.logical_query` | FOL p/i/u/n |
| `ekell_style.neighborhood_expander` | Neighborhood expansion |
| `ekell_style.stepwise_prompt_chain` | Logical prompt chain |
| `ekell_style.kg_construction` | Sec 3.1 scaffolding |

**Corrected rule:** faithful must **not** set `embedding_scorer=None` as a fidelity requirement. It must use the E-KELL-native vector retriever and must not call generic dense_rag.

## Automated tests

`tests/test_ekell_faithful_dependency_audit.py`

## Paper table roles

- Main: `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm`
- Paper fidelity experiment (separate): `ekell_style_paper_fidelity`
- Supplemental: `dense_rag`, `hybrid_rag`, `ekell_style_enhanced`
