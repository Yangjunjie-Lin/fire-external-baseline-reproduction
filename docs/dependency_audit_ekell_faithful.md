# Dependency Audit: `ekell_style_faithful`

**Goal:** Prove the main-table faithful method does not call dense RAG, hybrid RAG, enhanced hooks, or target-system safety/routing code.

## Static import closure (faithful module)

Allowed roots for `src/external_baselines/ekell_style/pipeline.py` and its local ekell helpers:

- `external_baselines.ekell_style.*` (scenario_parser, entity_matcher, kg_loader, subgraph_retriever, prompt_chain)
- `external_baselines.common.*` (llm_client, schema, checksums, text_utils, io)
- `external_baselines.evaluation.normalizer` (injection **default off**)

Forbidden for faithful:

| Forbidden dependency | Status |
|---|---|
| `external_baselines.dense_rag` | Must not import |
| `external_baselines.hybrid_rag` | Must not import |
| `external_baselines.ekell_style.enhanced_pipeline` | Must not import |
| `fire_agent_demo` | Must not import |
| SAFE-Router / Safety Checker / Dynamic REG / HITL / risk scoring APIs | Must not import/call |

## Runtime guards

- `run_scenario` sets `embedding_scorer=None`.
- Config flags `dense_entity_retrieval`, `hybrid_subgraph_ranking`, `reranker`, `self_consistency`, `structured_verification` raise if true under faithful.
- Enhanced lives only in `enhanced_pipeline.py` (supplemental).

## Automated test

`tests/test_ekell_faithful_dependency_audit.py` AST-scans faithful pipeline + recursively imported `ekell_style` modules for forbidden imports.

## Paper table role

- Main table: `direct_llm`, `bm25_rag`, `ekell_style_faithful`
- Supplemental only: `dense_rag`, `hybrid_rag`, `ekell_style_enhanced`
