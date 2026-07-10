# Method Fidelity Matrix

Honest status snapshot for all registered external baseline methods.
Machine-readable twin: [`method_fidelity_matrix.json`](method_fidelity_matrix.json).

**Rule:** interface implemented ≠ empirically validated. Heuristic / smoke runs are not paper-final.

| method_id | class | paper table | formal track | implementation status | empirical status | claim (short) |
|---|---|---|---|---|---|---|
| `direct_llm` | canonical_baseline | main | A_main_table | implemented_but_not_empirically_run | heuristic_smoke_only | Strong no-retrieval LLM baseline |
| `bm25_rag` | canonical_baseline | main | A_main_table | implemented_but_not_empirically_run | heuristic_smoke_only | True BM25 lexical RAG |
| `dense_rag` | supplemental_extension | comparison_suite | A_comparison_suite | real_text2vec_index_code_ready | not_empirically_run_with_real_embeddings | Dense RAG (controlled supplemental) |
| `hybrid_rag` | supplemental_extension | comparison_suite | A_comparison_suite | bm25_dense_rrf_code_ready | not_empirically_run_with_real_embeddings | Hybrid BM25+dense RRF (reuses Dense) |
| `ekell_style_controlled_shared_llm` | paper_reproduction | main | A_main_table | implemented_but_not_empirically_run | code_complete_shared_llm_run_pending | E-KELL-style reimplementation (not official) |
| `ekell_style_paper_fidelity` | paper_reproduction | paper_fidelity | B_paper_fidelity | implemented_but_not_empirically_run | chatglm6b_interface_ready_run_pending | Paper-fidelity track (ChatGLM-6B) |
| `ekell_style_enhanced` | supplemental_extension | supplemental | A_supplemental | implemented_but_not_empirically_run | supplemental_only_unvalidated | Enhanced hooks only |
| `ekell_style_legacy_bm25` | legacy_diagnostic | diagnostics | legacy_diagnostic | legacy_scaffold_retained | diagnostics_only | Legacy BM25 scaffold (not controlled) |
| `lightrag` | official_system_adapter | not main | adapter_fallback | **fallback_only** | no_actual_index_query_integration | Adapter stub unless actual index+query |
| `microsoft_graphrag` | official_system_adapter | not main | adapter_fallback | **fallback_only** | no_actual_index_query_integration | Adapter stub unless actual index+query |
| `fallback_graph_retrieval` | fallback_only | not main | adapter_fallback | **fallback_only** | local_kg_subgraph_fallback | Local KG fallback (never GraphRAG board) |

## Alias notes

| Alias | Canonical |
|---|---|
| `vanilla_rag` | `bm25_rag` |
| `ekell`, `ekell_style`, `e-kell-style`, `ekell_style_faithful` | `ekell_style_controlled_shared_llm` |
| `graphrag` | `microsoft_graphrag` |
| *(none)* | `ekell_style_legacy_bm25` is **not** aliased to controlled |

## Reporting rules

1. Main-table controlled comparison: `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm` (`--method-set main_table`).
2. Five-method fair system contrast: add `dense_rag` + `hybrid_rag` via `--method-set comparison_suite`.
3. Dense/Hybrid are controlled supplemental baselines; they never enter E-KELL paper-fidelity and never enable E-KELL enhanced hooks.
4. Paper-fidelity ChatGLM track is separate: `ekell_style_paper_fidelity`.
5. `ekell_style_enhanced` remains supplemental ablation only (not in comparison_suite).
6. LightRAG / Microsoft GraphRAG remain `fallback_only` until actual package indexing + query + checksums are recorded.
7. Source of truth for IDs/wiring: `src/external_baselines/method_registry.py`.
8. Formal freeze uses stage-aware validation: template (provisional) → dry_run (provisional|frozen) → formal (frozen + freeze_manifest).
