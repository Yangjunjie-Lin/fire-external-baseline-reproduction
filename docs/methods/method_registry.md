# Method Registry

**Single source of truth:** `src/external_baselines/method_registry.py`

All of the following must derive from that module (not maintain parallel lists):

- `runner.get_pipeline` / `resolve_pipeline`
- experiment manifest main/supplemental/paper-fidelity sets
- interop `METHOD_ID_ALIASES` (via `method_id_aliases()`)
- eligibility / claim labels
- fidelity matrix method IDs

## Canonical IDs and aliases

| Canonical method_id | Aliases (CLI only) | Layer |
|---|---|---|
| `direct_llm` | — | formal main table |
| `bm25_rag` | `vanilla_rag` | formal main table |
| `ekell_style_controlled_shared_llm` | `ekell`, `ekell_style`, `e-kell-style`, `ekell_style_faithful` | formal main table |
| `ekell_style_paper_fidelity` | — | paper-fidelity track |
| `dense_rag` | — | supplemental |
| `hybrid_rag` | — | supplemental |
| `ekell_style_enhanced` | — | supplemental |
| `ekell_style_legacy_bm25` | — | legacy diagnostic |
| `lightrag` | — | fallback_only |
| `microsoft_graphrag` | `graphrag` | fallback_only |
| `fallback_graph_retrieval` | — | fallback_only |

## Naming notes

- `ekell_style_faithful` is an **alias of controlled**, not of legacy.
- Legacy entrypoint is `run_legacy_bm25` (deprecated alias: `run_scenario_faithful`).
- Physical package for BM25 remains `vanilla_rag/` for import stability; canonical ID is `bm25_rag`.
