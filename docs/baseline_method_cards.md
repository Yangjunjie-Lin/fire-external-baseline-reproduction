# Baseline Method Cards

> **Canonical taxonomy:** [`docs/methods/method_registry.md`](methods/method_registry.md)  
> **Status:** [`docs/status/current_project_status.md`](status/current_project_status.md)  
> If this file disagrees with the registry, the registry wins.

## Taxonomy

| method_id | Class | Paper table | Notes |
|---|---|---|---|
| `direct_llm` | canonical baseline | **main** | No retrieval |
| `bm25_rag` | canonical baseline | **main** | Alias: `vanilla_rag` (CLI only) |
| `ekell_style_controlled_shared_llm` | paper-style reimplementation | **main** | Alias: `ekell_style_faithful` → controlled |
| `ekell_style_paper_fidelity` | paper-fidelity track | separate | ChatGLM-6B interface; not main table |
| `dense_rag` / `hybrid_rag` | supplemental | supplemental | Formal only with real dense |
| `ekell_style_enhanced` | supplemental | supplemental | Never replaces controlled/fidelity |
| `ekell_style_legacy_bm25` | legacy diagnostic | no | Not aliased to controlled |
| `lightrag` / `microsoft_graphrag` | adapter | fallback_only | Until actual index+query |
| `fallback_graph_retrieval` | fallback | no | Never actual GraphRAG board |

## Shared claims

- E-KELL-style = **pipeline-level reimplementation, not official reproduction**.
- Heuristic LLM = smoke only.
- Proxy metrics ≠ shared paper evaluator / expert correctness.
- Target modules (SAFE-Router, Safety Checker, Dynamic REG, HITL, risk scoring, final gate) = **No** for all baselines.

## Per-method cards (summary)

### direct_llm
- Source: standard no-retrieval LLM baseline
- Inputs: scenario text only
- Prompt: structured JSON; raw preserved
- Deviations: none from architecture

### bm25_rag
- Physical package: `external_baselines.vanilla_rag` (compatibility path)
- Retrieval: deterministic BM25, duplicate suppression, CJK+Latin tokenize
- Inputs: scenario + corpus evidence chunks

### dense_rag
- Requires real embedding model/version + index checksum for formal claims
- Current default backend: `smoke_hash_embedding` → `method_status=smoke_fixture_only`

### hybrid_rag
- BM25 + dense RRF; weights frozen from DEV
- Formal only when dense side is real embeddings

### ekell_style_controlled_shared_llm
- Full pipeline: FOL + vector retrieval + neighborhood + stepwise prompts
- Shared LLM / schema with other main-table methods
- Label: E-KELL-style paper-faithful pipeline-level reimplementation, not official

### ekell_style_paper_fidelity
- Same architecture; ChatGLM-6B / paper-fidelity config track
- Separate experiment from controlled main table

### ekell_style_enhanced
- Supplemental hooks only; must not replace controlled or paper-fidelity

### ekell_style_legacy_bm25
- Old BM25 + 3-stage scaffold; diagnostics / regression only

### lightrag / microsoft_graphrag
- Official-system adapters; currently fallback_only unless actual package indexing + query configured
