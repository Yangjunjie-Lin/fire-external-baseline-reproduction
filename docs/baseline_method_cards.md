# Baseline Method Cards

## Taxonomy

| method_id | Class | Paper table | Formal Track A |
|---|---|---|---|
| `direct_llm` | baseline | **main** | Yes (A1) |
| `bm25_rag` | baseline | **main** | Yes (A2) |
| `ekell_style_faithful` | faithful | **main** | Yes (A2) |
| `dense_rag` | enhanced / smoke | supplemental | Only with real embeddings |
| `hybrid_rag` | enhanced / smoke | supplemental | Only with real dense |
| `ekell_style_enhanced` | enhanced | supplemental | Separate row; never replaces faithful |
| `lightrag` / `microsoft_graphrag` | actual or fallback | not main table | Actual only if indexing+query |
| `fallback_graph_retrieval` | fallback | no | Never actual GraphRAG board |

## Shared claims

- E-KELL-style = **paper-faithful pipeline-level reimplementation, not official reproduction**.
- Heuristic LLM = smoke only.
- Proxy metrics ≠ expert correctness.
- Target modules (SAFE-Router, Safety Checker, Dynamic REG, HITL, risk scoring, final gate) = **No** for all baselines.

## Per-method cards (summary)

### direct_llm
- Source: standard no-retrieval LLM baseline
- Inputs: scenario text only
- Prompt: structured JSON; raw preserved
- Deviations: none from architecture

### bm25_rag
- Retrieval: deterministic BM25, duplicate suppression, CJK+Latin tokenize
- Inputs: scenario + corpus evidence chunks

### dense_rag
- Requires real embedding model/version + index checksum for formal claims
- Current default backend: `smoke_hash_embedding` → `method_status=smoke_fixture_only`

### hybrid_rag
- BM25 + dense RRF; weights frozen from DEV
- Component scores in context metadata

### ekell_style_faithful
- Pipeline: Scenario → Parse → Entity Match → Subgraph → Evidence → Prompt Chain → Normalize
- Traces: entities, triples, graph paths, stage raw outputs, prompt hashes, context IDs
- No enhanced hooks

### ekell_style_enhanced
- Same pipeline + optional dense entity scoring / hybrid subgraph ranking
- Must be reported separately from faithful

### lightrag / microsoft_graphrag
- Actual only when: package installed, index built, query run, version+checksum recorded, `actual_external_package_used=true`, `fallback_retrieval_used=false`
- Otherwise: `method_status=fallback_only` — excluded from actual GraphRAG leaderboard
