# Baseline Method Cards

## Taxonomy

| method_id | Class | Formal Track A | Notes |
|---|---|---|---|
| `direct_llm` | baseline | Yes (A1) | No retrieval |
| `bm25_rag` (`vanilla_rag` alias) | baseline | Yes (A2) | True BM25 + multilingual tokenize |
| `dense_rag` | enhanced / smoke | Only with real embeddings | Smoke hash fixture ≠ formal dense |
| `hybrid_rag` | enhanced / smoke | Only with real dense | RRF fusion; component scores recorded |
| `ekell_style_faithful` | faithful | Yes (A2; A3 only if architecture supports) | **Not** official E-KELL |
| `ekell_style_enhanced` | enhanced | Yes (separate row) | Dense entity / hybrid ranking optional |
| `lightrag` | actual **or** fallback | Actual only if indexing+query | Currently fallback_only |
| `microsoft_graphrag` | actual **or** fallback | Actual only if indexing+query | Currently fallback_only |
| `fallback_graph_retrieval` | fallback | No actual GraphRAG board | Explicit fallback |

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
