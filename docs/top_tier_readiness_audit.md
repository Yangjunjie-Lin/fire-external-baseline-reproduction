# Top-tier Paper Readiness Audit (updated)

Current label:

> E-KELL-style paper-faithful pipeline-level reimplementation, not official reproduction.

This audit evaluates structural readiness for a fair, strong, independent baseline comparison. It does **not** claim completed paper results.

## Readiness gates

| Gate | Ready? | Evidence | Blocker |
|---|---|---|---|
| baseline_independence_ready | **true** | No `fire_agent_demo` imports; `tests/test_no_fire_agent_demo_import.py`; gold isolation | Keep CI green |
| interop_ready | **true** | `scripts/run_interop_baselines.py`, `schemas/firebench_interop_v1_prediction.schema.json`, adapter | Need real Runner Bundle from main project for final run |
| faithful_reproduction_ready | **partial → true for Level 3** | `ekell_style_faithful` traces + frozen config | Official E-KELL assets unavailable (do not claim Level 5) |
| strong_baseline_ready | **true (architecture-allowed)** | BM25 strengthened; dense/hybrid interfaces; faithful/enhanced split; no safety injection | Real embeddings still required for formal dense/hybrid rows |
| actual_graphrag_ready | **false** | Adapters set `fallback_only` | Real LightRAG/GraphRAG index+query not implemented |
| real_llm_ready | **partial** | Client supports usage/tokens/guards; example configs | User must supply credentials + shared model version; agents must not auto-call paid APIs |
| final_comparison_ready | **false** | Protocol + frozen configs + interop path exist | Need Runner Bundle, real LLM, shared evaluator, license clearance, expert scores |
| paper_ready | **false** | Scaffold + fairness protocol ready | Experiments + stats + expert eval incomplete |

## Method strength upgrades (non-strawman)

- Direct LLM: structured JSON prompt, parsing status, raw preserved, no target policy injection
- BM25: true BM25, multilingual tokenize, dedupe, no-result handling, evidence IDs
- Dense/Hybrid: real interfaces + RRF; smoke fixture explicitly non-formal
- E-KELL faithful vs enhanced: separate method_ids; enhanced features not smuggled into faithful
- GraphRAG: fallback explicitly excluded from actual leaderboard

## Still required from user

1. Data license review / redistribution decisions
2. Real shared LLM API runs (not heuristic)
3. Main-project Runner Bundle + shared evaluator
4. Expert / manual scoring
5. Optional: real embedding model; actual LightRAG/GraphRAG if claimed
