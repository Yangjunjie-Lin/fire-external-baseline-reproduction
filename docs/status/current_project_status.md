# Current Project Status

## Current phase

```text
unified decision I/O ready for five-method comparison
FireBench taxonomy contract ready
real resources not yet installed
real indexes not yet built
real dry run not yet executed
formal experiment not yet executed
```

Five methods share one Runner Bundle input protocol and independently emit taxonomy-compliant structured decision JSON, natural-language response, and per-method `firebench-interop-v1` prediction JSONL. Native retrieval/reasoning designs are preserved. Formal evaluation remains owned by `fire-agent-demo`.

Structured IDs use the FireBench taxonomy snapshot (`configs/contracts/firebench_taxonomy_v1.json`). Character-level normalization and exact aliases only; free-form natural language stays in human-readable text fields. Unknown/unmapped IDs fail formal validation.

## Valid claim

The repository is **ready to receive real scenarios, corpora, indexes and model resources** for unified decision comparison. **No formal experiment has been started.**

It is **not** paper-ready, **not** empirically validated, and **not** an official E-KELL reproduction.

## Preparation complete (this phase)

| Item | Status |
|---|---|
| Shared real LLM config | prepared (env vars only; gitignored) |
| `main_table` + `comparison_suite` method sets | implemented |
| Unified `DecisionOutput` + strict formal parser | implemented |
| FireBench taxonomy snapshot + exact aliases | `configs/contracts/firebench_taxonomy_v1.json` |
| Taxonomy normalizer (character-level only) | `src/external_baselines/common/taxonomy_normalizer.py` |
| Output taxonomy checker | `scripts/check_output_taxonomy.py` |
| Schema snapshot checker | `scripts/check_firebench_contract_snapshot.py` |
| Per-method decision suite runner | `scripts/run_decision_comparison_suite.py` |
| Dense real text2vec index build/load/query | implemented (fake-model tests only) |
| Hybrid BM25 + Dense + RRF | implemented; reuses Dense index |
| Shared embedding backend factory | `src/external_baselines/retrieval/embedding_backends.py` |
| Comparison readiness checker | `scripts/check_comparison_readiness.py` |
| Index build entry (`--validate-only`) | `scripts/build_comparison_indexes.py` |
| Stage-aware formal validator | template / dry_run / formal |
| Freeze manifest helper | `scripts/create_freeze_manifest.py` |
| Staged execution plan | `docs/experiments/staged_execution_plan.md` |

## Still pending (deferred)

- Main-project v1 Runner Bundle + scenarios/corpus
- Embedding model revision mount / download
- Real Dense + E-KELL index builds
- 1–3 case dry run with shared SiliconFlow LLM
- DEV parameter selection + human freeze
- Formal TEST + main-project evaluator scoring

## Method layers

| Layer | Methods |
|---|---|
| Formal main table | `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm` |
| Comparison suite | main table + `dense_rag` + `hybrid_rag` |
| Paper-fidelity (separate) | `ekell_style_paper_fidelity` |
| Supplemental ablation | `ekell_style_enhanced` |
| Fallback / legacy | `lightrag`, `microsoft_graphrag`, `fallback_graph_retrieval`, `ekell_style_legacy_bm25` |

Dense/Hybrid never modify E-KELL controlled paper structure (`dense_entity_retrieval` / hybrid subgraph / reranker / self-consistency / structured verification remain false).

## Authority split

| Authority | Owner |
|---|---|
| Scenarios / gold / evaluator | main project (`fire-agent-demo`) |
| External baselines / predictions | this repository |
| Formal freeze | human after DEV evidence |
