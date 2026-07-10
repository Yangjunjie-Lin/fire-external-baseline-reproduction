# Current Project Status

## Current phase

```text
five-method comparison implementation ready
real resources not yet installed
real indexes not yet built
real dry run not yet executed
formal experiment not yet executed
```

Five-method controlled comparison code and resource interfaces are complete. Real indexes, paid API dry runs, DEV freeze, and formal TEST remain deferred until main-project Runner Bundle + embedding model revision are installed.

Manual `status.main_project_v1_ready: true` is an approval signal only and cannot bypass branch/bundle/schema/checksum validation.

Formal model identity is frozen in YAML. Environment variables provide credentials and endpoints only.

## Valid claim

The repository is **five-method controlled comparison code complete** with stage-aware freeze validation. **No formal experiment has been started.**

It is **not** paper-ready, **not** empirically validated, and **not** an official E-KELL reproduction.

## Preparation complete (this phase)

| Item | Status |
|---|---|
| Shared real LLM config | prepared (env vars only; gitignored) |
| `main_table` + `comparison_suite` method sets | implemented |
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
- 1–3 case dry run
- DEV parameter selection + human freeze
- Formal TEST + main-project evaluator

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
