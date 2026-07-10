# Paper-grade Experiment Protocol

## A. Research question

Does SAFE Fire Agent improve over external KG/RAG/GraphRAG/LLM baseline systems in safety compliance, risk recognition, evidence support, unsafe action blocking, missing confirmation detection, and decision boundary control?

## B. Systems to compare

| Label | method_id | Status |
|---|---|---|
| B0 | `direct_llm` | Strong no-retrieval baseline (main table) |
| B1 | `bm25_rag` | True BM25 lexical RAG (main table) |
| B2 | `ekell_style_controlled_shared_llm` | Complete E-KELL architecture; shared LLM/schema (main table) |
| B2f | `ekell_style_paper_fidelity` | Paper-fidelity track (ChatGLM-6B); separate experiment |
| B3 | `dense_rag` | Supplemental; formal only with real embeddings |
| B4 | `hybrid_rag` | Supplemental; formal only with real dense + frozen RRF |
| B5 | `ekell_style_enhanced` | Supplemental only; must not replace B2/B2f |
| B6 | `lightrag` | Actual only if index+query; else fallback_only |
| B7 | `microsoft_graphrag` | Actual only if index+query; else fallback_only |
| Ours | SAFE Fire Agent | Exported via main project / shared evaluator |

Tuning: DEV only; freeze under `configs/frozen/`. See `docs/baseline_tuning_protocol.md`.

## C. Fairness constraints

All final runs should use:

1. Same scenario matrix.
2. Same fire corpus / evidence source version.
3. Same LLM backend where applicable.
4. Same temperature.
5. Same max tokens.
6. Same prompt visibility: each method receives only the information allowed by its baseline design.
7. Same output schema.
8. Same manual evaluation rubric.
9. No target SAFE modules in external baselines.
10. No target-project risk scoring or final gate in this repository.

## D. Required run records

Record the following for every final run:

- dataset checksum
- corpus checksum
- model provider
- model version
- API family/base URL family without exposing keys
- run date and time zone
- prompt template commit
- repository git commit
- random seed if applicable
- output file checksums
- run manifest path
- config file path and checksum
- whether heuristic mode was used; final paper LLM results must not use heuristic mode

## E. Required final outputs

- `outputs/baseline_outputs.jsonl`
- SAFE normalized outputs JSONL exported outside this repo
- `outputs/side_by_side_comparison.md`
- `outputs/baseline_metrics.csv`
- manual evaluation CSV
- statistical summary CSV/Markdown
- appendix tables
- filled data card
- filled scenario matrix card
- filled model/run cards

## F. Reporting rules

- Report B2 as "E-KELL-style paper-faithful pipeline-level reimplementation, not official reproduction."
- Do not report B3/B4 as actual GraphRAG/LightRAG reproduction unless actual indexing/query integration was executed.
- Automatic metrics must be described as proxy metrics.
- Manual expert scores and statistical tests should be used for paper-level claims.

