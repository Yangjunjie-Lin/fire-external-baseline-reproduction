# Paper-grade Experiment Protocol

## A. Research question

Does SAFE Fire Agent improve over external KG/RAG/GraphRAG/LLM baseline systems in safety compliance, risk recognition, evidence support, unsafe action blocking, missing confirmation detection, and decision boundary control?

## B. Systems to compare

| Label | System | Status |
|---|---|---|
| B0 | Direct LLM | Implemented baseline |
| B1 | Vanilla RAG | Implemented lexical RAG baseline |
| B2 | E-KELL-style KG + LLM Prompt Chain | Implemented pipeline-level reimplementation, not official reproduction |
| B3 | LightRAG actual indexing/query | Only claim when actual external package indexing and query integration are implemented |
| B4 | Microsoft GraphRAG actual indexing/query | Only claim when actual external package workspace/index/query integration is implemented |
| Ours | SAFE Fire Agent | Must be exported separately from `fire-agent-demo` as normalized JSONL |

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

