# Reproduction notes

This repository is an independent external baseline reproduction scaffold. It is intended to compare external KG/RAG/GraphRAG/LLM-style baselines against the separate `fire-agent-demo` SAFE Fire Agent prototype.

## Scope

First milestone:

- B0 `direct_llm`
- B1 `vanilla_rag`
- B2 `ekell_style`
- B3 optional `lightrag` / `microsoft_graphrag` adapters with fallback

## Independence rule

The repository may copy data files from `fire-agent-demo`, but it must not import target-project code.

The following target modules are intentionally absent:

- SAFE-Router
- Safety Checker
- Dynamic REG
- HITL Gate
- internal target risk scoring
- internal target ablation/evaluation code

## Local deterministic LLM fallback

A deterministic heuristic LLM client is included only to keep the runner executable without API keys. This is a reproducibility aid, not a method improvement. For real LLM runs, configure `configs/llm.yaml` and pass it with `--config configs/llm.yaml`.

## Structured safety fields

Most external baselines do not natively expose fields such as `blocked_or_unsafe_actions` or `missing_confirmations`. For schema compatibility, this project can infer those fields from baseline text. This is documented in `method_specific.structured_safety_fields = inferred_from_text`.
