# Reproduction Notes

## Current status

This repository implements an independent, pipeline-level, paper-faithful E-KELL-style KG + LLM prompt-chain baseline for fire emergency decision-support comparison.

It is **not** an official E-KELL reproduction and does not claim to reproduce official E-KELL results.

## Main reproduced pipeline

Scenario Input
→ Scenario Understanding / Parsing
→ KG Entity Matching
→ KG Subgraph / Fact Retrieval
→ Evidence Context Construction
→ Prompt Chain Reasoning
→ Final Emergency Decision Support Output
→ Unified Output Normalization

## E-KELL-style mapping

The E-KELL paper describes a KG-enhanced LLM emergency decision-support system that structures emergency knowledge into a KG and guides LLM reasoning through a prompt chain. This repository maps that concept into:

- deterministic or LLM JSON scenario parser
- robust KG/evidence loaders
- transparent entity matching
- subgraph/fact/evidence retrieval
- three prompt stages stored in `configs/prompts/`
- normalized output schema

## Independence constraints

This repository must not import or call:

- SAFE-Router
- Safety Checker
- Dynamic REG
- HITL Gate
- target-project risk scoring
- target-project final gate logic
- any other `fire-agent-demo` internal module

Copied input data files are allowed; copied code is not.

## Heuristic mode warning

The default heuristic LLM mode is only for smoke tests and CI-style reproducibility. Real comparison should use a recorded LLM provider/model/temperature/date, for example through an OpenAI-compatible endpoint config.

## GraphRAG / LightRAG status

LightRAG and Microsoft GraphRAG adapters are optional and transparent. Unless the actual external package, index construction, and query integration are configured, their outputs are marked as fallback retrieval outputs and should not be claimed as full official reproductions.
