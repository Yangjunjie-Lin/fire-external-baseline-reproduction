# Baseline Strengths and Limitations

## Direct LLM

Strength: Simple no-retrieval baseline showing what the model can infer from scenario text alone.

Limitation: No external evidence, KG, citations, or retrieval grounding.

Expected failure mode: Hallucinated actions, missed domain-specific restrictions, weak citation/evidence support.

## Vanilla RAG

Strength: Uses text evidence retrieval and can cite retrieved chunks.

Limitation: No explicit KG structure or multi-hop entity/relation reasoning.

Expected failure mode: Retrieves lexically similar but incomplete evidence; may miss relation-level constraints.

## E-KELL-style KG + LLM Prompt Chain

Strength: Closest implemented external baseline to the E-KELL paper-level idea: KG/evidence retrieval plus staged LLM reasoning.

Limitation: Not official E-KELL code/data/prompts/results; uses independently implemented matching/retrieval and copied fire-domain inputs.

Expected failure mode: Entity-linking errors, subgraph retrieval misses, prompt-chain propagation errors, weak behavior under sparse KG.

Fidelity level: Level 3 data-compatible pipeline-level reimplementation when KG/evidence/scenario inputs are present.

## LightRAG adapter

Current status: Transparent adapter stub with fallback graph/text retrieval unless actual LightRAG indexing/query is implemented.

Needed for actual reproduction:

- Install compatible LightRAG version.
- Build LightRAG index from the same corpus.
- Run official LightRAG query path.
- Record index config, package version, and output traces.

## Microsoft GraphRAG adapter

Current status: Transparent adapter stub with fallback graph/text retrieval unless actual Microsoft GraphRAG workspace/index/query is implemented.

Needed for actual reproduction:

- Install compatible Microsoft GraphRAG version.
- Create official workspace/index from the same corpus.
- Run official query workflow.
- Record config, package version, index artifacts, and output traces.

## SAFE Fire Agent target comparison

SAFE Fire Agent outputs must be exported separately from `fire-agent-demo` into the normalized schema. This repository only consumes exported files and does not import target code.

