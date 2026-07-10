# Reproduction Fidelity Audit

## Reproduction target

The primary reproduction target is the paper-level E-KELL system described in:

> E-KELL: Enhancing Emergency Decision-making with Knowledge Graphs and Large Language Models  
> https://arxiv.org/abs/2311.08732

The target concept is an emergency decision-support pipeline that constructs/uses an emergency knowledge graph, retrieves relevant KG facts, guides a large language model through prompt-chain reasoning, and produces evidence-based emergency decision-support output.

This repository does **not** claim to reproduce official E-KELL results.

## Fidelity levels

| Level | Name | Meaning |
|---|---|---|
| Level 0 | Baseline name only | Uses the paper/system name but does not reproduce pipeline structure. |
| Level 1 | Pipeline-shape reproduction | Recreates the high-level input → KG/RAG → LLM → output shape. |
| Level 2 | Module-level reproduction | Implements explicit modules corresponding to paper concepts. |
| Level 3 | Data-compatible reproduction | Runs the reproduced pipeline on compatible domain KG/evidence/scenario inputs, while documenting data substitution. |
| Level 4 | Metric-compatible reproduction | Reproduces or closely maps the paper's evaluation dimensions with comparable protocol and qualified evaluation. |
| Level 5 | Official code/data/result reproduction | Uses official code, official data, official prompts, and reproduces reported paper results. |

## Current achieved level

**Current label: Level 3 — data-compatible, pipeline-level, paper-faithful reimplementation.**

Method IDs:

- `ekell_style_faithful` — paper-faithful path (no enhanced hooks)
- `ekell_style_enhanced` — separate enhanced path (dense entity / hybrid ranking optional)

This repository implements the E-KELL-style module sequence and runs it on copied fire emergency KG/evidence/scenario inputs. It does not reach Level 4 because the original E-KELL expert evaluation is not reproduced with emergency commanders/firefighters. It does not reach Level 5 because official E-KELL code/data/prompts/results are not integrated.

Traces recorded for faithful runs include: parsed scenario, matched entities + scores, triples, graph paths, evidence chunks, retrieval scores, stage-1/2/3 raw outputs, prompt hashes, context IDs, latency/tokens, fidelity level, and deviations.

## What is reproduced

- Scenario input
- Emergency scenario understanding / parsing
- KG loading from entity, relation, triple, and evidence files
- KG entity matching
- KG subgraph / fact retrieval
- Evidence context construction
- Three-stage prompt-chain reasoning
  - Stage 1: Situation Understanding
  - Stage 2: KG-grounded Decision Reasoning
  - Stage 3: Final Emergency Response
- Evidence-grounded final response
- Unified output schema for comparison with `fire-agent-demo` exports
- Transparent metric/report generation for prototype comparison

## What is not reproduced

- Official E-KELL KG
- Official E-KELL code
- Official E-KELL data preprocessing and KG construction pipeline
- Official exact prompt templates if not public
- Official paper evaluation with emergency commanders / firefighters
- Official exact reported result tables
- Any certified real-world emergency-response behavior

## Deviation table

| Original paper concept | This repo implementation | Reason for deviation | Impact on validity |
|---|---|---|---|
| Emergency KG built from Chinese emergency standards and regulations | Fire emergency KG/evidence copied as input files from a separate project, without importing target code | Official E-KELL KG is not integrated; this project needs fire-domain comparison inputs | Supports system-level fire comparison but not official result reproduction |
| KG triples and schema support emergency reasoning | Robust JSONL loaders for `entities`, `relations`, `triples`, and `evidence_chunks` with field-variant tolerance | Input files may use different schemas | Increases robustness while remaining transparent |
| LLM-guided query decomposition / prompt-chain reasoning over KG | Three explicit prompt stages with JSON schemas | Exact paper prompts are not public/integrated | Preserves pipeline-level behavior, not exact prompt-level reproduction |
| Relevant KG segment supplied to LLM | Entity matching + relation/triple/evidence retrieval with trace | Official retrieval implementation unavailable | Comparable KG-grounded context construction, but retrieval scores may differ |
| E-KELL expert evaluation dimensions | Prototype automatic proxy metrics + manual rubric template | Original human panel is not available in this repo | Automatic metrics are only proxies; manual evaluation is needed for closer comparison |
| Official E-KELL outputs | Unified schema outputs from local pipeline | Official code/data unavailable | Enables fair comparison against SAFE Fire Agent outputs, not official E-KELL claims |

## Final label

**E-KELL-style paper-faithful pipeline-level reimplementation, not official reproduction.**
