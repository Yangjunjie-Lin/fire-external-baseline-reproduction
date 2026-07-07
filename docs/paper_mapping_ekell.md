# E-KELL paper mapping

Mapped paper:

- Minze Chen, Zhenxiang Tao, Weitong Tang, Tingxin Qin, Rui Yang, Chunli Zhu. *Enhancing Emergency Decision-making with Knowledge Graphs and Large Language Models*. arXiv:2311.08732.

## Paper core idea

E-KELL constructs a structured emergency knowledge graph and guides an LLM to reason over it through a prompt chain for evidence-based emergency decision-making.

## Reimplementation mapping in this repository

| Paper-level idea | Repository module | Notes |
|---|---|---|
| Emergency scenario input | `scripts/run_baseline.py`, `common/io.py` | Scenario matrix is flattened into `scenario_text`. |
| Emergency KG | `data/corpus/entities.jsonl`, `relations.jsonl`, `triples.jsonl` | Copied fire corpus/KG files are used as input. |
| Scenario understanding | `ekell_style/scenario_parser.py` | Deterministic parser by default; optional LLM parser interface. |
| Entity matching | `ekell_style/entity_matcher.py` | Exact, normalized, and keyword-overlap matching. |
| KG subgraph/fact retrieval | `ekell_style/subgraph_retriever.py` | Neighbor-like matching over triples/relations plus linked evidence chunks. |
| Prompt-chain reasoning | `ekell_style/prompt_chain.py` | Three stages: situation understanding, KG-grounded reasoning, final response. |
| Evidence-based decision output | `ekell_style/pipeline.py` | Normalized to unified comparison schema. |

## Deviations

This is **not** an official reproduction unless official E-KELL code is integrated later.

Current deviations:

1. Uses copied fire-agent-demo corpus/KG input files instead of the original E-KELL KG.
2. Uses a local entity matcher and subgraph retriever because official E-KELL source code is not vendored.
3. Uses deterministic heuristic fallback when no LLM provider is configured.
4. Outputs a comparison schema rather than the exact paper UI/output format.
5. Safety-related fields are inferred from text for comparability rather than produced by a target-project Safety Checker.

## Non-improvement rule

Do not add SAFE-Router-like routing, Safety Checker logic, Dynamic REG state, HITL gate, or target-project risk scoring to this baseline.
