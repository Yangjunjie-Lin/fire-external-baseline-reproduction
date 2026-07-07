# E-KELL Paper Mapping

Paper: **Enhancing Emergency Decision-making with Knowledge Graphs and Large Language Models**  
URL: https://arxiv.org/abs/2311.08732

## Paper-level concept to repository mapping

| E-KELL paper-level concept | Repository implementation |
|---|---|
| Emergency scenario / decision demand | Flattened scenario records from `scenario_matrix_v2.json` |
| Emergency knowledge graph | JSONL assets under `data/corpus/`: entities, relations, triples |
| Standards/regulations/cases as evidence | `evidence_chunks.jsonl` copied as input data |
| Query/scenario understanding | `src/external_baselines/ekell_style/scenario_parser.py` |
| KG entity linking | `src/external_baselines/ekell_style/entity_matcher.py` |
| Relevant KG segment / neighborhood evidence | `src/external_baselines/ekell_style/subgraph_retriever.py` |
| Prompt-chain reasoning | `src/external_baselines/ekell_style/prompt_chain.py` + `configs/prompts/` |
| Final decision support response | Unified schema in `src/external_baselines/common/schema.py` |
| Evaluation dimensions | Proxy metrics + manual rubric mapped to comprehensibility, accuracy, conciseness, instructiveness |

## Key deviation

The project maximizes theoretical/pipeline fidelity from paper-level details, but it does not integrate official E-KELL code/data/prompts/results.
