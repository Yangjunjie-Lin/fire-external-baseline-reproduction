# E-KELL Paper Mapping

Paper: **Enhancing Emergency Decision-making with Knowledge Graphs and Large Language Models**  
URL: https://arxiv.org/abs/2311.08732

## Paper-level concept to repository mapping

See the detailed audit: [`docs/paper_to_code_fidelity_audit.md`](paper_to_code_fidelity_audit.md).

| E-KELL paper-level concept | Repository implementation |
|---|---|
| Emergency scenario / decision demand | Flattened scenario records; gold stripped via `to_prediction_input` |
| Emergency knowledge graph | JSONL assets under `data/corpus/` (substituted fire KG) |
| Standards/regulations/cases as evidence | `evidence_chunks.jsonl` |
| Query/scenario understanding | `ekell_style/scenario_parser.py` |
| KG entity linking | `ekell_style/entity_matcher.py` |
| Relevant KG segment / neighborhood evidence | `ekell_style/subgraph_retriever.py` |
| Prompt-chain reasoning | `ekell_style/prompt_chain.py` + `configs/prompts/` |
| Final decision support response | `ekell_style/pipeline.py` (`ekell_style_faithful`) |
| Enhanced (non-faithful) extensions | `ekell_style/enhanced_pipeline.py` (**supplemental only**) |

Main paper table method for E-KELL-style: **`ekell_style_faithful` only**.

## Key deviation

The project maximizes theoretical/pipeline fidelity from paper-level details, but it does not integrate official E-KELL code/data/prompts/results.
