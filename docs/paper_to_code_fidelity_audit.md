# Paper-to-Code Fidelity Audit (E-KELL)

Paper: **E-KELL: Enhancing Emergency Decision-making with Knowledge Graphs and Large Language Models**  
URL: https://arxiv.org/abs/2311.08732

Reproduction claim in this repo:

> E-KELL-style paper-faithful pipeline-level reimplementation, **not** official E-KELL reproduction.

Main-table method: `ekell_style_faithful` only.  
Supplemental (must not replace faithful): `ekell_style_enhanced`.

## Module mapping

| Paper module / concept | Code path | Key function(s) | Public basis | Assumptions | Deviations |
|---|---|---|---|---|---|
| Emergency scenario input | `src/external_baselines/common/io.py` | `flatten_scenario`, `to_prediction_input` | Paper uses emergency scenarios as LLM+KG inputs | Input-only fields; gold stripped at generation | Uses fire-domain scenario matrix, not original E-KELL scenario set |
| Situation / scenario understanding | `src/external_baselines/ekell_style/scenario_parser.py` | `parse_scenario`, `deterministic_parse` | Paper describes scenario understanding before KG retrieval | Optional LLM parse vs deterministic parse | Exact paper parser prompts/schema not public → local schema |
| Entity matching / linking | `src/external_baselines/ekell_style/entity_matcher.py` | `match_entities` | Paper links scenario mentions to KG entities | Exact/fuzzy/alias + bilingual lexicon | No official E-KELL linker; local scoring |
| Emergency KG | `src/external_baselines/ekell_style/kg_loader.py` | `load_kg`, `audit_corpus` | Paper uses emergency KG | JSONL entities/relations/triples/evidence | **Substituted** fire corpus/KG snapshot, not official E-KELL KG |
| Subgraph / fact retrieval | `src/external_baselines/ekell_style/subgraph_retriever.py` | `retrieve_subgraph` | Paper retrieves relevant KG neighborhood/facts | Entity-touch + BM25 ranking; 1-hop path serialization | Multi-hop depth/ranking details not fully public → transparent local heuristic |
| Evidence context construction | `ekell_style/pipeline.py` + `prompt_chain._context_block` | `_dedupe_contexts`, `_context_block` | Paper grounds LLM on retrieved KG/evidence | Score-ordered contexts; char budget | Context serialization format is local |
| Prompt-chain reasoning | `src/external_baselines/ekell_style/prompt_chain.py` + `configs/prompts/ekell_stage{1,2,3}_*.txt` | `run_prompt_chain` | Paper uses multi-stage prompt chaining | 3 stages: understanding → KG-grounded reasoning → final JSON | Exact official prompts unavailable → documented local templates |
| Final decision-support response | `ekell_style/pipeline.py` + `common/schema.py` | `normalize_response_payload` | Paper outputs actionable emergency support text/structure | Maps to unified/interop schema | Schema is firebench-interop / baseline unified, not official E-KELL export |
| Evaluation | proxy metrics + manual rubric; shared evaluator external | `evaluation/metrics.py` (proxy only) | Paper uses expert evaluation dimensions | Proxy ≠ expert correctness | Level 4/5 not claimed |

## Faithful path call graph

```text
ekell_style.pipeline.run_scenario
  → kg_loader.load_kg
  → scenario_parser.parse_scenario
  → entity_matcher.match_entities   (embedding_scorer=None)
  → subgraph_retriever.retrieve_subgraph
  → prompt_chain.run_prompt_chain
  → schema.normalize_response_payload
  → normalizer.maybe_infer_structured_safety_fields  (default OFF)
```

## Explicit non-claims

- Not official code/data/prompt/result reproduction (Level 5).
- Not expert-commander evaluation reproduction (Level 4).
- `ekell_style_enhanced` is supplemental and may use dense entity scoring; **forbidden** in faithful path.

## Evidence artifacts

- Stage traces / prompt hashes / entity scores / triples / graph paths in `method_specific`
- Dependency audit: `docs/dependency_audit_ekell_faithful.md` + `tests/test_ekell_faithful_dependency_audit.py`
