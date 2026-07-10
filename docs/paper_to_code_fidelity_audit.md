# Paper-to-Code Fidelity Audit (E-KELL)

Paper: **E-KELL: Enhancing Emergency Decision-making with Knowledge Graphs and Large Language Models**  
URL: https://arxiv.org/abs/2311.08732

Reproduction claim in this repo:

> E-KELL-style paper-faithful pipeline-level reimplementation, **not** official E-KELL reproduction.

**Authoritative matrix:** `docs/ekell_paper_to_code_matrix.md`  
**Spec:** `docs/ekell_reproduction_spec.md`

Tracks:

- Controlled (main table): `ekell_style_controlled_shared_llm`
- Paper fidelity: `ekell_style_paper_fidelity`
- Supplemental: `ekell_style_enhanced` (must not replace either)

## Module mapping (complete pipeline)

| Paper module / concept | Code path | Public basis | Deviations |
|---|---|---|---|
| KG structuring (Sec 3.1) | `ekell_style/kg_construction/` | Top-down schema + extraction | Local fire KG = substituted; auto triples = candidate |
| NL → logical expression (Sec 3.2) | `logical_query/query_decomposer.py` | Constrained AST | Prompt paraphrased |
| FOL p/∧/∨/¬ (Eqs 2–5) | `logical_query/fol_executor.py` | Deterministic executor | Approximated Choudhary-style |
| Vector KG retrieval (Sec 4) | `vector_index.py`, `vector_retriever.py` | text2vec-compatible + smoke | Smoke forbidden under paper_final |
| Neighborhood expansion (Eqs 6–9) | `neighborhood_expander.py` | k-hop + budget | Approximated |
| Stepwise prompt chain (Fig 3, App A) | `stepwise_prompt_chain.py` | Paraphrased templates | Not official verbatim |
| Full orchestration | `full_pipeline.py` | Dual-track outputs | Controlled vs fidelity split |
| Legacy BM25 scaffold | `pipeline.py` | Diagnostics only | Not main-table faithful |

## Complete call graph

```text
ekell_style.full_pipeline.run_* 
  → scenario_parser / query understanding
  → logical_query.query_decomposer + validator
  → vector_retriever (E-KELL-native; not generic dense_rag)
  → neighborhood_expander
  → fol_executor.execute_query
  → stepwise_prompt_chain
  → track-specific payload (paper_fidelity | controlled)
```

## Explicit non-claims

- Official 2264-triple KG / official results / expert 9.xx scores
- Smoke/hash embedding as real vector retrieval
- Enhanced as paper fidelity
- Interface-ready ≡ empirically validated
