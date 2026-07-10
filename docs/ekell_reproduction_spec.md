# E-KELL Reproduction Spec (pipeline-level, not official)

Paper: E-KELL (arXiv:2311.08732)
Claim: **E-KELL-style paper-faithful pipeline-level reimplementation, not official reproduction.**

## Tracks

| Track | method_id | Purpose |
|---|---|---|
| Paper Fidelity | `ekell_style_paper_fidelity` | Closest architecture + ChatGLM-6B interface |
| Controlled Comparison | `ekell_style_controlled_shared_llm` | Same architecture; shared SiliconFlow LLM/data/schema with main project |
| Enhanced (supplemental) | `ekell_style_enhanced` | Must not replace either track above |

## Complete pipeline

```text
Scenario
→ Query Understanding
→ Logical Query Decomposition
→ AST Validation
→ Vector KG Retrieval
→ Neighborhood Expansion
→ FOL Execution (p / ∧ / ∨ / ¬)
→ Stepwise Prompt Chain
→ Evidence-grounded Final Response
→ Trace/Provenance Export
```

## Module status

| Module | Status | Notes |
|---|---|---|
| KG structuring (Sec 3.1) | scaffolding ready | Auto triples = `candidate`; no forged human review; local KG = `substituted_fire_domain_kg` |
| FOL ops (Sec 3.2 Eqs 2–5) | implemented + tested | Constrained AST only |
| Neighborhood expansion (Eqs 6–9) | implemented + tested | k-hop, budget, provenance |
| Vector retrieval (Sec 4, text2vec/LlamaIndex) | interface + smoke + text2vec adapter | Formal needs actual embedding; smoke forbidden under paper_final |
| Stepwise prompt chain (Fig 3, App A) | implemented | Paraphrased templates; not official verbatim dump |
| ChatGLM-6B | config/adapter only | `paper_fidelity_model_run=false` until user server run |
| Official 2264-triple KG | unavailable | Do not claim ownership |
| Expert eval (14 FF + 5 commanders) | protocol templates only | Scores empty; no fabricated 9.xx |

## Output formats

- **paper_fidelity**: decision support + KG facts + standards + reasoning trace (no forced SAFE gates)
- **controlled**: shared FireBench outcome schema (`controlled_output_format=true`)

## Forbidden overclaims

- Official E-KELL results / expert scores
- Official KG identity
- Smoke/hash embedding as real vector retrieval
- Enhanced as paper fidelity
- Interface-ready ≡ empirically validated
