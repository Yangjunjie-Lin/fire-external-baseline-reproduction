# E-KELL Paper-to-Code Matrix

Paper: https://arxiv.org/abs/2311.08732 (ar5iv HTML used for section anchors)

| 论文模块 | 论文位置 | 论文要求 | 当前代码 | 缺失/状态 | 实现路径 | 偏差标记 |
|---|---|---|---|---|---|---|
| Top-down emergency KG schema | Sec 3.1; Fig 2 | Schema before data; 8 primary / 22 subclass decision demands | `kg_construction/schema.py` | Fire-domain substitution | schema scaffolding | **substituted** |
| Semi-automatic triple extraction | Sec 3.1; App A KG Construction | LLM extract + manual fusion | `triple_extractor.py` | No forged human review | candidate triples only | **approximated** |
| Manual fusion/refinement | Sec 3.1; Sec 4 (2264 triples) | Expert revise coarse KG | `review_queue.py` | Human review empty | review_status workflow | **unavailable** (official process) |
| Provenance to standards | Sec 3.1; Sec 4 | Link triples to standards text | `provenance.py`, triple fields | Depends on corpus metadata | source_id/chunk_id | **substituted** corpus |
| NL query decomposition | Sec 3.2; Fig 3 | LLM → logical expression | `logical_query/query_decomposer.py` | — | constrained AST | **approximated** prompts |
| Logical expression / FOL | Sec 3.2 Eqs (2)–(5) | p / ∧ / ∨ / ¬ | `fol_executor.py` | — | deterministic executor | **approximated** (Choudhary-style) |
| Prompt-chain generation | Sec 3.2; Fig 3; App A | Stepwise sub-prompts | `stepwise_prompt_chain.py` + `configs/prompts/paper_fidelity/` | Not official verbatim | paraphrased App A | **approximated** |
| KG segment vector retrieval | Sec 4; Fig 4 | LlamaIndex + text2vec | `vector_index.py`, `embedding_backends.py` | Real text2vec optional | smoke vs text2vec | smoke=**smoke**; text2vec=**actual** when installed |
| Neighborhood expansion | Sec 3.2 Eqs (6)–(9); Fig 6 | k-level expansion | `neighborhood_expander.py` | — | k-hop + budget | **approximated** |
| Stepwise KG reasoning | Sec 3.2; Fig 3/6 | Chain over KG segments | `full_pipeline.py` | — | FOL + stepwise LLM | **approximated** |
| Evidence/standard traceability | Sec 4–5 | Cite standards/items | contexts + path provenance | Corpus-dependent | IDs preserved | **substituted** |
| Final decision-support response | Sec 4–5; App A Decision Support | Answer from provided KG only | stepwise final prompt | — | track-specific outputs | controlled vs fidelity split |
| Objective evaluation | Sec 5.2 Table 1 | Grammar/fact/compliance on 10 queries | `evaluation_protocols/ekell_original/` | Cases not official 10 | templates | **unavailable** official queries/results |
| Expert evaluation | Sec 5.2 Table 2 | 14 firefighters + 5 commanders; 4 dims | forms + empty scores | No experts yet | protocol only | **unavailable** |
| ChatGLM-6B system LLM | Sec 4 | Local ChatGLM-6B on A100 | `configs/models/chatglm6b_local.yaml.example` | No empirical run | adapter/config | **unavailable** until user run |
| Official KG size 2264 | Sec 4 | Final expert-refined KG | local counts only | Not same KG | marked substituted | **unavailable** |

Legend: **unavailable** / **assumed** / **approximated** / **substituted** — never presented as official.
