# E-KELL Logical Reasoning (FOL)

Paper: Sec 3.2; Eqs (2)–(5); Fig 3.

## Constrained AST

LLM may only emit validated operations:

- `projection` / `p`
- `intersection` / `i`
- `union` / `u`
- `negation` / `n`

No arbitrary code execution. Invalid plans are marked `degraded` with fallback reason.

## Code

| Piece | Path |
|---|---|
| Schema / AST | `ekell_style/logical_query/schema.py` |
| Decomposer | `query_decomposer.py` |
| Parser / validator | `parser.py`, `validator.py` |
| Executor | `fol_executor.py` |
| Trace | `trace.py` |
| Stepwise prompts | `stepwise_prompt_chain.py` + `configs/prompts/paper_fidelity/` |

## Negation

Negation is relative to an explicit `candidate_universe` (seed entities + neighborhood), not an unbounded world.

## Tests

`tests/test_fol_executor.py`, `tests/test_stepwise_prompt_chain.py`
