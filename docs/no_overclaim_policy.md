# No-overclaim Policy

## Accepted phrases

- E-KELL-style paper-faithful pipeline-level reimplementation, not official E-KELL reproduction
- external baseline reproduction package
- independent strong baseline (architecture-allowed best effort)
- system-level comparison baseline (`firebench-interop-v1`)
- proxy automatic / diagnostic metrics
- manual/expert rubric required for paper-level correctness claims
- faithful vs enhanced reported as separate `method_id`s
- fallback GraphRAG / smoke dense fixture (explicitly non-actual)

## Forbidden phrases

Do not use these unless conditions are explicitly satisfied:

- official E-KELL reproduction
- reproduces E-KELL results
- certified emergency response / proves safety
- top-tier ready **results** (scaffold ≠ completed experiments)
- complete / actual GraphRAG or LightRAG reproduction, unless package installed **and** indexing **and** query completed with version/checksum recorded
- final paper results, unless real LLM runs + shared evaluator + expert scores + stats are done
- reporting `ekell_style_enhanced` as faithful reproduction
- reporting heuristic smoke as paper-final

## Required caveats

- Faithful/complete and enhanced must appear as different method rows.
- Paper fidelity and controlled comparison must not be merged as one reproduction result.
- Fallback must never enter the actual GraphRAG leaderboard.
- Proxy metrics must not replace the shared paper evaluator or expert correctness.
- Baselines must not import or emulate target SAFE-Router / Safety Checker / Dynamic REG / HITL / risk scoring / final gate.
- Do not claim `embedding_scorer=None` is required for fidelity; E-KELL-native vector retrieval is required, generic dense_rag is forbidden.
- Public corpus redistribution requires completed license audit (`docs/data_license_audit.md`).
- `real_world_execution_allowed=false` means system capability only; it is not a clearance that model language is safe.
