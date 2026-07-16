# DeepEval Handoff Readiness

Status: implementation-ready for centralized evaluation handoff. No evaluation has been run.

## Implemented

- `fireagent-external-prediction-v1` snapshot and exact provenance.
- Read-only FireBench interop adapter with fail-closed actual output handling.
- Ordered retrieval context preservation through unified serialization.
- Direct LLM output-only semantics and RAG context-required semantics.
- Frozen top-k original-prefix submission with transparent truncation metadata.
- Formal source, coverage, safety, Gold-isolation, Schema, and hash validation.
- Transactional export, deterministic packaging, offline fixtures, and CI smoke.

## Authority boundary

- DeepEval is not installed.
- DeepEval is not executed.
- Gold is not read.
- Judge calls and paid evaluation APIs are not used.
- Completed prediction artifacts are the only exported evaluation input.
- `fire-agent-demo` remains the sole evaluator and leaderboard authority.

## Readiness interpretation

A valid handoff means only that prediction artifacts satisfy the frozen exchange contract. It does not mean DeepEval validated a baseline, a score passed, accuracy was externally validated, or the system is safe for real-world execution.

Formal publication eligibility is decided from immutable Formal-run evidence. Development exports remain visibly ineligible even when their handoff structure validates.
