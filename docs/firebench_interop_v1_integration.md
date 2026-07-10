# firebench-interop-v1 Integration

This repository consumes the **Runner Bundle** only from the main project’s `firebench-interop-v1` protocol (`fire-agent-demo @ evaluation/benchmark-v1`).

## Authority split

| Authority | Owner |
|---|---|
| Prediction generation | this external baseline repo |
| Benchmark / evaluation scoring | `fire-agent-demo` shared evaluator |

Local proxy metrics are **diagnostic only**.

## Formal input

Prefer `manifest.files.input_cases` → `input_cases.jsonl` with nested `input.scenario`.

Also preserved: `language`, `input_mode`, `context`, `dynamic_snapshots` (methods that do not consume dynamic state must set `dynamic_state_consumed=false`).

Checksums: validate **per-file** `manifest.checksums`. Aggregate `consumer_computed_bundle_hash` is diagnostic and must not be confused with a producer-declared aggregate checksum.

## Formal output

Root `schema_version: firebench-interop-v1`. Track A prediction fields match the main-project schema:

- `blocked_actions`: string ID array
- `missing_confirmations`: string ID array
- `final_decision_gate`: `allow_response|await_human_confirmation|block_response|unknown`
- `final_response.status`: `provided|awaiting_human_confirmation|blocked|not_applicable|unknown`
- `real_world_execution_allowed`: always `false`
- `evidence_refs`: objects with `evidence_id`

Extended diagnostics (`retrieved_evidence`, `parsing_status`, `raw_output`, authorization fields) live under `method_metadata`.

## Formal command

```bash
python scripts/run_interop_baselines.py \
  --experiment-manifest configs/experiments/paper_main_table_v1.yaml \
  --bundle path/to/formal_runner_bundle \
  --output outputs/interop/test_public/canonical/predictions.jsonl \
  --manifest outputs/interop/test_public/manifests/run_manifest.json
```

Heuristic cross-repo smoke (no paid API):

```bash
python scripts/smoke_main_runner_bundle.py
```

## Main table vs supplemental

- Main table (controlled): `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm`
- Paper fidelity (separate): `ekell_style_paper_fidelity`
- Supplemental: `dense_rag`, `hybrid_rag`, `ekell_style_enhanced` (smoke dense ≠ formal)
- Fallback-only: `lightrag`, `microsoft_graphrag`, `fallback_graph_retrieval`

`cross_repository_interop_verified` stays **false** until formal shared-LLM generation + main-project evaluator confirmation.
