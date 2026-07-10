# firebench-interop-v1 Integration

This repository consumes the **Runner Bundle** only from the main project’s `firebench-interop-v1` protocol.

## Boundary

| Allowed | Forbidden |
|---|---|
| Runner Bundle (scenarios input-only, corpus/KG snapshot, experiment config, manifests/checksums) | Evaluator Bundle |
| Neutral prediction schema | Gold / expected / labels / annotations |
| Shared model config | Target SAFE-Router / Safety Checker / Dynamic REG / HITL / risk scoring / final gate |

## Command

```bash
python scripts/run_interop_baselines.py \
  --bundle path/to/runner_bundle \
  --methods direct_llm,bm25_rag,dense_rag,hybrid_rag,ekell_style_faithful \
  --config configs/paper_shared_model.yaml \
  --output outputs/firebench_interop_v1_predictions.jsonl
```

## Canonical prediction record

Each JSONL line matches `schemas/firebench_interop_v1_prediction.schema.json`:

- `case_id`, `method_id`
- `prediction` (risks, actions, blocked, missing confirmations, evidence, gate, final_response)
- `runtime` (latency_ms, llm_calls, token_usage, cost)
- `provenance`, `method_metadata`

## Adapter rules

1. `baseline_row_to_interop` only maps fields the baseline actually produced.
2. Normalizer does **not** invent blocked actions / missing confirmations / gates (`infer_structured_safety_fields: false`).
3. `raw_output` is always preserved.
4. Parsing failures are recorded (`parsing_status`), not silently filled with high-score defaults.
5. Baselines do not emit SAFE-Router internal module routing.

## Split workflow

1. `scripts/generate_predictions.py` — gold-isolated generation
2. `scripts/evaluate_predictions.py` — local **proxy** diagnostics only
3. Main-project neutral evaluator — paper scores
4. `scripts/build_report.py` — local report

## Checksums

Interop runs record:

- bundle checksum
- corpus directory manifest aggregate SHA256
- per-record prediction SHA256 under `provenance.record_sha256`
