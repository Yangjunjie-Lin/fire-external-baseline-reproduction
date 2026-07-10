# firebench-interop-v1 Integration

This repository consumes the **Runner Bundle** only from the main project’s `firebench-interop-v1` protocol.

## Boundary

| Allowed | Forbidden |
|---|---|
| Runner Bundle (scenarios input-only, corpus/KG snapshot, experiment config, manifests/checksums) | Evaluator Bundle |
| Neutral prediction schema | Gold / expected / labels / annotations |
| Shared model config (SiliconFlow-aligned) | Target SAFE-Router / Safety Checker / Dynamic REG / HITL / risk scoring |

## Formal command (single experiment manifest)

```bash
python scripts/run_interop_baselines.py \
  --experiment-manifest configs/experiments/paper_main_table_v1.yaml \
  --bundle path/to/formal_runner_bundle \
  --output outputs/firebench_interop_v1_predictions.jsonl
```

Merge order per method: `base_config` → `shared_model_config` → method `config`.

Multiple `--config` CLI overlays are **rejected** (ambiguous for paper runs).

## Main table vs supplemental

- Main table (controlled): `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm`
- Paper fidelity (separate experiment): `ekell_style_paper_fidelity`
- Supplemental (`--include-supplemental`): `dense_rag`, `hybrid_rag`, `ekell_style_enhanced`

Integrity details: `docs/interop_integrity_audit.md`. Schema draft proposal: `schemas/firebench_interop_v1_1_draft_prediction.schema.json`.

## Adapter rules

1. Maps only fields the baseline actually produced.
2. No invented blocked/missing/gate fields (`infer_structured_safety_fields: false`).
3. `raw_output` preserved; parsing failures recorded.
4. Evidence text is never promoted to evidence IDs; global evidence is not auto-bound to every action.
5. `system_execution_capability=false`; `output_authorization_status` / `real_world_execution_violation` come from baseline language (never auto-cleared to safe).
6. `real_world_execution_allowed=false` is v1 capability compatibility only — not a safety clearance.
7. Predictions validated with jsonschema Draft 2020-12 against local or bundle schema.
8. `cross_repository_interop_verified` stays **false** until a formal main-project Runner Bundle is actually consumed and hashes verified.

## Pending cross-repo verification

After formal bundle arrives, verify:

1. schema hash  
2. scenario hash  
3. corpus hash  
4. input-only / gold isolation  
5. baseline predictions  
6. neutral evaluator compatibility  
