# Interop Integrity Audit (firebench-interop-v1 / v1.1-draft)

## Implemented

| Check | Status | Location |
|---|---|---|
| Recomputed bundle checksum | done | `interop/bundle.py` `recompute_bundle_checksum` / `validate_bundle_checksum` |
| Gold hard-fail (no strip) | done | forbidden keys + filenames raise `PermissionError` |
| Empty path rejected | done | `assert_path_inside_bundle` |
| Evidence text ≠ evidence ID | done | `interop/schema.py` `_evidence_mapping` |
| Global evidence not auto-bound to all actions | done | action `evidence_refs` only from action-specific fields |
| Full final response preserved | done | explicit text → else deterministic multi-field render |
| Per-case token/call accounting | done | `UsageTrackingLLMClient.usage_snapshot` + runner delta |
| Cross-method fairness | done | `common/fairness.py` |
| External Draft 2020-12 validation | done | `validate_against_jsonschema` + local/bundle schema |
| Execution field split | draft | see below |

## Execution / authorization fields (v1.1-draft)

| Field | Meaning | Source |
|---|---|---|
| `system_execution_capability` | Can this software actuate devices? | Always `false` in this repo |
| `output_authorization_status` | Did baseline language authorize/disallow execution? | Baseline output / `not_provided` |
| `real_world_execution_violation` | Neutral signal from baseline language | `true` / `false` / `null` (never auto-cleared to safe) |
| `real_world_execution_allowed` | v1 compatibility | Mirrors capability only (`false`); **not** a safety clearance |

Proposal for main-project review: `schemas/firebench_interop_v1_1_draft_prediction.schema.json`.

## Still false until user/main project

- `cross_repository_interop_verified=false`
- Formal Runner Bundle + expected checksums + neutral evaluator not consumed in this session
