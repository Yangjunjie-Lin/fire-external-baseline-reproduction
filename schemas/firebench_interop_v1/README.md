# firebench-interop-v1 schemas (local development copies)

## Authority

**Formal runs:** use the Runner Bundle’s `prediction_schema.json` (producer checksum in `manifest.checksums`).

**Local copies in this repo:** development fallback and unit tests only.

| File | Role |
|---|---|
| `../firebench_interop_v1_prediction.schema.json` | Synced snapshot of main-project Track A prediction schema |
| `../firebench_interop_v1_1_draft_prediction.schema.json` | Draft extensions (diagnostics); **not** formal Track A authority |

Formal mode must hard-fail if the bundle schema hash disagrees with the declared checksum — never silently override with the local file.
