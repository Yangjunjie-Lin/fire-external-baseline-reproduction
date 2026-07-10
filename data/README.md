# Local data directory

This directory holds **copied** fire corpus / scenario inputs for local development.

## Rules

- Copy data only (e.g. `python scripts/prepare_data.py --source ../fire-agent-demo --target data/`).
- Do **not** import or vendor `fire_agent_demo` code.
- Do **not** commit private corpora, gold labels, or API keys.
- Prefer checksums / manifests when sharing data.

## Formal experiments

Formal interop runs use the main-project **Runner Bundle** (`input_cases.jsonl`), not `data/scenarios/` as the primary input.

`data/scenarios/` and `data/corpus/` remain for local/legacy development only.
