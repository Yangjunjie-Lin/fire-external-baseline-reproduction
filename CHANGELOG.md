# Changelog

## 0.2.0 — Engineering convergence for submission-grade baseline package

- Single method registry as source of truth (`method_registry.py`)
- Formal manifest unified: `controlled_main_table_v1.yaml.example`
- Frozen E-KELL config renamed: `ekell_controlled_shared_llm_v1.yaml` (faithful ID retired from formal configs)
- Formal config validator rejects heuristic LLM and smoke/hash embeddings
- E-KELL controlled requires explicit `ekell_vector` with `reject_smoke: true`
- Legacy / smoke / proxy paths isolated in docs and scripts
- Cross-repo contract verification tool (heuristic smoke)
- Reproducibility artifact packaging (dry-run)
- Release-readiness audit and E-KELL fidelity audit automation
- Card templates and expanded CI gates

## 0.1.0 — Initial interop scaffold

- firebench-interop-v1 Runner Bundle support
- E-KELL-style full pipeline (controlled + paper-fidelity tracks)
- Proxy diagnostics evaluator (not shared paper evaluator)
