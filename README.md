# fire-external-baseline-reproduction

Independent external baseline reproduction project for system-level comparison with `fire-agent-demo`.

This repository is intended to remain separate from the target project. It must not import or call SAFE-Router, Safety Checker, Dynamic REG, HITL Gate, or other internal target-project control modules.

I generated a complete first-milestone scaffold locally in this session, including:

- B0 `direct_llm`
- B1 `vanilla_rag`
- B2 `ekell_style` / E-KELL-style paper-faithful reimplementation
- optional GraphRAG / LightRAG adapter fallbacks
- unified output schema
- lightweight metrics and report export scripts
- reproduction/comparison documentation

Because the current GitHub connector session could not bulk-upload the full multi-file tree reliably, the complete scaffold is provided as an external generated artifact in the ChatGPT response. Extract it into this repository root, then run:

```bash
pip install -r requirements.txt
python scripts/run_all_baselines.py --methods direct_llm,vanilla_rag,ekell_style --dataset data/scenarios/scenario_matrix_v2.json --limit 1
```

Main boundaries:

- research comparison only
- no real-world emergency advice
- copied data input allowed
- target-project code import not allowed
- deviations from external papers must be documented
