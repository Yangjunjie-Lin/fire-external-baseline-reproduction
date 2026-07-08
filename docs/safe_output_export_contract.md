# SAFE Fire Agent Normalized Output Export Contract

This repository only consumes exported normalized files from `fire-agent-demo`. It does not import or run target-project code.

The target system should be run separately and exported as JSONL with one record per scenario.

## Required schema

```json
{
  "scenario_id": "...",
  "method": "safe_fire_agent",
  "situation_summary": "...",
  "key_risks": [],
  "recommended_actions": [],
  "blocked_or_unsafe_actions": [],
  "missing_confirmations": [],
  "supporting_evidence": [],
  "citations": [],
  "final_decision_gate": "...",
  "retrieved_contexts": [],
  "latency_sec": 0.0,
  "raw_output": {},
  "method_specific": {
    "source_project": "fire-agent-demo",
    "export_only": true
  }
}
```

## Rules

- This repository only reads the exported JSONL file.
- This repository must not import `fire_agent_demo`.
- This repository must not call SAFE-Router, Safety Checker, Dynamic REG, HITL Gate, or target risk scoring.
- Target outputs should be produced in the target repository or an independent export step.
- Keep scenario IDs identical to the baseline scenario matrix.

