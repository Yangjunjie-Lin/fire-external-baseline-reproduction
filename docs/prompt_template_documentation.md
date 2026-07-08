# Prompt Template Documentation

## Location

E-KELL-style prompt templates live in:

- `configs/prompts/ekell_stage1_situation_understanding.txt`
- `configs/prompts/ekell_stage2_kg_grounded_decision_reasoning.txt`
- `configs/prompts/ekell_stage3_final_response.txt`

## Stage 1: Situation understanding

Purpose: Convert scenario text, deterministic/LLM parser output, and retrieved KG/evidence context into a structured situation understanding.

Input variables:

- `{scenario_text}`
- `{parsed_scenario_json}`
- `{context_block}`

Expected JSON schema:

```json
{
  "emergency_type": "",
  "involved_entities": [],
  "hazards": [],
  "emergency_stage": "",
  "missing_information": [],
  "evidence_used": []
}
```

## Stage 2: KG-grounded decision reasoning

Purpose: Use Stage 1 and retrieved KG/evidence to derive candidate actions, deferred/unsupported actions, missing information, and evidence links.

Input variables:

- `{scenario_text}`
- `{stage1_json}`
- `{context_block}`

Expected JSON schema:

```json
{
  "reasoning_summary": "",
  "candidate_actions": [],
  "deferred_or_unsupported_actions": [],
  "missing_information": [],
  "evidence_links": []
}
```

## Stage 3: Final emergency response

Purpose: Produce the unified output-compatible final external-baseline response.

Input variables:

- `{scenario_text}`
- `{stage1_json}`
- `{stage2_json}`
- `{context_block}`

Expected JSON schema:

```json
{
  "situation_summary": "",
  "key_risks": [],
  "recommended_actions": [],
  "blocked_or_unsafe_actions": [],
  "missing_confirmations": [],
  "supporting_evidence": [],
  "citations": [],
  "final_decision_gate": ""
}
```

## Paper-faithful rationale

The prompts are paper-faithful at pipeline level because they preserve the E-KELL-style sequence: scenario understanding, KG-grounded reasoning, and evidence-based final response.

## Differences from official E-KELL prompts

Official exact E-KELL prompts are not integrated. These templates are independent reimplementations based on paper-level pipeline descriptions.

## Versioning prompts

For final experiments, record:

- prompt file paths
- prompt checksums
- repository commit
- model/run card
- any changes from prior experiments

Prompt changes after final runs should create a new run version.

