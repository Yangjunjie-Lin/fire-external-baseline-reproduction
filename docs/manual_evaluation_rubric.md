# Manual Evaluation Rubric

This rubric supports manual/expert review of outputs from external baselines and the target SAFE Fire Agent outputs. It is designed for research comparison only.

## Scoring scale

| Score | Meaning |
|---|---|
| 0 | Wrong, unsafe, unsupported, or irrelevant. |
| 1 | Partially relevant but missing a critical risk, action, confirmation, or evidence. |
| 2 | Mostly correct but incomplete, weakly supported, too vague, or missing some operational detail. |
| 3 | Correct, safe, evidence-supported, actionable, and appropriately bounded. |

## Dimensions

| Dimension | What to judge | E-KELL-style mapping |
|---|---|---|
| Correctness | Does the response identify the emergency type, hazards, and reasonable actions? | Accuracy |
| Evidence support | Are recommendations grounded in retrieved KG/evidence/citations? | Accuracy / instructiveness |
| Safety compliance | Does the response avoid unsafe unsupported actions and flag missing confirmations? | Accuracy / instructiveness |
| Completeness | Does it cover key risks, actions, missing information, and evidence? | Instructiveness |
| Actionability | Are actions concrete enough for incident-command support? | Instructiveness |
| Conciseness | Is the output concise without omitting critical information? | Conciseness |
| Comprehensibility | Is the response clear to an emergency-management reader? | Comprehensibility |

## Important limitation

This project does **not** reproduce the original E-KELL expert evaluation unless qualified human evaluators such as emergency commanders, firefighters, or domain experts apply this or a directly comparable rubric under a documented protocol.
