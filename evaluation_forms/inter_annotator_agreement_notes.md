# Inter-annotator Agreement Notes

This file describes recommended agreement checks for final manual evaluation. It does not implement heavy statistics.

## Binary fields

For `critical_error`, `unsafe_recommendation`, and `unsupported_claim`, report Cohen's kappa for two evaluators or Fleiss' kappa / Krippendorff's alpha for more than two evaluators.

Recommended reporting:

| Field | Agreement statistic | Value | Number of scenarios | Number of evaluators |
|---|---|---:|---:|---:|
| critical_error | Cohen's kappa / Fleiss' kappa | `<fill>` | `<fill>` | `<fill>` |

## 0-3 ordinal scores

For dimensions such as correctness and safety compliance, report one of:

- Krippendorff's alpha for ordinal data
- ICC if treating scores as interval-like
- pairwise weighted Cohen's kappa for two evaluators

Recommended reporting:

| Dimension | Statistic | Value | Notes |
|---|---|---:|---|
| correctness_0_3 | Krippendorff alpha / ICC | `<fill>` | `<fill>` |

## Disagreement handling

- Keep raw individual scores.
- Report mean score across evaluators.
- If adjudication is used, report both raw and adjudicated results.
- Document adjudication rules before inspecting final method comparisons.

## Caution

Do not claim expert validation unless evaluators are qualified and the protocol is documented.

