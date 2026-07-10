# Statistical Analysis Plan

## Scope

This plan supports final analysis after real LLM runs and manual evaluation have been completed. It does not fabricate or assume results.

## Unit of analysis

The preferred unit is scenario-level paired comparison: each scenario receives outputs from all methods, and scores are compared within the same scenario.

## Primary comparisons

Paired case-level predictions enable:

- paired bootstrap CI
- Wilcoxon / permutation tests
- McNemar for safety binary outcomes
- effect sizes
- Holm correction
- category / occupancy / language breakdowns

Do not report only mean total scores.

Primary pairs (when eligible):

- Target system vs `direct_llm`
- Target system vs `bm25_rag`
- Target system vs `ekell_style_faithful` (and separately vs `ekell_style_enhanced`)
- Target system vs `dense_rag` / `hybrid_rag` only with real embeddings
- Target system vs actual LightRAG / Microsoft GraphRAG only when actual flags are true

## Metrics

Manual dimensions:

- correctness
- evidence support
- safety compliance
- completeness
- actionability
- conciseness
- comprehensibility
- overall

Binary error rates:

- critical error
- unsafe recommendation
- unsupported claim

Automatic proxy metrics should be reported as secondary/proxy evidence only.

## Recommended statistics

For 0-3 manual scores:

- mean and standard deviation by method
- paired mean difference vs SAFE
- bootstrap confidence intervals over scenarios
- paired effect size Cohen's d
- win/tie/loss rate by scenario

For binary fields:

- error rate by method
- paired difference in error rate vs SAFE
- confidence interval when sample size permits

## Multiple comparisons

If many methods/dimensions are tested, report correction strategy or clearly label analyses as exploratory.

## Reporting caveats

- Do not claim safety proof.
- Do not claim official E-KELL reproduction.
- Do not claim final top-tier validity without qualified evaluation and statistical reporting.

