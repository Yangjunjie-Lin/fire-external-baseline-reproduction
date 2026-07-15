# Stage 0 Engineering Closeout

## Status

```text
package: fire-external-baseline-reproduction
version: 0.2.0
stage: Stage 0 — Engineering Complete / Pre-Experiment
closeout_date: 2026-07-15
audited_base_commit: a440ed4301db51732f01717f987930f289ac9de6
engineering_scope_closed: true
empirical_ready: false
paper_ready: false
formal_experiment_started: false
```

The audited base commit identifies the repository state reviewed before this closeout document was added. The authoritative closeout commits and CI results are recorded in the `main` branch history and attached GitHub Actions runs.

## Repository purpose

This repository is an independent prediction producer for system-level comparison with `fire-agent-demo`.

It is responsible for:

- consuming a main-project FireBench Runner Bundle;
- running independent external baseline methods;
- emitting taxonomy-compliant structured decisions and natural-language responses;
- exporting per-method `firebench-interop-v1` prediction JSONL;
- preserving runtime, configuration, index and artifact provenance.

It is not responsible for:

- benchmark Gold labels;
- final deterministic scoring;
- DeepEval or another LLM Judge;
- FireAgent SAFE-Router, Safety Checker, Dynamic REG or HITL;
- paper-facing claims before real experiments are executed.

## Closed method scope

### Formal main table

- `direct_llm`
- `bm25_rag`
- `ekell_style_controlled_shared_llm`

### Controlled supplemental comparison

- `dense_rag`
- `hybrid_rag`

### Separate or deferred methods

The following remain outside the primary Stage 0 comparison contract:

- `ekell_style_paper_fidelity`
- `ekell_style_enhanced`
- LightRAG and Microsoft GraphRAG adapters
- fallback graph retrieval
- legacy BM25/E-KELL diagnostics

No additional baseline method should be added before the main project publishes a stable comparison bundle and the existing five-method suite has completed a real dry run.

## Engineering infrastructure completed

The repository contains the engineering components required to receive future real resources:

- one Runner Bundle input contract;
- one `firebench-interop-v1` prediction contract;
- a single canonical method registry;
- `main_table` and `comparison_suite` method sets;
- strict FireBench taxonomy normalization and validation;
- Gold-isolation checks;
- external prediction-schema enforcement;
- exact case-set and duplicate-ID checks;
- Runner Bundle checksum and schema authority checks;
- shared generation-model identity checks;
- Dense and Hybrid retrieval/index implementations;
- controlled E-KELL pipeline-level reimplementation;
- strict FireKG loaders and audits;
- persisted index identity and integrity validation;
- stage-aware configuration validation;
- DEV evidence and complete-freeze infrastructure;
- fail-closed five-method preflight;
- runtime evidence recording;
- transactional Formal artifact publication;
- reproducibility artifact packaging;
- repository hygiene checks;
- Python 3.10–3.12 CI configuration.

These items establish engineering readiness only. They are not evidence of empirical performance.

## Artifact handoff contract

Future Formal prediction artifacts are expected under:

```text
outputs/formal/test_public/
├── predictions/
├── decisions/
├── suite_summary.json
├── run_manifest.json
└── diagnostics/
```

The evaluator handoff is:

```text
outputs/formal/test_public/predictions/
```

`fire-agent-demo` remains the authority for Track A scoring, semantic evaluation and final comparison reporting.

## Current verification boundary

The repository includes offline tests, audits and CI jobs for compilation, linting, unit tests, formal hardening, schema authority, freeze integrity, strict FireKG, packaging and repository hygiene.

This closeout does not itself claim that:

- a remote CI run passed;
- a real SiliconFlow request succeeded;
- a real embedding model was loaded;
- a real Dense or E-KELL index was built;
- a cross-repository Runner Bundle was executed;
- Formal TEST produced valid research results.

The authoritative remote CI status is the GitHub Actions result attached to the closeout commits. A missing, cancelled or failing workflow must be treated as unresolved engineering work.

## Deferred empirical work

The following remain intentionally deferred:

1. receive the stable main-project v1 Runner Bundle;
2. verify live schema and taxonomy parity;
3. select and configure the shared generation model;
4. install or mount the real embedding model and revision;
5. build real Dense and E-KELL persisted indexes;
6. run a 1–3 case SiliconFlow dry run;
7. tune permitted parameters on DEV only;
8. save selected DEV evidence;
9. generate and manually review the complete freeze;
10. mark the experiment configuration frozen;
11. run full Formal TEST without partial-case options;
12. hand predictions to the main-project evaluator;
13. produce statistical and paper-facing analysis.

## Claim boundary

The valid Stage 0 claim is:

> The external baseline repository has completed its engineering implementation and formal experiment infrastructure, and is ready to receive stable scenarios, corpora, indexes and model resources for a future controlled comparison.

The following claims are not valid at this stage:

- formal experiment complete;
- empirically validated;
- paper ready;
- official E-KELL reproduction;
- real firefighter decision accuracy;
- superiority or inferiority relative to FireAgent.

## Reopening criteria

Baseline development should resume only when at least one of the following becomes true:

- `fire-agent-demo` publishes a stable Runner Bundle and evaluator contract;
- the shared generation-model identity is selected for DEV;
- real embedding resources are available;
- a real cross-repository dry run reveals a concrete correctness defect;
- the main-project evaluator requires a narrowly scoped output-contract fix;
- a reproducibility or CI regression is identified.

Until then, work should focus on the main FireAgent system and its evaluation quality rather than adding new baseline features.
