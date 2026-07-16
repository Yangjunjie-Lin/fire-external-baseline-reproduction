# External Baseline DeepEval Handoff v1

This repository produces predictions. It does not evaluate them.

DeepEval is not installed in this repository. DeepEval is not executed in this repository. This repository only exports completed predictions. Formal scoring remains owned by `fire-agent-demo`.

## Contract

The handoff uses `fireagent-external-prediction-v1`, snapshotted from:

```text
Yangjunjie-Lin/fire-agent-demo
schemas/deepeval_external_v1/external_prediction.schema.json
```

`schemas/deepeval_handoff/contract_provenance.json` binds the local snapshot to the exact main-project commit and SHA-256. Check parity without modifying either repository:

```bash
python scripts/check_deepeval_contract_snapshot.py --main-repo ../fire-agent-demo
```

Snapshot updates are explicit and reject uncommitted source Schema changes:

```bash
python scripts/check_deepeval_contract_snapshot.py \
  --main-repo ../fire-agent-demo \
  --update-snapshot
```

## Retrieval semantics

`direct_llm` is output-only. Its records omit `retrieval_context`; an empty array is not emitted and corpus content is never fabricated.

`bm25_rag`, `dense_rag`, `hybrid_rag`, and `ekell_style_controlled_shared_llm` are output-and-RAG methods only when their real ordered retrieval contexts are present. The handoff preserves the submitted order and explicit ranks. Invalid, duplicate, mixed, or decreasing ranks fail closed.

`handoff_top_k` limits only the submitted evaluation context. It does not change native retrieval. Truncation keeps exactly the original rank prefix and records native count, submitted count, truncation status, and top-k. Missing retrieval context is N/A or handoff-invalid, never automatically scored as zero or one.

## Export and validation

```bash
python scripts/export_deepeval_handoff.py \
  --formal-run-root outputs/formal/test_public \
  --main-repo ../fire-agent-demo \
  --output outputs/formal/test_public/deepeval_handoff \
  --top-k 5

python scripts/validate_deepeval_handoff.py \
  --handoff outputs/formal/test_public/deepeval_handoff \
  --main-repo ../fire-agent-demo
```

Formal export requires committed transactional publication, complete approved method and case coverage, frozen source hashes, Runner Bundle identity, and source worktree identity. `--allow-development-source` is only for fixtures and diagnostics; it fixes `formal_source=false`, `publication_eligible=false`, and `development_artifact=true`.

Export is transactional: a temporary directory is fully generated, Schema-checked, coverage-checked, hash-checked, and manifest-checked before atomic replacement. A failed replacement preserves the previous valid handoff and does not alter Formal source artifacts.

The bundle contains one JSONL per method, `handoff_manifest.json`, `validation_report.json`, `contract_provenance.json`, and an automatically generated README. It contains no Gold, expected output, score, Judge result, or evaluation label.

## Evaluation boundary

A valid handoff bundle has not yet been evaluated. A handoff artifact is not an evaluation result.

`fire-agent-demo` alone loads canonical cases and Gold, runs DeepEval and deterministic FireAgent-Bench evaluation, and produces comparisons or leaderboards. This repository does not call a Judge, use a paid evaluation API, or claim externally validated or real-world fire-safety accuracy. Real-world execution remains prohibited.
