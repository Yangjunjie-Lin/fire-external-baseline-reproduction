# Baseline Handoff Checklist

Use this checklist only after `fire-agent-demo` publishes a stable comparison contract. Do not treat completion of the repository's Stage 0 engineering work as permission to skip DEV selection, freeze review or Formal validation.

## 1. Receive the main-project bundle

- [ ] Obtain the approved Runner Bundle from `fire-agent-demo`.
- [ ] Confirm the intended split: DEV, `test_public`, or another explicitly approved split.
- [ ] Confirm the bundle contains only prediction-time fields.
- [ ] Confirm no Gold, severity label, expected decision or annotation metadata is exposed.
- [ ] Record the bundle path and producer commit.
- [ ] Verify the aggregate bundle checksum and every declared file checksum.

## 2. Verify interop contracts

- [ ] Run `scripts/check_firebench_contract_snapshot.py` against the main project.
- [ ] Run `scripts/check_firebench_taxonomy_snapshot.py` against the main project.
- [ ] Validate the Runner Bundle with the repository's strict bundle loader.
- [ ] Confirm the in-bundle prediction schema is the Formal schema authority.
- [ ] Confirm all case IDs are canonical, unique and complete.
- [ ] Resolve any contract mismatch before model or index work begins.

## 3. Prepare the shared generation model

- [ ] Copy `configs/models/shared_real_model.yaml.example` to the non-example path.
- [ ] Select the shared generation model on DEV, not on TEST.
- [ ] Freeze provider, model, model version, temperature, top-p, max tokens, seed and thinking mode in YAML.
- [ ] Keep credentials and endpoint settings in environment variables only.
- [ ] Confirm `SILICONFLOW_API_KEY` is not committed or printed.
- [ ] Confirm environment variables cannot silently override the frozen model identity.

## 4. Prepare real retrieval resources

- [ ] Install or mount the selected embedding model and exact revision.
- [ ] Record embedding backend, model, version, dimension and normalization policy.
- [ ] Confirm corpus and FireKG files pass strict validation.
- [ ] Confirm all required E-KELL prompt files exist and are non-empty.
- [ ] Confirm resource licenses and redistribution rules before packaging artifacts.

## 5. Build and validate indexes

- [ ] Run full `index_build_candidate` validation before embedding initialization.
- [ ] Build the real Dense persisted index.
- [ ] Confirm Hybrid reuses the approved Dense index identity.
- [ ] Build the real E-KELL persisted index.
- [ ] Confirm `actual_embedding_used=true`.
- [ ] Confirm `smoke_fallback_used=false`.
- [ ] Validate document, embedding, corpus and KG checksums.
- [ ] Validate finite, nonzero embeddings.
- [ ] Validate unit norm when normalization is enabled.
- [ ] Run `build_comparison_indexes.py --validate-only` and require exit code 0.

## 6. Run a real API dry run

- [ ] Open only the dry-run readiness gates.
- [ ] Use 1–3 representative cases initially.
- [ ] Keep `formal_result=false`.
- [ ] Run the five-method comparison suite with the shared model identity.
- [ ] Confirm schema validity and complete output fields.
- [ ] Confirm parser stability for structured decision arrays.
- [ ] Confirm token, latency and cost recording.
- [ ] Confirm no method reads evaluator Gold.
- [ ] Confirm failed methods do not produce a partial Formal artifact.

## 7. Tune on DEV only

- [ ] Select BM25 parameters on DEV.
- [ ] Select Dense top-k on DEV.
- [ ] Select Hybrid/RRF parameters on DEV.
- [ ] Select E-KELL retrieval and context parameters on DEV.
- [ ] Do not tune on `test_public` or `challenge`.
- [ ] Save the selected DEV run as an immutable evidence artifact.
- [ ] Record all rejected and selected configurations honestly.

## 8. Create the complete freeze

- [ ] Keep the experiment manifest `freeze_status: provisional` during freeze-candidate validation.
- [ ] Run freeze-candidate validation.
- [ ] Generate the complete freeze with `scripts/create_freeze_manifest.py`.
- [ ] Verify the Runner Bundle aggregate before index hashing.
- [ ] Bind the selected DEV evidence path and SHA-256.
- [ ] Bind all method configs and model identities.
- [ ] Bind Dense, Hybrid and E-KELL index identities.
- [ ] Bind the E-KELL prompt directory, required prompt hashes and prompt-tree hash.
- [ ] Confirm all paths are portable and non-authoritative machine-local paths are diagnostic only.
- [ ] Manually review the complete freeze.
- [ ] Mark the experiment manifest `freeze_status: frozen` only after review.

## 9. Run Formal validation

- [ ] Run `validate_formal_config.py --validation-stage formal`.
- [ ] Require the frozen Runner Bundle identity to match exactly.
- [ ] Require all live index identities to match the freeze exactly.
- [ ] Confirm no `--limit` is present.
- [ ] Confirm no partial-case mode is enabled.
- [ ] Confirm DEV aliases are disabled.
- [ ] Confirm only the five approved comparison methods are enabled.
- [ ] Confirm all five methods pass preflight before any LLM call.

## 10. Execute full Formal TEST

- [ ] Use the single approved `--formal-run-root`.
- [ ] Process the complete Runner Bundle.
- [ ] Close method runtimes and caches before staged validation.
- [ ] Revalidate staged predictions against the frozen in-bundle schema.
- [ ] Verify every output artifact hash.
- [ ] Require transactional publication to complete.
- [ ] Treat any pre-commit failure as a failed Formal run.
- [ ] Preserve external control diagnostics and publish receipts.
- [ ] Do not rewrite immutable Formal artifacts after commit.

## 11. Validate evaluator handoff

- [ ] Run `scripts/check_output_taxonomy.py` on the Formal prediction directory.
- [ ] Recheck FireBench schema parity.
- [ ] Recheck taxonomy parity.
- [ ] Confirm one prediction JSONL exists for every approved method.
- [ ] Confirm every method covers the identical case set.
- [ ] Confirm `real_world_execution_allowed` remains false.
- [ ] Validate `fireagent-external-prediction-v1` snapshot parity.
- [ ] Export one DeepEval handoff JSONL per method.
- [ ] Verify Direct LLM has no fabricated retrieval context.
- [ ] Verify RAG methods preserve real ranked retrieval contexts.
- [ ] Verify handoff top-k and original-rank-prefix policy.
- [ ] Verify all method case sets are identical.
- [ ] Verify Gold was not accessed.
- [ ] Verify `deepeval_executed=false`.
- [ ] Hand the complete handoff bundle to `fire-agent-demo`.
- [ ] Hand `outputs/formal/test_public/predictions/` to `fire-agent-demo`.
- [ ] Keep Gold and scoring exclusively in the main-project evaluator.

## 12. Preserve reproducibility evidence

- [ ] Archive the experiment manifest and complete freeze.
- [ ] Archive the Runner Bundle manifest and checksums.
- [ ] Archive method configs and prompt hashes.
- [ ] Archive persisted index manifests and hashes.
- [ ] Archive prediction, decision and response artifacts.
- [ ] Archive suite summary, run manifest and diagnostics.
- [ ] Record repository commits for both projects.
- [ ] Record environment and package versions.
- [ ] Record model usage, token counts, latency and cost.
- [ ] Produce data, model and run cards.

## 13. Reporting boundary

Before reporting results, verify:

- [ ] deterministic Track A scoring was performed by `fire-agent-demo`;
- [ ] any DeepEval or semantic Judge used the same predictions and frozen rubric;
- [ ] Critical Failures were not averaged away;
- [ ] RAG-only metrics were not assigned to Direct LLM as zero or one;
- [ ] no smoke or DEV result was presented as Formal TEST;
- [ ] benchmark-label and synthetic-state limitations are stated;
- [ ] no real-world firefighting accuracy claim is made.
