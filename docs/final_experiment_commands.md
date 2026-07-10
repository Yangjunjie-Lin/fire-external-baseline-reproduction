# Final Experiment Commands

> Do **not** auto-run paid APIs. Commands below are for the user to execute after license review and credential setup.

## 0. Prepare Runner Bundle (from main project)

Place the main-project Runner Bundle locally (scenarios input-only, corpus/KG snapshot, experiment config, checksums). Do not use Evaluator Bundle.

## 1. Smoke (heuristic, free)

```bash
python scripts/generate_predictions.py \
  --methods direct_llm,bm25_rag,dense_rag,hybrid_rag,ekell_style_faithful,ekell_style_enhanced,fallback_graph_retrieval \
  --config configs/deterministic_heuristic_smoke.yaml \
  --limit 1 \
  --output outputs/smoke_predictions.jsonl

python -m pytest -q
```

## 2. Interop generation (real shared model — user runs)

```bash
# Copy and edit:
#   configs/shared_real_model.yaml.example → configs/shared_real_model.yaml
# Set OPENAI_API_KEY / OPENAI_BASE_URL (or provider equivalents).

python scripts/run_interop_baselines.py \
  --bundle path/to/runner_bundle \
  --methods direct_llm,bm25_rag,ekell_style_faithful \
  --config configs/shared_real_model.yaml \
  --config configs/frozen/bm25_rag_v1.yaml \
  --config configs/frozen/ekell_style_faithful_v1.yaml \
  --output outputs/firebench_interop_v1_predictions.jsonl
```

Optional dense/hybrid (only after real embedding model is configured):

```bash
python scripts/run_interop_baselines.py \
  --bundle path/to/runner_bundle \
  --methods dense_rag,hybrid_rag,ekell_style_enhanced \
  --config configs/shared_real_model.yaml \
  --config configs/frozen/dense_rag_v1.yaml \
  --config configs/frozen/hybrid_rag_v1.yaml \
  --config configs/frozen/ekell_style_enhanced_v1.yaml \
  --output outputs/firebench_interop_v1_predictions_enhanced.jsonl
```

## 3. Paper-final guard

`paper_final: true` rejects:

- heuristic provider
- missing `model` / `model_version`
- (optional) missing bundle checksum when `require_bundle_checksum: true`

Fallback GraphRAG cannot set `actual_graphrag=true`.

## 4. Evaluation

- Local proxy only: `python scripts/evaluate_predictions.py --predictions ... --dataset ...`
- Paper scores: main-project neutral shared evaluator on `outputs/firebench_interop_v1_predictions.jsonl`

## 5. Formal leaderboard eligibility

| Enter formal system-level board | Smoke / fallback only |
|---|---|
| direct_llm, bm25_rag, ekell_style_faithful, ekell_style_enhanced (separate rows) | heuristic runs |
| dense/hybrid **with real embeddings** | dense/hybrid smoke_hash |
| actual LightRAG / GraphRAG when flags prove indexing+query | lightrag/microsoft_graphrag fallback; `fallback_graph_retrieval` |
