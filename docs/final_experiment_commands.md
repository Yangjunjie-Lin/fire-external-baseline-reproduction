# Final Experiment Commands

> **Do not auto-run paid APIs.** Commands are for the user after license review + credentials.

## Unique formal command (main table only)

```bash
# 1) Align LLM with fire-agent-demo (SiliconFlow)
cp .env.example .env
# edit .env → set SILICONFLOW_API_KEY=...
cp configs/shared_real_model.yaml.example configs/shared_real_model.yaml
cp configs/experiments/paper_main_table_v1.yaml.example configs/experiments/paper_main_table_v1.yaml
# edit paper_main_table_v1.yaml → set bundle: <formal Runner Bundle path>

# 2) After main project exports formal firebench-interop-v1 Runner Bundle:
python scripts/run_interop_baselines.py \
  --experiment-manifest configs/experiments/paper_main_table_v1.yaml \
  --bundle path/to/formal_runner_bundle \
  --expected-bundle-checksum <bundle_checksum> \
  --output outputs/firebench_interop_v1_predictions.jsonl
```

This runs **only** main-table methods: `direct_llm`, `bm25_rag`, `ekell_style_faithful`.

Merge order per method: `base_config` → `shared_model_config` → method `config`.

## Supplemental (optional; never replaces faithful)

```bash
python scripts/run_interop_baselines.py \
  --experiment-manifest configs/experiments/paper_main_table_v1.yaml \
  --bundle path/to/formal_runner_bundle \
  --include-supplemental \
  --output outputs/firebench_interop_v1_predictions_supplemental.jsonl
```

## Post-bundle verification checklist (await formal bundle)

1. schema hash verification  
2. scenario hash verification  
3. corpus hash verification  
4. input-only / gold isolation  
5. baseline predictions JSONL  
6. neutral evaluator compatibility  

Until then: `cross_repository_interop_verified=false`.

## Smoke only (free / heuristic)

```bash
python scripts/generate_predictions.py \
  --methods direct_llm,bm25_rag,ekell_style_faithful \
  --config configs/deterministic_heuristic_smoke.yaml \
  --limit 1 \
  --output outputs/smoke_predictions.jsonl
```

## Deprecated

Multiple `--config` overlays on `run_interop_baselines.py` are **rejected**. Use the experiment manifest.
