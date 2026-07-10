# Final Experiment Commands

> **Do not auto-run paid APIs.** Commands are for the user after license review + credentials.

## A. Controlled comparison (main table; shared SiliconFlow)

```bash
cp .env.example .env
# edit .env → set SILICONFLOW_API_KEY=...
cp configs/shared_real_model.yaml.example configs/shared_real_model.yaml
cp configs/experiments/paper_main_table_v1.yaml.example configs/experiments/paper_main_table_v1.yaml
# edit paper_main_table_v1.yaml → set bundle: <formal Runner Bundle path>
# set expected_bundle_checksum when paper_final=true

python scripts/run_interop_baselines.py \
  --experiment-manifest configs/experiments/paper_main_table_v1.yaml \
  --bundle path/to/formal_runner_bundle \
  --expected-bundle-checksum <bundle_checksum> \
  --output outputs/firebench_interop_v1_predictions.jsonl
```

Main-table methods: `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm`.

Merge order per method: `base_config` → `shared_model_config` → method `config`.

## B. Paper fidelity (ChatGLM-6B; separate experiment)

```bash
cp configs/experiments/ekell_paper_fidelity.yaml.example configs/experiments/ekell_paper_fidelity.yaml
cp configs/models/chatglm6b_local.yaml.example configs/models/chatglm6b_local.yaml
# Configure local ChatGLM-6B paths/hardware on the user server.
# paper_fidelity_model_run remains false until a real run completes.

python scripts/generate_predictions.py \
  --methods ekell_style_paper_fidelity \
  --config configs/ekell_paper_fidelity_chatglm6b.yaml \
  --limit <N> \
  --output outputs/ekell_paper_fidelity_predictions.jsonl
```

Do not merge paper-fidelity outputs with controlled FireBench rows as one result.

## Supplemental (optional; never replaces controlled/fidelity)

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
  --methods direct_llm,bm25_rag,ekell_style_controlled_shared_llm \
  --config configs/deterministic_heuristic_smoke.yaml \
  --limit 1 \
  --output outputs/smoke_predictions.jsonl
```

## Deprecated

Multiple `--config` overlays on `run_interop_baselines.py` are **rejected**. Use the experiment manifest.
