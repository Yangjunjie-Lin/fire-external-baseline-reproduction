# Final Experiment Commands

> **Do not auto-run paid APIs.** Commands are for the user after license review + credentials.
>
> `.example` files are templates only. Copy to non-`.example` paths before formal runs.
> `python scripts/validate_formal_config.py --allow-placeholders` checks template structure only.

## A. Controlled comparison (main table; shared SiliconFlow)

```bash
cp .env.example .env
# edit .env → set SILICONFLOW_API_KEY=...

cp configs/models/shared_real_model.yaml.example configs/models/shared_real_model.yaml
cp configs/experiments/controlled_main_table_v1.yaml.example configs/experiments/controlled_main_table_v1.yaml
# edit controlled_main_table_v1.yaml → set bundle: <formal Runner Bundle path>
# replace all ekell_vector / model placeholders

python scripts/validate_formal_config.py \
  --config configs/experiments/controlled_main_table_v1.yaml

python scripts/run_interop_baselines.py \
  --execution-stage formal \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle path/to/formal_runner_bundle \
  --output outputs/interop/controlled_main_table_v1/predictions.jsonl
```

Formal stage forbids `--limit` and `--allow-partial`.

Dry-run (after main project v1; not paper results):

```bash
python scripts/run_interop_baselines.py \
  --execution-stage dry_run \
  --experiment-manifest configs/experiments/controlled_main_table_v1.yaml \
  --bundle path/to/runner_bundle \
  --limit 3 \
  --output outputs/dry_run/controlled_v1/predictions.jsonl \
  --manifest outputs/dry_run/controlled_v1/run_manifest.json
```

Formal model identity is frozen in YAML (`configs/models/shared_real_model.yaml`). Env vars supply credentials/endpoints only; `SILICONFLOW_MODEL` does not silently override YAML.

Main-table methods: `direct_llm`, `bm25_rag`, `ekell_style_controlled_shared_llm`.

## B. Paper fidelity (ChatGLM-6B; separate experiment)

```bash
cp configs/experiments/ekell_paper_fidelity_v1.yaml.example configs/experiments/ekell_paper_fidelity_v1.yaml
cp configs/models/chatglm6b_local.yaml.example configs/models/chatglm6b_local.yaml
cp configs/ekell_paper_fidelity_chatglm6b.yaml.example configs/ekell_paper_fidelity_chatglm6b.yaml
# Configure local ChatGLM-6B + text2vec; paper_fidelity_model_run remains false until evidenced.

python scripts/validate_formal_config.py \
  --config configs/experiments/ekell_paper_fidelity_v1.yaml
```

Do not merge paper-fidelity outputs with controlled FireBench rows as one result.

## Template validation (structure only)

```bash
python scripts/validate_formal_config.py \
  --config configs/experiments/controlled_main_table_v1.yaml.example \
  --allow-placeholders
```

## Supplemental (optional; never replaces controlled/fidelity)

```bash
python scripts/run_interop_baselines.py \
  --experiment-manifest configs/experiments/supplemental_v1.yaml \
  --bundle path/to/formal_runner_bundle \
  --include-supplemental \
  --output outputs/interop/supplemental_v1/predictions.jsonl
```

## Local validation evidence

```bash
python scripts/collect_validation_evidence.py
# writes outputs/diagnostics/validation_evidence.json
```

Contract tool ready ≠ contract verified. See `docs/status/readiness_summary.md`.
