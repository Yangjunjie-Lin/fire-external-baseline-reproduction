# Baseline Tuning Protocol

## Policy

- Tune **only** on DEV / validation.
- **Never** tune on test or private test.
- Freeze selected configs under `configs/frozen/`.
- Record search space, all DEV results, selection criterion, timestamp, and config SHA in `outputs/tuning/` before paper runs.

## Allowed knobs (DEV)

| Area | Knobs |
|---|---|
| Retrieval | top-k, candidate pool, max context chars |
| Hybrid | RRF k, lexical/dense weights |
| E-KELL | entity threshold, multi-hop/triple depth, context size |
| Prompts | structure / parser robustness (versioned templates) |
| Runtime | retry policy (not safety policy) |

## Forbidden

- Reading gold during generation
- Injecting target-system Safety Checker / SAFE-Router / HITL
- Adding enhanced features into `ekell_style_faithful`
- Claiming fallback GraphRAG as actual GraphRAG
- Using heuristic provider with `paper_final=true`

## Selection criteria (multi-objective)

Do not select on a single total score. Consider at least:

1. Safety-Gated score
2. Critical Failure Rate
3. Risk F1
4. Action F1
5. Blocked recall
6. Evidence support
7. Latency / cost

## Frozen files

```text
configs/frozen/direct_llm_v1.yaml
configs/frozen/bm25_rag_v1.yaml
configs/frozen/dense_rag_v1.yaml
configs/frozen/hybrid_rag_v1.yaml
configs/frozen/ekell_style_faithful_v1.yaml
configs/frozen/ekell_style_enhanced_v1.yaml
configs/frozen/freeze_manifest.json
```

Each frozen YAML declares `split_policy: tuned_on_dev_only_test_frozen` and currently **`freeze_status: provisional`**.

Promotion to locked freeze requires shared-LLM DEV trial logs, selection-criterion evidence, selected config SHA, and an explicit no-test-tuning statement (`configs/frozen/freeze_manifest.json`).

## Evidence that test was not used for tuning

- Freeze manifest forbids test tuning.
- CI and smoke use heuristic + tiny committed sample only.
- No test-split search logs are checked into `outputs/tuning/` for selection.
- Formal paper runs must attach DEV search logs + frozen config SHA before claiming freeze integrity.
