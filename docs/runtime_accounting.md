# Runtime Accounting

## Per-case scope

For each case:

```text
before = llm.usage_snapshot()
run pipeline
after = llm.usage_snapshot()
case_usage = after - before
```

Recorded on each prediction (`method_specific.runtime` / interop `runtime`):

- `case_llm_calls` / `llm_calls`
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `latency_sec` / `latency_ms`
- optional `cost`

## Run-level aggregation

Run manifest aggregates totals across cases/methods. Per-case values must not be cumulative mid-run counters.

## Tests

- `tests/test_per_case_usage.py`
- Runner uses `UsageTrackingLLMClient` when available (`src/external_baselines/runner.py`)
