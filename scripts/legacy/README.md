# Legacy / diagnostic scripts

These scripts are **not** the formal firebench-interop-v1 entrypoint.

Formal entrypoint:

```bash
python scripts/run_interop_baselines.py --experiment-manifest ... --bundle ...
```

| Script | Role |
|---|---|
| `../run_baseline.py` | Single-method local generate |
| `../run_all_baselines.py` | Multi-method local generate |
| `../generate_predictions.py` | Local/dev prediction generation |
| `../evaluate_predictions.py` | **Proxy diagnostics only** (not shared paper evaluator) |
| `../build_report.py` / `../export_report.py` | Proxy reports |
| `../compare_with_target_outputs.py` | Side-by-side diagnostic |
| `../prepare_data.py` / `../validate_data.py` / `../audit_corpus.py` | Local data helpers |
| `../doctor.py` | Environment checks |
| `../smoke_main_runner_bundle.py` | Heuristic cross-repo smoke |
| `../smoke_interop.py` | Alias to smoke_main_runner_bundle |

Do not present these as paper-final experiment commands in the README main flow.
