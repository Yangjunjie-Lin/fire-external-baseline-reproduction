# Outputs

Runtime artifacts belong here. Do not commit predictions, metrics, logs, or model responses.

Suggested layout:

```text
outputs/
├─ interop/       # formal / smoke interop JSONL + manifests
├─ smoke/         # heuristic smoke
├─ diagnostics/   # proxy metrics / reports
└─ legacy/        # older local runs
```

Only `README.md` and `.gitkeep` should be tracked in git.
