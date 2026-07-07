# Data directory

This directory is for copied input data only. Do not import or copy code from `fire-agent-demo`.

Expected files:

```text
corpus/evidence_chunks.jsonl
corpus/entities.jsonl
corpus/relations.jsonl
corpus/triples.jsonl
scenarios/scenario_matrix_v2.json
```

Run:

```bash
python scripts/prepare_data.py --source ../fire-agent-demo --target data/
```

A tiny example scenario is committed so the command-line scripts can be smoke-tested without the target repository. Replace it with the copied scenario matrix for real comparison.
