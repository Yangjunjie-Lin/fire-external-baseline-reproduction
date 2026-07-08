# Release Packaging Instructions

## Tagging a release

After final experiments and artifact QA:

```bash
git status
git tag -a vX.Y-paper-baselines -m "Paper baseline artifact release"
git push origin vX.Y-paper-baselines
```

## What to include

- Source code
- Config files
- Prompt templates
- Documentation
- Data cards and scenario cards
- Run manifests
- Metrics CSV files
- Manual evaluation summaries
- Statistical summaries
- Checksums

## What not to include

- API keys or `.env` files
- Private data that cannot be redistributed
- Large copied data without license clearance
- Smoke-test outputs mixed with real LLM outputs
- Target-project implementation code

## Checksums

Generate checksums for frozen input/output artifacts:

```bash
sha256sum path/to/file > path/to/file.sha256
```

## Archiving prompts/configs

For each final run, archive:

- config file
- prompt templates
- prompt checksums
- run manifest
- repository commit

## Citing external baselines

Cite the E-KELL paper and external repositories when used. Do not claim official reproduction unless official code/data/results are integrated and verified.

## Avoid exposing keys

- Never commit API keys.
- Record only provider, model, and API family/base URL family.
- Keep `.env` ignored.

## Separate smoke and real outputs

Use different output directories or filenames for heuristic smoke tests and real LLM experiments.

