# Model / Run Card Template

Fill one card per method and final run. Do not expose API keys.

## Run identity

- Run name:
- Method name:
- Method label:
- Run timestamp:
- Repository commit:
- Run manifest path:
- Output path:
- Output SHA256:

## Model configuration

- Model provider:
- Model version / model name:
- API family or base URL family, without keys:
- Temperature:
- Max tokens:
- Seed if applicable:
- Heuristic mode used? yes/no:

> Heuristic mode is only for smoke tests and must not be reported as a final LLM result.

## Prompt configuration

- Prompt template version:
- Prompt template commit:
- Prompt files:
- Prompt checksum(s):
- Prompt changes since previous run:

## Retrieval configuration

- Retrieval backend:
- Corpus version:
- Dataset/scenario version:
- top_k:
- top_k_entities:
- top_k_triples:
- top_k_evidence:
- GraphRAG package used? yes/no:
- Indexing performed? yes/no:
- Query performed? yes/no:

## Known failure modes

- Hallucination risk:
- Missing evidence risk:
- Retrieval miss risk:
- Citation mismatch risk:
- Over-refusal / under-action risk:
- Domain/jurisdiction mismatch risk:

## Notes

- Deviations from protocol:
- Errors encountered:
- Rerun reason if applicable:

