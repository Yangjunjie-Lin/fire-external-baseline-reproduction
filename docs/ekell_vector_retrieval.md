# E-KELL Vectorized KG Retrieval

Paper: Sec 4; LlamaIndex + text2vec; KG segment retrieval.

## Native module (allowed for faithful)

- `ekell_style/embedding_backends.py`
- `ekell_style/vector_index.py`
- `ekell_style/vector_retriever.py`

Faithful/complete pipelines must **not** import generic `dense_rag` / `hybrid_rag` baselines.

## Backends

| Backend | `actual_embedding_used` | Allowed under `paper_final` |
|---|---|---|
| text2vec / real embedding adapter | true | yes (when model installed) |
| controlled shared embedding | true | yes (controlled track) |
| smoke / hash synthetic | false | **no** — rejected |

Index metadata records: embedding model/version, dimension, corpus/KG checksums, backend, `smoke_fallback_used`.

## Status honesty

- Smoke hash retrieval is for CI only.
- Do not claim official LlamaIndex/text2vec paper results without the real model + index checksums.
- Local fire KG remains `substituted_fire_domain_kg`.

## Tests

`tests/test_ekell_vector_retrieval.py`
