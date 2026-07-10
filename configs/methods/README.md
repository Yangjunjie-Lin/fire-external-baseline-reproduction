# Method configs

Prefer frozen configs under `configs/frozen/` for paper-facing method knobs.

Root-level method YAMLs (`configs/ekell_style.yaml`, `configs/vanilla_rag.yaml`, …) remain for local/dev compatibility.

Canonical method IDs live in `src/external_baselines/method_registry.py` — config filenames may still say `vanilla_rag` or `ekell_style_faithful` historically while method_id is canonicalized at runtime.
