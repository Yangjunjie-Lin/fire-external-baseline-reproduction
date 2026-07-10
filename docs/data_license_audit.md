# Data License Audit

**Status:** `user_review_required` — do not expand public redistribution until cleared.

## Policy (public baseline repo)

1. Audit every corpus/KG/scenario source for redistribution rights.
2. Maintain a data manifest (template: `data/manifests/data_license_manifest.template.json`) with:
   - source URL, organization, license/terms
   - `redistribution_allowed`
   - collected_at, checksum, language, source type, review status
3. If license cannot be confirmed:
   - do not commit additional raw content
   - use `scripts/prepare_data.py` locally
   - keep checksums
   - mark `user_review_required`
4. Never publish main-project private / uncleared data to this public repo.

## Current committed sample

Tiny smoke fixtures under `data/corpus/` and `data/scenarios/` exist for CI. Provenance/license of any content copied from `fire-agent-demo` is **not yet user-cleared** for public redistribution.

## Migration recommendation (for user review — no automatic deletion)

1. Fill `data/manifests/data_license_manifest.template.json` per source.
2. If redistribution is disallowed: keep prepare/download scripts + checksums; gitignore full corpus; retain only synthetic smoke fixtures if needed.
3. If allowed: record license text URL + checksums; update `data/README.md`.
4. Do not delete existing files until the user approves the migration plan.

## Related

- `data/README.md`
- `docs/release_packaging.md`
- `scripts/prepare_data.py`
- `scripts/audit_corpus.py`
