from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from external_baselines.common.checksums import sha256_json, sha256_text

from .schema import SCHEMA_VERSION, KGTriple, schema_document
from .triple_validator import validate_triple


def _as_triple(value: KGTriple | Mapping[str, Any]) -> KGTriple:
    return value if isinstance(value, KGTriple) else KGTriple.from_mapping(value)


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def build_kg(
    triples: Iterable[KGTriple | Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    include_candidates: bool = True,
) -> dict[str, Any]:
    """Build reproducible KG assets from valid approved and, optionally, candidate triples."""
    selected: list[KGTriple] = []
    excluded: list[dict[str, str]] = []
    for raw in triples:
        triple = _as_triple(raw)
        validation = validate_triple(triple)
        if not validation.valid:
            raise ValueError(f"invalid triple {triple.triple_id}: {'; '.join(validation.errors)}")
        if triple.review_status == "rejected":
            excluded.append({"triple_id": triple.triple_id, "reason": "rejected"})
        elif triple.review_status == "candidate" and not include_candidates:
            excluded.append({"triple_id": triple.triple_id, "reason": "candidate_excluded"})
        else:
            selected.append(triple)

    triple_rows = [triple.to_dict() for triple in sorted(selected, key=lambda item: item.triple_id)]
    names = sorted({name for triple in selected for name in (triple.subject, triple.object)})
    entities = [
        {"entity_id": f"entity:{sha256_text(name)[:20]}", "name": name}
        for name in names
    ]
    relations = [{"relation_id": relation, "name": relation} for relation in sorted({t.predicate for t in selected})]
    evidence_by_chunk = {
        (t.source_id, t.chunk_id): {
            "source_id": t.source_id,
            "chunk_id": t.chunk_id,
            "text": t.source_text,
            "text_sha256": sha256_text(t.source_text),
        }
        for t in selected
    }
    evidence = [evidence_by_chunk[key] for key in sorted(evidence_by_chunk)]
    assets = {
        "entities.jsonl": entities,
        "relations.jsonl": relations,
        "triples.jsonl": triple_rows,
        "evidence_chunks.jsonl": evidence,
    }
    asset_checksums = {
        filename: sha256_text(_jsonl(rows)) for filename, rows in assets.items()
    }
    manifest = {
        "manifest_version": 1,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_candidates": include_candidates,
        "candidate_count": sum(t.review_status == "candidate" for t in selected),
        "approved_count": sum(t.review_status == "approved" for t in selected),
        "excluded": excluded,
        "counts": {
            "entities": len(entities), "relations": len(relations),
            "triples": len(triple_rows), "evidence_chunks": len(evidence),
        },
        "asset_sha256": asset_checksums,
        "input_triples_sha256": sha256_json(triple_rows),
        "schema_sha256": sha256_json(schema_document()),
    }
    manifest["rebuild_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "generated_at_utc"}
    )

    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for filename, rows in assets.items():
            (root / filename).write_text(_jsonl(rows), encoding="utf-8")
        (root / "schema.json").write_text(
            json.dumps(schema_document(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "rebuild_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {**assets, "manifest": manifest}


build_knowledge_graph = build_kg
