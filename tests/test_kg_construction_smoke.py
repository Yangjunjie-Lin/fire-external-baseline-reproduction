from __future__ import annotations

import json

from external_baselines.ekell_style.kg_construction import (
    DECISION_DEMANDS,
    build_kg,
    extract_candidate_triples,
    parse_document,
    schema_document,
    validate_triple,
)


def test_candidate_construction_and_manifest(tmp_path) -> None:
    segments = parse_document(
        "Electrical fire requires power isolation.",
        source_id="manual-1",
    )
    triples = extract_candidate_triples(segments)
    assert triples
    assert triples[0].review_status == "candidate"
    assert triples[0].source_text == segments[0].text
    assert validate_triple(triples[0]).valid

    built = build_kg(triples, output_dir=tmp_path)
    manifest = json.loads((tmp_path / "rebuild_manifest.json").read_text(encoding="utf-8"))
    assert built["manifest"]["rebuild_sha256"] == manifest["rebuild_sha256"]
    assert manifest["candidate_count"] == 1


def test_schema_documents_eight_primary_and_twenty_two_subclasses() -> None:
    schema = schema_document()
    assert len(DECISION_DEMANDS) == schema["primary_demand_count"] == 8
    assert schema["subclass_demand_count"] == 22
    assert all(item["substituted_for_fire_domain"] for item in schema["decision_demands"])
