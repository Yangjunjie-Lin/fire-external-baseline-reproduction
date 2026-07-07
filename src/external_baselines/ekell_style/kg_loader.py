from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from external_baselines.common.io import read_jsonl
from external_baselines.common.text_utils import normalize_text


@dataclass
class FireKG:
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    triples: list[dict[str, Any]]
    evidence_chunks: list[dict[str, Any]]


def entity_name(entity: dict[str, Any]) -> str:
    return str(entity.get("name") or entity.get("label") or entity.get("text") or entity.get("entity") or entity.get("entity_id") or "")


def entity_id(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_id") or entity.get("id") or entity_name(entity))


def entity_aliases(entity: dict[str, Any]) -> list[str]:
    aliases = entity.get("aliases") or entity.get("alias") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    values = [entity_name(entity)] + [str(a) for a in aliases]
    return [v for v in values if v]


def triple_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    head = row.get("head") or row.get("source") or row.get("subject") or row.get("from") or row.get("h") or ""
    rel = row.get("relation") or row.get("predicate") or row.get("type") or row.get("label") or row.get("r") or "related_to"
    tail = row.get("tail") or row.get("target") or row.get("object") or row.get("to") or row.get("t") or ""
    return str(head), str(rel), str(tail)


def triple_to_text(row: dict[str, Any]) -> str:
    h, r, t = triple_parts(row)
    return f"{h} --{r}--> {t}"


def load_kg(corpus_dir: str | Path) -> FireKG:
    corpus_dir = Path(corpus_dir)
    return FireKG(
        entities=read_jsonl(corpus_dir / "entities.jsonl"),
        relations=read_jsonl(corpus_dir / "relations.jsonl"),
        triples=read_jsonl(corpus_dir / "triples.jsonl"),
        evidence_chunks=read_jsonl(corpus_dir / "evidence_chunks.jsonl"),
    )


def normalized_entity_index(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for e in entities:
        for value in entity_aliases(e):
            norm = normalize_text(value)
            if norm:
                index[norm] = e
    return index
