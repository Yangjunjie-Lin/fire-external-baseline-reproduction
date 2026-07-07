from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from external_baselines.common.io import read_jsonl
from external_baselines.common.text_utils import normalize_text

CORPUS_FILES = {
    "entities": "entities.jsonl",
    "relations": "relations.jsonl",
    "triples": "triples.jsonl",
    "evidence_chunks": "evidence_chunks.jsonl",
}


@dataclass
class FireKG:
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    triples: list[dict[str, Any]] = field(default_factory=list)
    evidence_chunks: list[dict[str, Any]] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"entities": len(self.entities), "relations": len(self.relations), "triples": len(self.triples), "evidence_chunks": len(self.evidence_chunks)}


def _first(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def entity_name(entity: dict[str, Any]) -> str:
    return str(_first(entity, ["name", "label", "text", "entity", "entity_name", "title", "entity_id", "id"], ""))


def entity_id(entity: dict[str, Any]) -> str:
    return str(_first(entity, ["entity_id", "id", "uid", "node_id", "name", "label"], entity_name(entity)))


def entity_type(entity: dict[str, Any]) -> str:
    return str(_first(entity, ["type", "entity_type", "category", "label_type"], ""))


def entity_aliases(entity: dict[str, Any]) -> list[str]:
    aliases = _first(entity, ["aliases", "alias", "synonyms", "keywords", "terms"], [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.replace(";", ",").split(",") if a.strip()]
    if not isinstance(aliases, list):
        aliases = []
    values = [entity_name(entity), entity_id(entity), entity_type(entity)] + [str(a) for a in aliases]
    return list(dict.fromkeys(v for v in values if v and v != "None"))


def triple_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    head = _first(row, ["head", "source", "subject", "from", "h", "head_entity", "src"], "")
    rel = _first(row, ["relation", "predicate", "type", "label", "r", "edge", "relation_type"], "related_to")
    tail = _first(row, ["tail", "target", "object", "to", "t", "tail_entity", "dst"], "")
    return str(head), str(rel), str(tail)


def triple_id(row: dict[str, Any], index: int | None = None) -> str:
    value = _first(row, ["triple_id", "id", "edge_id", "relation_id"], "")
    if value:
        return str(value)
    h, r, t = triple_parts(row)
    if h or t:
        return f"{h}|{r}|{t}"
    return f"triple_{index}" if index is not None else "triple_unknown"


def triple_to_text(row: dict[str, Any]) -> str:
    h, r, t = triple_parts(row)
    evidence = _first(row, ["evidence", "description", "text", "content"], "")
    text = f"{h} --{r}--> {t}".strip()
    return f"{text}. {evidence}" if evidence else text


def evidence_chunk_id(row: dict[str, Any], index: int | None = None) -> str:
    value = _first(row, ["chunk_id", "id", "evidence_id", "doc_id"], "")
    return str(value) if value else f"chunk_{index}" if index is not None else "chunk_unknown"


def evidence_text(row: dict[str, Any]) -> str:
    return str(_first(row, ["text", "content", "chunk", "body", "page_content", "document"], ""))


def evidence_source_id(row: dict[str, Any]) -> str:
    return str(_first(row, ["source_id", "source", "doc_id", "document_id", "file", "path"], "evidence_chunk"))


def evidence_citation(row: dict[str, Any]) -> str:
    return str(_first(row, ["citation", "url", "source_url", "source_id", "chunk_id", "id"], evidence_source_id(row)))


def _schema_warnings(kind: str, rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    sample = rows[:25]
    if kind == "entities":
        for i, row in enumerate(sample):
            if not entity_name(row):
                warnings.append(f"entities row {i} has no recognizable name/label/entity field")
    elif kind in {"relations", "triples"}:
        for i, row in enumerate(sample):
            h, r, t = triple_parts(row)
            if not h or not t:
                warnings.append(f"{kind} row {i} has incomplete head/tail fields")
            if not r:
                warnings.append(f"{kind} row {i} has no recognizable relation/predicate field")
    elif kind == "evidence_chunks":
        for i, row in enumerate(sample):
            if not evidence_text(row):
                warnings.append(f"evidence_chunks row {i} has no recognizable text/content field")
    return warnings


def load_asset(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def load_kg(corpus_dir: str | Path, *, require_any: bool = False) -> FireKG:
    corpus_dir = Path(corpus_dir)
    missing: list[str] = []
    data: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    for kind, filename in CORPUS_FILES.items():
        path = corpus_dir / filename
        if not path.exists():
            missing.append(filename)
            data[kind] = []
            continue
        rows = load_asset(path)
        data[kind] = rows
        warnings.extend(_schema_warnings(kind, rows))
    kg = FireKG(entities=data["entities"], relations=data["relations"], triples=data["triples"], evidence_chunks=data["evidence_chunks"], missing_files=missing, schema_warnings=warnings)
    if require_any and sum(kg.counts().values()) == 0:
        raise FileNotFoundError(f"No corpus assets found in {corpus_dir}")
    return kg


def audit_corpus(corpus_dir: str | Path) -> dict[str, Any]:
    kg = load_kg(corpus_dir)
    counts = kg.counts()
    return {"entity_count": counts["entities"], "relation_count": counts["relations"], "triple_count": counts["triples"], "evidence_chunk_count": counts["evidence_chunks"], "missing_files": kg.missing_files, "schema_warnings": kg.schema_warnings}


def normalized_entity_index(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for e in entities:
        for value in entity_aliases(e):
            norm = normalize_text(value)
            if norm:
                index[norm] = e
    return index
