from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_json
from external_baselines.common.io import read_jsonl
from external_baselines.common.text_utils import normalize_text

CORPUS_FILES = {
    "entities": "entities.jsonl",
    "relations": "relations.jsonl",
    "triples": "triples.jsonl",
    "evidence_chunks": "evidence_chunks.jsonl",
}


@dataclass(frozen=True)
class JsonlObjectRow:
    line_no: int
    value: dict[str, Any]


@dataclass
class FireKG:
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    triples: list[dict[str, Any]] = field(default_factory=list)
    evidence_chunks: list[dict[str, Any]] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "triples": len(self.triples),
            "evidence_chunks": len(self.evidence_chunks),
        }


def canonical_fire_kg_payload(kg: FireKG) -> dict[str, Any]:
    """Canonical FireKG content shared by build, freeze, and Formal preflight."""
    if not isinstance(kg, FireKG):
        raise TypeError("canonical_fire_kg_payload requires FireKG")
    return {
        "entities": kg.entities,
        "relations": kg.relations,
        "triples": kg.triples,
        "evidence_chunks": kg.evidence_chunks,
    }


def fire_kg_checksum(kg: FireKG) -> str:
    return sha256_json(canonical_fire_kg_payload(kg))


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


def read_jsonl_object_records_strict(
    path: str | Path,
    *,
    require_nonempty: bool,
) -> list[JsonlObjectRow]:
    """Read every non-empty JSONL line and require an object record."""
    source = Path(path)
    filename = source.name
    if not source.exists():
        raise ValueError(f"kg_jsonl_missing:{filename}")
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"kg_jsonl_not_plain_file:{filename}")
    rows: list[JsonlObjectRow] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"kg_jsonl_invalid_json:{filename}:line_{line_no}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"kg_jsonl_record_must_be_object:{filename}:line_{line_no}"
                    )
                rows.append(JsonlObjectRow(line_no=line_no, value=value))
    except UnicodeDecodeError as exc:
        raise ValueError(f"kg_jsonl_not_utf8:{filename}:line_{exc.start}") from exc
    if require_nonempty and not rows:
        raise ValueError(f"kg_jsonl_empty:{filename}")
    return rows


def _nonempty_field(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if type(value) is str and value.strip():
            return value.strip()
        if value is not None and type(value) in (int, float):
            return str(value)
    return ""


def _validate_strict_row(kind: str, row: JsonlObjectRow, *, filename: str) -> None:
    value = row.value
    prefix = f"kg_schema_invalid:{filename}:line_{row.line_no}"
    if kind == "entities":
        if not _nonempty_field(
            value,
            ("entity_id", "id", "uid", "node_id", "name", "label", "entity"),
        ):
            raise ValueError(f"{prefix}:entity_id_or_name_missing")
        return
    if kind == "relations":
        relation_identity = _nonempty_field(
            value,
            ("relation_id", "id", "name", "label", "relation", "predicate", "type"),
        )
        head = _nonempty_field(value, ("head", "source", "subject", "from", "h", "src"))
        relation = _nonempty_field(
            value,
            ("relation", "predicate", "type", "label", "r", "edge", "relation_type"),
        )
        tail = _nonempty_field(value, ("tail", "target", "object", "to", "t", "dst"))
        if not relation_identity and not (head and relation and tail):
            raise ValueError(f"{prefix}:relation_identity_missing")
        return
    if kind == "triples":
        fields = {
            "head": _nonempty_field(
                value,
                ("head", "source", "subject", "from", "h", "head_entity", "src"),
            ),
            "relation": _nonempty_field(
                value,
                ("relation", "predicate", "type", "label", "r", "edge", "relation_type"),
            ),
            "tail": _nonempty_field(
                value,
                ("tail", "target", "object", "to", "t", "tail_entity", "dst"),
            ),
        }
        for field, resolved in fields.items():
            if not resolved:
                raise ValueError(f"{prefix}:triple_{field}_missing")
        return
    if kind == "evidence_chunks":
        if not _nonempty_field(value, ("chunk_id", "id", "evidence_id", "doc_id")):
            raise ValueError(f"{prefix}:evidence_chunk_id_missing")
        if not _nonempty_field(
            value,
            ("text", "content", "chunk", "body", "page_content", "document"),
        ):
            raise ValueError(f"{prefix}:evidence_text_missing")
        if not _nonempty_field(
            value,
            (
                "source_id",
                "source",
                "citation",
                "url",
                "source_url",
                "document_id",
                "file",
                "path",
            ),
        ):
            raise ValueError(f"{prefix}:evidence_source_or_citation_missing")


def load_kg_strict(
    corpus_dir: str | Path,
    *,
    require_entities: bool = True,
    require_relations: bool = True,
    require_triples: bool = True,
    require_evidence_chunks: bool = True,
) -> FireKG:
    """Load the complete official FireKG contract without lossy JSONL parsing."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise ValueError(f"kg_corpus_dir_missing:{root}")
    requirements = {
        "entities": require_entities,
        "relations": require_relations,
        "triples": require_triples,
        "evidence_chunks": require_evidence_chunks,
    }
    data: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for kind, filename in CORPUS_FILES.items():
        path = root / filename
        required = requirements[kind]
        if not required and not path.exists():
            data[kind] = []
            missing.append(filename)
            continue
        rows = read_jsonl_object_records_strict(path, require_nonempty=required)
        for row in rows:
            _validate_strict_row(kind, row, filename=filename)
        data[kind] = [row.value for row in rows]
    return FireKG(
        entities=data["entities"],
        relations=data["relations"],
        triples=data["triples"],
        evidence_chunks=data["evidence_chunks"],
        missing_files=missing,
        schema_warnings=[],
    )


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
    kg = FireKG(
        entities=data["entities"],
        relations=data["relations"],
        triples=data["triples"],
        evidence_chunks=data["evidence_chunks"],
        missing_files=missing,
        schema_warnings=warnings,
    )
    if require_any and sum(kg.counts().values()) == 0:
        raise FileNotFoundError(f"No corpus assets found in {corpus_dir}")
    return kg


def audit_corpus(corpus_dir: str | Path) -> dict[str, Any]:
    kg = load_kg(corpus_dir)
    counts = kg.counts()
    return {
        "entity_count": counts["entities"],
        "relation_count": counts["relations"],
        "triple_count": counts["triples"],
        "evidence_chunk_count": counts["evidence_chunks"],
        "missing_files": kg.missing_files,
        "schema_warnings": kg.schema_warnings,
    }


def normalized_entity_index(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for e in entities:
        for value in entity_aliases(e):
            norm = normalize_text(value)
            if norm:
                index[norm] = e
    return index
