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
        # Legacy comma/semicolon-delimited compatibility form.
        aliases = [a.strip() for a in aliases.replace(";", ",").split(",") if a.strip()]
    if not isinstance(aliases, list):
        aliases = []
    # Exact strings only: no lossy str() coercion of numbers/objects.
    alias_strings = [a for a in aliases if type(a) is str]
    values = [entity_name(entity), entity_id(entity), entity_type(entity)] + alias_strings
    return list(dict.fromkeys(v for v in values if v and v != "None"))


def triple_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    head = _first(row, ["head", "source", "subject", "from", "h", "head_entity", "src"], "")
    rel = _first(row, ["relation", "predicate", "type", "label", "r", "edge", "relation_type"], "related_to")
    tail = _first(row, ["tail", "target", "object", "to", "t", "tail_entity", "dst"], "")
    return str(head), str(rel), str(tail)


def triple_id(row: dict[str, Any], index: int | None = None) -> str:
    value = _first(row, ["triple_id", "id", "edge_id"], "")
    if value:
        return f"{value}"
    h, r, t = triple_parts(row)
    if h or t:
        return f"{h}|{r}|{t}"
    return f"triple_{index}" if index is not None else "triple_unknown"


def triple_to_text(row: dict[str, Any]) -> str:
    h, r, t = triple_parts(row)
    evidence = _first(row, ["evidence", "description", "text", "content"], "")
    if type(evidence) is not str:
        # Exact strings only: strict-loaded records guarantee this; lenient
        # records must not be silently stringified into retrieval text.
        evidence = ""
    text = f"{h} --{r}--> {t}".strip()
    return f"{text}. {evidence}" if evidence else text


def evidence_chunk_id(row: dict[str, Any], index: int | None = None) -> str:
    value = _first(row, ["chunk_id", "id", "evidence_id", "doc_id"], "")
    return f"{value}" if value else f"chunk_{index}" if index is not None else "chunk_unknown"


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


def _contains_control_character(text: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in text)


def _exact_nonempty_text_field(
    row: dict[str, Any],
    keys: tuple[str, ...],
    *,
    label: str,
) -> str:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if type(value) is not str:
            raise ValueError(f"{label}_must_be_string")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{label}_must_be_nonempty_string")
        if value != stripped:
            raise ValueError(f"{label}_must_not_have_surrounding_whitespace")
        return value
    return ""


def _entity_identifier_field(
    row: dict[str, Any],
    keys: tuple[str, ...],
    *,
    label: str,
) -> str:
    """Return the exact identifier value without silent normalization.

    Strings are returned verbatim; surrounding whitespace and control
    characters are schema errors, never trimmed.  Exact integers use their
    stable decimal form.  ``bool``/``float``/containers are rejected.
    """
    for key in keys:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if type(value) is str:
            if not value:
                raise ValueError(f"{label}_must_be_nonempty_string")
            if value != value.strip():
                raise ValueError(f"{label}_must_not_have_surrounding_whitespace")
            if _contains_control_character(value):
                raise ValueError(f"{label}_contains_control_character")
            return value
        if type(value) is int:
            return f"{value}"
        raise ValueError(f"{label}_invalid_type")
    return ""


ALIAS_FIELD_KEYS = ("aliases", "alias", "synonyms", "keywords", "terms")


def _validate_alias_string(value: str, *, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label}_must_be_nonempty_string")
    if value != value.strip():
        raise ValueError(f"{label}_must_not_have_surrounding_whitespace")
    if _contains_control_character(value):
        raise ValueError(f"{label}_contains_control_character")


def _validate_alias_fields(row: dict[str, Any], *, prefix: str) -> None:
    """Aliases must be an exact string (legacy delimited form) or a list of exact strings."""
    for key in ALIAS_FIELD_KEYS:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if type(value) is str:
            _strict_field(
                prefix,
                _validate_alias_string,
                value,
                label="entity_alias",
            )
            # Legacy comma/semicolon-delimited compatibility form: every
            # delimited item must be non-empty after separation.
            parts = value.replace(";", ",").split(",")
            if any(not part.strip() for part in parts):
                raise ValueError(f"{prefix}:entity_alias_legacy_delimited_item_empty")
            continue
        if type(value) is list:
            for element in value:
                if type(element) is not str:
                    raise ValueError(f"{prefix}:entity_alias_must_be_string")
                _strict_field(
                    prefix,
                    _validate_alias_string,
                    element,
                    label="entity_alias",
                )
            continue
        raise ValueError(f"{prefix}:entity_alias_must_be_string")


def _validate_identifier_list_fields(
    row: dict[str, Any],
    keys: tuple[str, ...],
    *,
    prefix: str,
    label: str,
) -> None:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        values = row[key]
        if type(values) is not list:
            raise ValueError(f"{prefix}:{label}_must_be_list")
        for value in values:
            _strict_field(
                prefix,
                _entity_identifier_field,
                {key: value},
                (key,),
                label=label,
            )


TRIPLE_SEMANTIC_TEXT_KEYS = ("evidence", "description", "text", "content")


def _strict_field(prefix: str, reader, *args, **kwargs) -> str:
    try:
        return reader(*args, **kwargs)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc


def _validate_optional_text_fields(
    row: dict[str, Any],
    keys: tuple[str, ...],
    *,
    prefix: str,
    label_prefix: str = "",
) -> None:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        _strict_field(
            prefix,
            _exact_nonempty_text_field,
            row,
            (key,),
            label=f"{label_prefix}{key}",
        )


def _validate_strict_row(kind: str, row: JsonlObjectRow, *, filename: str) -> None:
    value = row.value
    prefix = f"kg_schema_invalid:{filename}:line_{row.line_no}"
    if kind == "entities":
        identifier = _strict_field(
            prefix,
            _entity_identifier_field,
            value,
            ("entity_id", "id", "uid", "node_id"),
            label="entity_identifier",
        )
        name = _strict_field(
            prefix,
            _exact_nonempty_text_field,
            value,
            ("name", "label", "entity", "text", "entity_name", "title"),
            label="entity_name",
        )
        if not identifier and not name:
            raise ValueError(f"{prefix}:entity_id_or_name_missing")
        _validate_optional_text_fields(
            value,
            ("type", "entity_type", "category", "label_type"),
            prefix=prefix,
        )
        _validate_alias_fields(value, prefix=prefix)
        _validate_identifier_list_fields(
            value,
            ("source_chunk_ids", "evidence_chunk_ids"),
            prefix=prefix,
            label="entity_evidence_reference_identifier",
        )
        return
    if kind == "relations":
        relation_identifier = _strict_field(
            prefix,
            _entity_identifier_field,
            value,
            ("relation_id", "id"),
            label="relation_identifier",
        )
        relation_name = _strict_field(
            prefix,
            _exact_nonempty_text_field,
            value,
            ("name", "label", "relation", "predicate", "type", "r", "edge", "relation_type"),
            label="relation_label",
        )
        triple_shaped = any(
            key in value
            for key in (
                "head",
                "source",
                "subject",
                "from",
                "h",
                "src",
                "tail",
                "target",
                "object",
                "to",
                "t",
                "dst",
            )
        )
        if triple_shaped:
            head = _strict_field(
                prefix,
                _entity_identifier_field,
                value,
                ("head", "source", "subject", "from", "h", "src"),
                label="relation_head_identifier",
            )
            tail = _strict_field(
                prefix,
                _entity_identifier_field,
                value,
                ("tail", "target", "object", "to", "t", "dst"),
                label="relation_tail_identifier",
            )
            if not head or not relation_name or not tail:
                raise ValueError(f"{prefix}:relation_identity_missing")
            for keys, label in (
                (("head_entity_id",), "relation_head_entity_identifier"),
                (("tail_entity_id",), "relation_tail_entity_identifier"),
            ):
                _strict_field(
                    prefix,
                    _entity_identifier_field,
                    value,
                    keys,
                    label=label,
                )
        elif not relation_identifier and not relation_name:
            raise ValueError(f"{prefix}:relation_identity_missing")
        _validate_optional_text_fields(
            value,
            ("source_id", "citation", "url", "source_url", "file", "path"),
            prefix=prefix,
        )
        if triple_shaped:
            _validate_optional_text_fields(
                value,
                TRIPLE_SEMANTIC_TEXT_KEYS,
                prefix=prefix,
                label_prefix="relation_",
            )
        return
    if kind == "triples":
        fields = {
            "head": _strict_field(
                prefix,
                _entity_identifier_field,
                value,
                ("head", "source", "subject", "from", "h", "head_entity", "src"),
                label="triple_head_identifier",
            ),
            "relation": _strict_field(
                prefix,
                _exact_nonempty_text_field,
                value,
                ("relation", "predicate", "type", "label", "r", "edge", "relation_type"),
                label="triple_relation",
            ),
            "tail": _strict_field(
                prefix,
                _entity_identifier_field,
                value,
                ("tail", "target", "object", "to", "t", "tail_entity", "dst"),
                label="triple_tail_identifier",
            ),
        }
        for field, resolved in fields.items():
            if not resolved:
                raise ValueError(f"{prefix}:triple_{field}_missing")
        _strict_field(
            prefix,
            _entity_identifier_field,
            value,
            ("triple_id", "id", "edge_id"),
            label="triple_identifier",
        )
        for keys, label in (
            (("head_entity_id",), "triple_head_entity_identifier"),
            (("relation_id",), "triple_relation_identifier"),
            (("tail_entity_id",), "triple_tail_entity_identifier"),
            (
                (
                    "chunk_id",
                    "source_chunk_id",
                    "evidence_id",
                    "doc_id",
                    "evidence_chunk_id",
                    "supporting_chunk_id",
                ),
                "triple_evidence_reference_identifier",
            ),
        ):
            _strict_field(
                prefix,
                _entity_identifier_field,
                value,
                keys,
                label=label,
            )
        _validate_identifier_list_fields(
            value,
            ("source_chunk_ids", "evidence_chunk_ids"),
            prefix=prefix,
            label="triple_evidence_reference_identifier",
        )
        _validate_optional_text_fields(
            value,
            (
                "source_id",
                "citation",
                "url",
                "source_url",
                "document_id",
                "file",
                "path",
            ),
            prefix=prefix,
        )
        _validate_optional_text_fields(
            value,
            TRIPLE_SEMANTIC_TEXT_KEYS,
            prefix=prefix,
            label_prefix="triple_",
        )
        return
    if kind == "evidence_chunks":
        chunk_id = _strict_field(
            prefix,
            _entity_identifier_field,
            value,
            ("chunk_id", "id", "evidence_id", "doc_id", "source_chunk_id"),
            label="evidence_chunk_identifier",
        )
        if not chunk_id:
            raise ValueError(f"{prefix}:evidence_chunk_id_missing")
        text = _strict_field(
            prefix,
            _exact_nonempty_text_field,
            value,
            ("text", "content", "chunk", "body", "page_content", "document"),
            label="evidence_text",
        )
        if not text:
            raise ValueError(f"{prefix}:evidence_text_missing")
        source = _strict_field(
            prefix,
            _exact_nonempty_text_field,
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
            label="evidence_source_or_citation",
        )
        if not source:
            raise ValueError(f"{prefix}:evidence_source_or_citation_missing")


PROVENANCE_TEXT_REFERENCE_KEYS = (
    "source_id",
    "citation",
    "url",
    "source_url",
    "document_id",
    "file",
    "path",
)
PROVENANCE_IDENTIFIER_REFERENCE_KEYS = (
    "chunk_id",
    "source_chunk_id",
    "evidence_id",
    "doc_id",
    "evidence_chunk_id",
    "supporting_chunk_id",
)
PROVENANCE_IDENTIFIER_LIST_REFERENCE_KEYS = (
    "source_chunk_ids",
    "evidence_chunk_ids",
)


def _provenance_reference(value: dict[str, Any]) -> str:
    """Canonicalize every declared provenance reference without coercion.

    Protocol aliases share one semantic group so changing ``chunk_id`` to
    ``source_chunk_id`` (or ``url`` to ``source_url``) does not manufacture a
    distinct provenance identity.
    """
    groups: dict[str, set[str]] = {
        "source": set(),
        "citation": set(),
        "evidence": set(),
    }
    text_groups = {
        "source_id": "source",
        "document_id": "source",
        "citation": "citation",
        "url": "citation",
        "source_url": "citation",
        "file": "citation",
        "path": "citation",
    }
    for key in PROVENANCE_TEXT_REFERENCE_KEYS:
        if key in value and value[key] is not None:
            groups[text_groups[key]].add(
                _exact_nonempty_text_field(
                    value,
                    (key,),
                    label=f"triple_{key}",
                )
            )
    for key in PROVENANCE_IDENTIFIER_REFERENCE_KEYS:
        if key in value and value[key] is not None:
            groups["evidence"].add(
                _entity_identifier_field(
                    value,
                    (key,),
                    label="triple_evidence_reference_identifier",
                )
            )
    for key in PROVENANCE_IDENTIFIER_LIST_REFERENCE_KEYS:
        if key in value and value[key] is not None:
            raw_values = value[key]
            if type(raw_values) is not list:
                raise ValueError(
                    "triple_evidence_reference_identifier_must_be_list"
                )
            canonical_values = [
                _entity_identifier_field(
                    {key: item},
                    (key,),
                    label="triple_evidence_reference_identifier",
                )
                for item in raw_values
            ]
            groups["evidence"].update(canonical_values)
    references = [
        [group, *sorted(values)]
        for group, values in groups.items()
        if values
    ]
    return json.dumps(references, ensure_ascii=False, separators=(",", ":"))


def _strict_identity(kind: str, value: dict[str, Any]) -> tuple[str, str]:
    """Return ``(identity_kind, identity)`` for duplicate detection.

    ``identity_kind`` is ``"id"`` for explicit protocol identifiers and
    ``"provenance"`` for fact identities that include the evidence reference,
    so the same fact recorded under different provenance is not a duplicate.
    """
    if kind == "entities":
        identity = _entity_identifier_field(
            value,
            ("entity_id", "id", "uid", "node_id"),
            label="entity_identifier",
        ) or _exact_nonempty_text_field(
            value,
            ("name", "label", "entity", "text", "entity_name", "title"),
            label="entity_name",
        )
        return "id", identity
    if kind == "evidence_chunks":
        return "id", _entity_identifier_field(
            value,
            ("chunk_id", "id", "evidence_id", "doc_id", "source_chunk_id"),
            label="evidence_chunk_identifier",
        )
    if kind == "relations":
        explicit = _entity_identifier_field(
            value,
            ("relation_id", "id"),
            label="relation_identifier",
        )
        if explicit:
            return "id", explicit
    if kind == "triples":
        explicit = _entity_identifier_field(
            value,
            ("triple_id", "id", "edge_id"),
            label="triple_identifier",
        )
        if explicit:
            return "id", explicit
    if kind in {"relations", "triples"}:
        head = _entity_identifier_field(
            value,
            ("head", "source", "subject", "from", "h", "head_entity", "src"),
            label="triple_head_identifier",
        )
        relation = _exact_nonempty_text_field(
            value,
            ("relation", "predicate", "type", "label", "r", "edge", "relation_type", "name"),
            label="triple_relation",
        )
        tail = _entity_identifier_field(
            value,
            ("tail", "target", "object", "to", "t", "tail_entity", "dst"),
            label="triple_tail_identifier",
        )
        provenance = _provenance_reference(value)
        if head or tail:
            return "provenance", json.dumps(
                [head, relation, tail, provenance],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return "provenance", json.dumps(
            [relation, provenance],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return "id", ""


def _reject_duplicate_identities(
    kind: str,
    rows: list[JsonlObjectRow],
    *,
    filename: str | None = None,
) -> None:
    id_error_names = {
        "entities": "kg_duplicate_entity_id",
        "relations": "kg_duplicate_relation_id",
        "triples": "kg_duplicate_triple_id",
        "evidence_chunks": "kg_duplicate_evidence_chunk_id",
    }
    provenance_error_names = {
        "relations": "kg_duplicate_relation_provenance",
        "triples": "kg_duplicate_triple_provenance",
    }
    seen: dict[tuple[str, str], int] = {}
    for row in rows:
        identity_kind, identity = _strict_identity(kind, row.value)
        key = (identity_kind, identity)
        if key in seen:
            location = (
                f":{filename}:line_{row.line_no}:first_line_{seen[key]}"
                if filename
                else f":line_{row.line_no}:first_line_{seen[key]}"
            )
            if identity_kind == "provenance":
                identity_digest = sha256_json({"identity": identity})
                try:
                    identity_parts = json.loads(identity)
                    fact_parts = identity_parts[:3]
                    fact = "|".join(str(part) for part in fact_parts)
                except (TypeError, ValueError, json.JSONDecodeError):
                    fact = kind
                raise ValueError(
                    f"{provenance_error_names[kind]}:{fact}|"
                    f"provenance_sha256={identity_digest}{location}:"
                    "field=triple_provenance"
                )
            raise ValueError(
                f"{id_error_names[kind]}:{identity}{location}:field=identifier"
            )
        seen[key] = row.line_no


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
        _reject_duplicate_identities(kind, rows, filename=filename)
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
