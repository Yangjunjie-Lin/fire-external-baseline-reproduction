"""No-network Draft 2020-12 validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when a contract or handoff record is invalid."""


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"invalid_json:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise SchemaValidationError(f"json_root_must_be_object:{path}")
    return value


def check_draft202012_schema(schema: dict[str, Any]) -> None:
    if schema.get("$schema") not in {
        "https://json-schema.org/draft/2020-12/schema",
        "https://json-schema.org/draft/2020-12/schema#",
    }:
        raise SchemaValidationError("schema_must_declare_draft_2020_12")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema exposes several schema error classes
        raise SchemaValidationError(f"invalid_draft_2020_12_schema:{exc}") from exc


def validation_errors(instance: Any, schema: dict[str, Any]) -> list[str]:
    check_draft202012_schema(schema)
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))
    ]


def validate_or_raise(instance: Any, schema: dict[str, Any], *, subject: str) -> None:
    errors = validation_errors(instance, schema)
    if errors:
        raise SchemaValidationError(f"{subject}: " + "; ".join(errors))
