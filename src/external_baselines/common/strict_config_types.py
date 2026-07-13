"""Exact type helpers for formal safety-critical config fields."""

from __future__ import annotations

from typing import Any

MISSING = object()


class StrictConfigTypeError(ValueError):
    """Raised when a config value is not an exact required type."""


def exact_bool(
    value: Any = MISSING,
    *,
    field: str,
    default: Any = MISSING,
) -> bool:
    if value is MISSING:
        if default is not MISSING:
            if type(default) is not bool:
                raise StrictConfigTypeError(f"{field} default must be an exact boolean")
            return default
        raise StrictConfigTypeError(f"{field} is required")
    if value is None:
        raise StrictConfigTypeError(f"{field} must be an exact boolean")
    if type(value) is not bool:
        raise StrictConfigTypeError(f"{field} must be an exact boolean (got {value!r})")
    return value


def exact_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> int:
    if value is None:
        raise StrictConfigTypeError(f"{field} must be an exact integer")
    if type(value) is not int:
        raise StrictConfigTypeError(
            f"{field} must be an exact integer with exact YAML integer type (got {value!r})"
        )
    if minimum is not None:
        if minimum_inclusive and value < minimum:
            raise StrictConfigTypeError(f"{field} must be >= {minimum} (got {value})")
        if not minimum_inclusive and value <= minimum:
            raise StrictConfigTypeError(f"{field} must be > {minimum} (got {value})")
    if maximum is not None:
        if maximum_inclusive and value > maximum:
            raise StrictConfigTypeError(f"{field} must be <= {maximum} (got {value})")
        if not maximum_inclusive and value >= maximum:
            raise StrictConfigTypeError(f"{field} must be < {maximum} (got {value})")
    return value


def exact_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if value is None:
        raise StrictConfigTypeError(f"{field} must be an exact number")
    if type(value) not in (int, float):
        raise StrictConfigTypeError(
            f"{field} must be an exact integer or float (got {value!r})"
        )
    if type(value) is bool:
        raise StrictConfigTypeError(f"{field} must be an exact number (got {value!r})")
    number = float(value)
    if minimum is not None:
        if minimum_inclusive and number < minimum:
            raise StrictConfigTypeError(f"{field} must be >= {minimum} (got {number})")
        if not minimum_inclusive and number <= minimum:
            raise StrictConfigTypeError(f"{field} must be > {minimum} (got {number})")
    if maximum is not None:
        if maximum_inclusive and number > maximum:
            raise StrictConfigTypeError(f"{field} must be <= {maximum} (got {number})")
        if not maximum_inclusive and number >= maximum:
            raise StrictConfigTypeError(f"{field} must be < {maximum} (got {number})")
    return number


def require_exact_bool(
    value: Any = MISSING,
    *,
    field: str,
    default: Any = MISSING,
) -> bool:
    try:
        return exact_bool(value, field=field, default=default)
    except StrictConfigTypeError as exc:
        raise ValueError(str(exc)) from exc


def require_exact_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> int:
    try:
        return exact_int(
            value,
            field=field,
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=minimum_inclusive,
            maximum_inclusive=maximum_inclusive,
        )
    except StrictConfigTypeError as exc:
        raise ValueError(str(exc)) from exc


def require_exact_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    try:
        return exact_number(
            value,
            field=field,
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=minimum_inclusive,
            maximum_inclusive=maximum_inclusive,
        )
    except StrictConfigTypeError as exc:
        raise ValueError(str(exc)) from exc


def read_exact_bool(
    mapping: dict[str, Any],
    key: str,
    *,
    field: str,
    default: bool,
) -> bool:
    if key not in mapping:
        return default
    return require_exact_bool(mapping[key], field=field)


def read_exact_int(
    mapping: dict[str, Any],
    key: str,
    *,
    field: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> int:
    if key not in mapping:
        return default
    return require_exact_int(
        mapping[key],
        field=field,
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
        maximum_inclusive=maximum_inclusive,
    )


def read_exact_number(
    mapping: dict[str, Any],
    key: str,
    *,
    field: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if key not in mapping:
        return default
    return require_exact_number(
        mapping[key],
        field=field,
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
        maximum_inclusive=maximum_inclusive,
    )
