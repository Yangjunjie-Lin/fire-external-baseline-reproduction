"""Deterministic path resolution for repository and Formal-run resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PathPolicy = Literal[
    "experiment_relative",
    "repository_relative",
    "bundle_relative",
    "absolute_only",
]
ExpectedPathKind = Literal["file", "directory", "either"]


@dataclass(frozen=True)
class PathContext:
    repository_root: Path
    experiment_manifest_path: Path | None = None
    bundle_root: Path | None = None


def _validate_declared_path(declared: str | Path) -> str:
    if not isinstance(declared, (str, Path)):
        raise TypeError("declared path must be a string or Path")
    text = str(declared)
    if not text.strip():
        raise ValueError("declared_path_empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError("declared_path_contains_control_character")
    return text


def _policy_root(context: PathContext, policy: PathPolicy) -> Path:
    if policy == "repository_relative":
        return context.repository_root.resolve()
    if policy == "experiment_relative":
        if context.experiment_manifest_path is None:
            raise ValueError("experiment_relative_requires_manifest_path")
        return context.experiment_manifest_path.resolve().parent
    if policy == "bundle_relative":
        if context.bundle_root is None:
            raise ValueError("bundle_relative_requires_bundle_root")
        return context.bundle_root.resolve()
    raise ValueError(f"path policy {policy!r} has no relative root")


def _assert_within(path: Path, root: Path, *, policy: PathPolicy) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"declared_path_escapes_{policy}_root") from exc


def resolve_declared_path(
    declared: str | Path,
    *,
    context: PathContext,
    policy: PathPolicy,
    must_exist: bool = True,
    expected_kind: ExpectedPathKind = "either",
) -> Path:
    """Resolve a declared path without consulting the process working directory."""
    text = _validate_declared_path(declared)
    candidate = Path(text)
    if candidate.is_absolute():
        if policy == "bundle_relative":
            raise ValueError("bundle_relative_path_must_be_relative")
        resolved = candidate.resolve(strict=False)
    else:
        if policy == "absolute_only":
            raise ValueError("declared_path_must_be_absolute")
        root = _policy_root(context, policy)
        resolved = (root / candidate).resolve(strict=False)
        _assert_within(resolved, root, policy=policy)

    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    if expected_kind == "file" and (not resolved.is_file() or resolved.is_symlink()):
        raise ValueError(f"declared_path_not_plain_file:{resolved}")
    if expected_kind == "directory" and not resolved.is_dir():
        raise ValueError(f"declared_path_not_directory:{resolved}")
    if expected_kind not in {"file", "directory", "either"}:
        raise ValueError(f"unknown expected path kind: {expected_kind!r}")
    return resolved


def repository_relative_identity(path: str | Path, *, repository_root: Path) -> str:
    """Return a stable repository-relative identity for an already resolved path."""
    resolved_root = repository_root.resolve()
    resolved = Path(path).resolve(strict=False)
    _assert_within(resolved, resolved_root, policy="repository_relative")
    return resolved.relative_to(resolved_root).as_posix()
