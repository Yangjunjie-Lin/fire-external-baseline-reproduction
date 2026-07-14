"""Deterministic path resolution for repository and Formal-run resources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PathPolicy = Literal[
    "experiment_relative",
    "repository_relative",
    "bundle_relative",
    "absolute_only",
    "absolute_external",
]
ExpectedPathKind = Literal["file", "directory", "either"]


@dataclass(frozen=True)
class PathContext:
    repository_root: Path
    experiment_manifest_path: Path | None = None
    bundle_root: Path | None = None


@dataclass(frozen=True)
class ResolvedPathReference:
    """One authoritative interpretation of a declared resource path.

    ``resolved_path`` is machine-local read/write state.  Formal identity is
    the pair ``(path_policy, canonical_path)`` and never the absolute path.
    """

    declared_path: str
    resolved_path: Path
    path_policy: str
    canonical_path: str
    authoritative_path: str
    external: bool = False

    def to_dict(self, *, include_resolved: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "declared_path": self.declared_path,
            "path_policy": self.path_policy,
            "canonical_path": self.canonical_path,
            "authoritative_path": self.authoritative_path,
            "external": self.external,
        }
        if include_resolved:
            payload["resolved_path"] = str(self.resolved_path)
            payload["resolved_path_authoritative"] = False
        return payload


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


def classify_absolute_reference(
    resolved: Path,
    *,
    context: PathContext,
) -> tuple[PathPolicy, str] | None:
    """Classify a fully resolved absolute path against declared internal roots.

    Priority is fixed and CWD-independent: repository root first, then the
    experiment-manifest directory, then the Runner Bundle root.  ``resolved``
    must already be symlink-resolved, so symlinks cannot smuggle an external
    target into an internal identity (or vice versa).  Returns ``None`` when
    the path is outside every declared root.
    """
    candidates: list[tuple[PathPolicy, Path]] = [
        ("repository_relative", context.repository_root.resolve()),
    ]
    if context.experiment_manifest_path is not None:
        candidates.append(
            ("experiment_relative", context.experiment_manifest_path.resolve().parent)
        )
    if context.bundle_root is not None:
        candidates.append(("bundle_relative", context.bundle_root.resolve()))
    for policy, root in candidates:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return policy, relative.as_posix()
    return None


def _internal_roots(context: PathContext) -> list[tuple[PathPolicy, Path, Path]]:
    """Return ``(policy, lexical_root, resolved_root)`` in classification order."""
    roots: list[tuple[PathPolicy, Path]] = [
        ("repository_relative", context.repository_root),
    ]
    if context.experiment_manifest_path is not None:
        roots.append(("experiment_relative", context.experiment_manifest_path.parent))
    if context.bundle_root is not None:
        roots.append(("bundle_relative", context.bundle_root))
    return [
        (
            policy,
            Path(os.path.abspath(root)),
            root.resolve(),
        )
        for policy, root in roots
    ]


def _reject_absolute_symlink_escape(
    declared: Path,
    resolved: Path,
    *,
    context: PathContext,
) -> None:
    """Reject a lexically internal absolute path whose resolved target escapes.

    Classifying such a path as merely external would allow a symlink beneath an
    internal root to bypass the root boundary.  The diagnostic intentionally
    names only the policy, not either machine-local path.
    """
    lexical = Path(os.path.abspath(declared))
    for policy, lexical_root, resolved_root in _internal_roots(context):
        try:
            lexical.relative_to(lexical_root)
        except ValueError:
            continue
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                "internal_absolute_path_classification_failed:"
                f"declared_path_escapes_{policy}_root"
            ) from exc


def resolve_declared_path(
    declared: str | Path,
    *,
    context: PathContext,
    policy: PathPolicy,
    must_exist: bool = True,
    expected_kind: ExpectedPathKind = "either",
    allow_external_absolute: bool = True,
) -> Path:
    """Compatibility wrapper returning only the machine-local resolved path."""
    return resolve_path_reference(
        declared,
        context=context,
        policy=policy,
        must_exist=must_exist,
        expected_kind=expected_kind,
        allow_external_absolute=allow_external_absolute,
    ).resolved_path


def resolve_path_reference(
    declared: str | Path,
    *,
    context: PathContext,
    policy: PathPolicy,
    must_exist: bool = True,
    expected_kind: ExpectedPathKind = "either",
    allow_external_absolute: bool = False,
) -> ResolvedPathReference:
    """Resolve once under an explicit policy without consulting the CWD."""
    text = _validate_declared_path(declared)
    candidate = Path(text)
    if candidate.is_absolute():
        if policy not in {
            "absolute_only",
            "absolute_external",
            "repository_relative",
            "experiment_relative",
            "bundle_relative",
        }:
            raise ValueError(f"unknown path policy: {policy!r}")
        resolved = candidate.resolve(strict=False)
        _reject_absolute_symlink_escape(candidate, resolved, context=context)
        internal = classify_absolute_reference(resolved, context=context)
        if internal is not None:
            # An absolute path inside a declared root is not external: its
            # authoritative identity is the internal relative canonical path.
            resolved_policy, canonical = internal
            external = False
        else:
            if policy in {
                "repository_relative",
                "experiment_relative",
                "bundle_relative",
            } and not allow_external_absolute:
                raise ValueError(f"absolute_path_not_allowed_for_{policy}")
            resolved_policy = "absolute_external"
            canonical = resolved.as_posix()
            external = True
    else:
        if policy in {"absolute_only", "absolute_external"}:
            raise ValueError("declared_path_must_be_absolute")
        root = _policy_root(context, policy)
        resolved = (root / candidate).resolve(strict=False)
        _assert_within(resolved, root, policy=policy)
        resolved_policy = policy
        canonical = resolved.relative_to(root).as_posix()
        external = False

    if expected_kind not in {"file", "directory", "either"}:
        raise ValueError(f"unknown expected path kind: {expected_kind!r}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    if expected_kind == "file" and (not resolved.is_file() or resolved.is_symlink()):
        raise ValueError(f"declared_path_not_plain_file:{resolved}")
    if expected_kind == "directory" and not resolved.is_dir():
        raise ValueError(f"declared_path_not_directory:{resolved}")
    return ResolvedPathReference(
        declared_path=text,
        resolved_path=resolved,
        path_policy=resolved_policy,
        canonical_path=canonical,
        authoritative_path=canonical,
        external=external,
    )


def repository_relative_identity(path: str | Path, *, repository_root: Path) -> str:
    """Return a stable repository-relative identity for an already resolved path."""
    resolved_root = repository_root.resolve()
    resolved = Path(path).resolve(strict=False)
    _assert_within(resolved, resolved_root, policy="repository_relative")
    return resolved.relative_to(resolved_root).as_posix()
