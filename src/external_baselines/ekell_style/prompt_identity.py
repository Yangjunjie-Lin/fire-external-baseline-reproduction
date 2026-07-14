"""Authoritative identity for the controlled E-KELL prompt bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from external_baselines.common.checksums import sha256_file
from external_baselines.common.path_resolution import (
    PathContext,
    resolve_path_reference,
)
from external_baselines.common.strict_config_types import require_exact_nonempty_string

EKELL_REQUIRED_PROMPTS = (
    "stepwise_projection.txt",
    "stepwise_intersection.txt",
    "stepwise_union.txt",
    "stepwise_negation.txt",
    "final_kg_grounded_response.txt",
)


def validate_and_hash_prompt_bundle(
    prompt_dir: str | Path,
    *,
    path_context: PathContext,
) -> dict[str, Any]:
    """Validate required prompts and hash every plain file in the prompt tree."""
    declared_value = str(prompt_dir) if isinstance(prompt_dir, Path) else prompt_dir
    declared = require_exact_nonempty_string(
        declared_value,
        field="ekell_style.prompt_dir",
    )
    reference = resolve_path_reference(
        declared,
        context=path_context,
        policy="repository_relative",
        expected_kind="directory",
        allow_external_absolute=True,
    )
    root = reference.resolved_path
    policy = reference.path_policy
    if root.is_symlink():
        raise ValueError("ekell_prompt_dir_must_not_be_symlink")

    required_hashes: dict[str, str] = {}
    for name in EKELL_REQUIRED_PROMPTS:
        path = root / name
        if not path.exists():
            raise ValueError(f"ekell_prompt_missing:{name}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"ekell_prompt_not_plain_file:{name}")
        if path.stat().st_size <= 0:
            raise ValueError(f"ekell_prompt_empty:{name}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"ekell_prompt_not_utf8:{name}") from exc
        if not text.strip():
            raise ValueError(f"ekell_prompt_empty:{name}")
        required_hashes[name] = sha256_file(path)

    tree_entries: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(
                f"ekell_prompt_tree_symlink_forbidden:{path.relative_to(root).as_posix()}"
            )
        if path.is_file():
            tree_entries.append(
                f"{path.relative_to(root).as_posix()}:{sha256_file(path)}"
            )
    tree_sha = hashlib.sha256("\n".join(tree_entries).encode("utf-8")).hexdigest()

    canonical = reference.canonical_path
    return {
        "declared_prompt_dir": declared.replace("\\", "/"),
        "canonical_prompt_dir": canonical,
        "resolved_prompt_dir": str(root),
        "resolved_prompt_dir_authoritative": False,
        "path_policy": policy,
        "external": reference.external,
        "portable": not reference.external,
        "prompt_tree_sha256": tree_sha,
        "required_prompt_files": required_hashes,
    }
