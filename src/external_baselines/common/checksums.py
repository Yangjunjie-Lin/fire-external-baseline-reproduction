from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(payload)


def prompt_hash(system: str, user: str) -> str:
    return sha256_text(f"SYSTEM\n{system}\nUSER\n{user}")


def directory_manifest(path: str | Path, *, patterns: tuple[str, ...] = ("*.json", "*.jsonl", "*.yaml", "*.yml", "*.txt")) -> dict[str, Any]:
    root = Path(path)
    files: list[dict[str, Any]] = []
    if root.exists():
        for pattern in patterns:
            for fp in sorted(root.rglob(pattern)):
                if fp.is_file():
                    files.append({
                        "path": str(fp.relative_to(root)).replace("\\", "/"),
                        "sha256": sha256_file(fp),
                        "bytes": fp.stat().st_size,
                    })
    return {
        "root": str(root),
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": sha256_json([f["path"] + ":" + (f["sha256"] or "") for f in files]),
    }
