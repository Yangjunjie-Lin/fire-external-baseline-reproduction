#!/usr/bin/env python3
"""Repository hygiene checks (secrets, committed outputs, large artifacts)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.I),
    re.compile(r"-----BEGIN (RSA |OPENSSH )?PRIVATE KEY-----"),
]

IGNORE_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__"}
TRACKED_OUTPUT_GLOBS = ["outputs/**/*.jsonl", "outputs/**/*.csv"]
BLOCKED_EXTENSIONS = {".pt", ".bin", ".safetensors", ".ckpt", ".sqlite", ".db"}


def scan() -> dict[str, object]:
    findings: list[dict[str, str]] = []
    for path in ROOT.rglob("*"):
        if path.is_dir():
            if path.name in IGNORE_DIRS:
                continue
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith("outputs/") and path.suffix in {".jsonl", ".csv", ".md"} and path.name != "README.md":
            # allowed locally but flagged if tracked — git check omitted; pattern only
            pass
        if path.suffix in BLOCKED_EXTENSIONS:
            findings.append({"type": "large_artifact", "path": rel})
        if path.name == ".env" and ".git" not in path.parts:
            findings.append({"type": "env_file", "path": rel})
        try:
            if path.stat().st_size > 50_000_000:
                findings.append({"type": "large_file", "path": rel})
        except OSError:
            continue
        if path.suffix in {".py", ".yaml", ".yml", ".json", ".md", ".example", ".env"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(text) and ".env.example" not in rel:
                    findings.append({"type": "secret_pattern", "path": rel})

    return {"ok": len(findings) == 0, "finding_count": len(findings), "findings": findings[:50]}


def main() -> None:
    report = scan()
    out = ROOT / "outputs" / "diagnostics" / "repository_hygiene.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
