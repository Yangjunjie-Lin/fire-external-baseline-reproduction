#!/usr/bin/env python3
"""Repository hygiene checks (secrets, committed outputs, large artifacts)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.I),
    re.compile(r"-----BEGIN (RSA |OPENSSH )?PRIVATE KEY-----"),
]

IGNORE_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "node_modules"}
BLOCKED_EXTENSIONS = {".pt", ".bin", ".safetensors", ".ckpt", ".sqlite", ".db"}
BUILD_ARTIFACT_PREFIXES = ("dist/", "build/")
BUILD_ARTIFACT_SUFFIXES = (".whl",)
LOCAL_PATH_PATTERNS = [
    re.compile(r"C:\\Users\\", re.I),
    re.compile(r"/home/", re.I),
    re.compile(r"AppData\\Local\\Temp", re.I),
    re.compile(r"pytest-of-", re.I),
]
OUTPUTS_ALLOWED = {"outputs/README.md", "outputs/.gitkeep"}


def _tracked_paths() -> set[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {line.replace("\\", "/") for line in out.splitlines() if line.strip()}
    except Exception:
        return set()


def _is_build_artifact(rel: str) -> bool:
    if any(rel.startswith(prefix) for prefix in BUILD_ARTIFACT_PREFIXES):
        return True
    if rel.endswith(BUILD_ARTIFACT_SUFFIXES):
        return True
    # Source tarballs under dist/ only; avoid fixtures named *.tar.gz elsewhere.
    if rel.startswith("dist/") and rel.endswith(".tar.gz"):
        return True
    return False


def scan() -> dict[str, object]:
    findings: list[dict[str, str]] = []
    tracked = _tracked_paths()
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith("outputs/") and rel not in OUTPUTS_ALLOWED and rel in tracked:
            findings.append({"type": "tracked_generated_output", "path": rel})
        if _is_build_artifact(rel) and rel in tracked:
            findings.append({"type": "build_artifact", "path": rel})
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
            if rel.endswith(".json") or rel.endswith(".jsonl"):
                for pattern in LOCAL_PATH_PATTERNS:
                    if pattern.search(text):
                        if rel.startswith("outputs/") and rel not in tracked:
                            break
                        findings.append({"type": "local_temp_path_in_fixture", "path": rel})
                        break

    # Deduplicate
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for item in findings:
        key = (item["type"], item["path"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return {"ok": len(unique) == 0, "finding_count": len(unique), "findings": unique[:50]}


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
