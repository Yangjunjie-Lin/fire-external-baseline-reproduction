#!/usr/bin/env python3
"""Export environment snapshot (tool only; not a formal experiment record)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _installed_packages() -> dict[str, str]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        deps = {}
        for line in out.splitlines():
            if "==" in line:
                name, ver = line.split("==", 1)
                deps[name.lower()] = ver
        return deps
    except Exception:
        return {}


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from external_baselines import (  # noqa: WPS433
        __version__,
        artifact_format_version,
        interop_protocol_version,
        method_registry_version,
    )
    from external_baselines.interop.taxonomy import TAXONOMY_VERSION

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
        "interop_protocol_version": interop_protocol_version,
        "method_registry_version": method_registry_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "artifact_format_version": artifact_format_version,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--short"])),
        "dependencies": _installed_packages(),
        "optional_backends": {
            "text2vec": "text2vec" in _installed_packages(),
            "openai": "openai" in _installed_packages(),
        },
        "cuda": None,
    }
    out = ROOT / "outputs" / "diagnostics" / "environment_snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out)}, indent=2))


if __name__ == "__main__":
    main()
