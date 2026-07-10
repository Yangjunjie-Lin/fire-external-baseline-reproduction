#!/usr/bin/env python3
"""Package reproducibility artifact (dry-run supported; no secrets)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _redact_config(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    for key in list(data.keys()):
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            data[key] = "REDACTED"
    llm = data.get("llm")
    if isinstance(llm, dict):
        for k in list(llm.keys()):
            if "key" in k.lower() or "token" in k.lower():
                llm[k] = "REDACTED"
    return data


def package_artifact(
    *,
    experiment_manifest: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    from external_baselines import (
        __version__,
        artifact_format_version,
        interop_protocol_version,
        method_registry_version,
    )
    from external_baselines.common.experiment_manifest import load_experiment_manifest
    from external_baselines.method_registry import canonicalize_method_id

    manifest = load_experiment_manifest(experiment_manifest)
    method_ids = [canonicalize_method_id(str(m["method_id"])) for m in manifest["methods"] if m.get("enabled", True)]

    artifact_status = "dry_run_scaffold" if dry_run else "packaged_scaffold"
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_format_version": artifact_format_version,
        "package_version": __version__,
        "interop_protocol_version": interop_protocol_version,
        "method_registry_version": method_registry_version,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--short"])),
        "experiment_id": manifest.get("experiment_id"),
        "method_ids": method_ids,
        "artifact_status": artifact_status,
        "dry_run": dry_run,
        "predictions_present": False,
        "metrics_present": False,
    }

    if dry_run:
        return {"manifest": meta, "output_dir": str(output_dir), "dry_run": True}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "predictions" / "README.md").write_text(
        "No formal predictions packaged in scaffold dry-run.\n", encoding="utf-8"
    )
    (output_dir / "metrics" / "README.md").write_text(
        "No formal metrics packaged; use fire-agent-demo shared evaluator.\n", encoding="utf-8"
    )
    (output_dir / "code" / "git_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "code" / "git_metadata.json").write_text(
        json.dumps({"commit": meta["git_commit"], "dirty": meta["git_dirty"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    cfg_dir = output_dir / "configs"
    cfg_dir.mkdir(exist_ok=True)
    shutil.copy2(experiment_manifest, cfg_dir / "experiment_manifest.yaml")
    shared = ROOT / str(manifest["shared_model_config"])
    if shared.is_file():
        import yaml

        (cfg_dir / "shared_model_config.redacted.yaml").write_text(
            yaml.safe_dump(_redact_config(shared), sort_keys=False), encoding="utf-8"
        )

    checksums: dict[str, str] = {}
    for path in output_dir.rglob("*"):
        if path.is_file():
            rel = str(path.relative_to(output_dir)).replace("\\", "/")
            checksums[rel] = _sha256_file(path)

    (output_dir / "MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    lines = [f"{digest}  {name}" for name, digest in sorted(checksums.items())]
    (output_dir / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Reproducibility artifact scaffold\n\nNot a paper-final result package.\n",
        encoding="utf-8",
    )
    return {"manifest": meta, "output_dir": str(output_dir), "file_count": len(checksums)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Package reproducibility artifact (scaffold).")
    parser.add_argument(
        "--experiment-manifest",
        default="configs/experiments/controlled_main_table_v1.yaml.example",
    )
    parser.add_argument("--output", default="outputs/diagnostics/artifact_dry_run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = package_artifact(
        experiment_manifest=ROOT / args.experiment_manifest,
        output_dir=ROOT / args.output,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
