"""Deterministic ZIP packaging for a validated handoff directory."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from external_baselines.interop.deepeval_handoff.constants import BUNDLE_VERSION
from external_baselines.interop.deepeval_handoff.manifest import sha256_file, write_json
from external_baselines.interop.deepeval_handoff.schema_validation import load_json_object, validate_or_raise
from external_baselines.interop.deepeval_handoff.validator import validate_handoff


def package_handoff(*, handoff: Path, main_repo: Path, archive: Path) -> dict[str, Any]:
    report = validate_handoff(handoff, main_repo=main_repo, write_report=True)
    if not report["ok"]:
        raise ValueError("handoff_validation_failed:" + ";".join(report["errors"]))
    files = sorted(path for path in handoff.rglob("*") if path.is_file())
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(handoff).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes())
    metadata = {
        "schema_version": BUNDLE_VERSION,
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "handoff_manifest_sha256": sha256_file(handoff / "handoff_manifest.json"),
        "file_count": len(files),
    }
    schema = load_json_object(
        Path(__file__).resolve().parents[4] / "schemas/deepeval_handoff/deepeval_handoff_bundle_v1.schema.json"
    )
    validate_or_raise(metadata, schema, subject="handoff_bundle")
    write_json(archive.with_suffix(archive.suffix + ".manifest.json"), metadata)
    return metadata
