"""Cross-platform containment checks for manifest-declared artifact paths."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


class ManifestArtifactPathError(ValueError):
    """Raised when a manifest artifact path cannot resolve inside the run root."""


def validate_manifest_relative_path(manifest_path: str) -> str:
    raw = str(manifest_path or "").strip()
    if not raw:
        raise ManifestArtifactPathError("manifest_artifact_path_empty")
    if "\x00" in raw:
        raise ManifestArtifactPathError("manifest_artifact_path_traversal")

    lowered = raw.casefold()
    if lowered.startswith(("\\\\?\\", "\\\\.\\", "//?/", "//./")):
        raise ManifestArtifactPathError("manifest_artifact_path_device_namespace")

    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)

    if posix.is_absolute():
        raise ManifestArtifactPathError("manifest_artifact_path_absolute")
    if windows.is_absolute():
        raise ManifestArtifactPathError("manifest_artifact_path_absolute")
    if windows.drive:
        raise ManifestArtifactPathError("manifest_artifact_path_drive_qualified")
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise ManifestArtifactPathError("manifest_artifact_path_unc")
    if ".." in posix.parts or ".." in windows.parts:
        raise ManifestArtifactPathError("manifest_artifact_path_traversal")

    return raw


def resolve_manifest_artifact_path(run_root: Path, manifest_path: str) -> Path:
    raw = validate_manifest_relative_path(manifest_path)
    root = run_root.resolve()
    candidate = (root / Path(raw)).resolve()
    if candidate == root:
        raise ManifestArtifactPathError("manifest_artifact_path_outside_run_root")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestArtifactPathError("manifest_artifact_path_outside_run_root") from exc
    return candidate
