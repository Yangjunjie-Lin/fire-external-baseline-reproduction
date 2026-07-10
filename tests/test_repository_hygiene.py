"""Repository hygiene and build-artifact ignore tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_contains_build_artifacts() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "dist/" in text
    assert "build/" in text
    assert "*.egg-info/" in text


def test_repository_hygiene_rejects_dist_artifacts(tmp_path: Path, monkeypatch) -> None:
    import scripts.check_repository_hygiene as hygiene

    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    dist = fake_root / "dist"
    dist.mkdir()
    (dist / "pkg-0.0.0-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "pkg-0.0.0.tar.gz").write_bytes(b"sdist")

    monkeypatch.setattr(hygiene, "ROOT", fake_root)
    monkeypatch.setattr(
        hygiene,
        "_tracked_paths",
        lambda: {
            "dist/pkg-0.0.0-py3-none-any.whl",
            "dist/pkg-0.0.0.tar.gz",
        },
    )
    report = hygiene.scan()
    types = {f["type"] for f in report["findings"]}
    assert "build_artifact" in types
    assert report["ok"] is False
