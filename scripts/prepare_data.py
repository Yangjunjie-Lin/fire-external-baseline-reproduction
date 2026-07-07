#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

CORPUS_FILES = ["evidence_chunks.jsonl", "entities.jsonl", "relations.jsonl", "triples.jsonl"]
SCENARIO_FILE = "scenario_matrix_v2.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_found(filename: str, candidates: list[Path], target: Path) -> dict:
    record = {"filename": filename, "target_path": str(target), "copied": False, "searched": [str(c) for c in candidates]}
    for base in candidates:
        src = base / filename
        if src.exists() and src.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            record.update({
                "copied": True,
                "source_path": str(src),
                "file_size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            })
            print(f"copied {src} -> {target}")
            return record
    print(f"WARNING: missing {filename}; searched: {', '.join(str(c) for c in candidates)}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy data files from fire-agent-demo into this independent external baseline repo.")
    parser.add_argument("--source", required=True, help="Path to fire-agent-demo root")
    parser.add_argument("--target", default="data/", help="Target data directory in this repo")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    corpus_target = target / "corpus"
    scenario_target = target / "scenarios"

    corpus_candidates = [
        source / "data" / "processed" / "4B_fire_kg_graphrag",
        source / "data" / "processed" / "kg_graphrag",
        source / "data" / "corpus",
        source / "data" / "processed",
    ]
    scenario_candidates = [
        source / "data" / "examples",
        source / "data" / "scenarios",
        source / "data",
    ]

    files = []
    for filename in CORPUS_FILES:
        files.append(copy_if_found(filename, corpus_candidates, corpus_target / filename))
    files.append(copy_if_found(SCENARIO_FILE, scenario_candidates, scenario_target / SCENARIO_FILE))

    missing = [f["filename"] for f in files if not f.get("copied")]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "target_data_dir": str(target),
        "copy_only_no_code_import": True,
        "files": files,
        "missing_files": missing,
        "warnings": [
            "Copied data are input artifacts only; this repository must not import fire-agent-demo code.",
            "Do not commit copied private or large data by default. .gitignore excludes copied data unless explicitly curated.",
        ],
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote data manifest to {target / 'data_manifest.json'}")
    if SCENARIO_FILE in missing:
        print("WARNING: scenario_matrix_v2.json is missing; baseline runs need --dataset pointing to a valid scenario file.")
    for filename in CORPUS_FILES:
        if filename in missing:
            print(f"WARNING: corpus file missing: {filename}")


if __name__ == "__main__":
    main()
