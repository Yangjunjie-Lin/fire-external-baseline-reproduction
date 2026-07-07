#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CORPUS_FILES = ["evidence_chunks.jsonl", "entities.jsonl", "relations.jsonl", "triples.jsonl"]
SCENARIO_FILE = "scenario_matrix_v2.json"


def copy_if_found(filename: str, candidates: list[Path], target: Path) -> bool:
    for base in candidates:
        src = base / filename
        if src.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            print(f"copied {src} -> {target}")
            return True
    print(f"missing {filename}; searched: {', '.join(str(c) for c in candidates)}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy data files from fire-agent-demo into this external baseline repo.")
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

    copied = 0
    for filename in CORPUS_FILES:
        copied += int(copy_if_found(filename, corpus_candidates, corpus_target / filename))
    copied += int(copy_if_found(SCENARIO_FILE, scenario_candidates, scenario_target / SCENARIO_FILE))

    print(f"done; copied {copied}/{len(CORPUS_FILES) + 1} expected files")
    print("Reminder: data files are copied input only. Do not import target-project code.")


if __name__ == "__main__":
    main()
