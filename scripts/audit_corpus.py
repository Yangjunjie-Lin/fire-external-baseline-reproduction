#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.ekell_style.kg_loader import audit_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit copied corpus assets for external baselines.")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = audit_corpus(args.corpus)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
