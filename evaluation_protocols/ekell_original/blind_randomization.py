from __future__ import annotations

"""Blind randomization helper for E-KELL-style expert evaluation.

Does not fabricate scores. Only shuffles method labels into blind codes.
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path


def blind_codes(methods: list[str], *, seed: int = 7) -> dict[str, str]:
    rng = random.Random(seed)
    codes = [f"M{i+1}" for i in range(len(methods))]
    rng.shuffle(codes)
    return dict(zip(methods, codes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="ekell_style_paper_fidelity,chatglm6b,gpt35")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="evaluation_protocols/ekell_original/blind_map.csv")
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    mapping = blind_codes(methods, seed=args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method_id", "blind_code", "seed", "map_sha256"])
        w.writeheader()
        digest = hashlib.sha256(str(sorted(mapping.items())).encode()).hexdigest()
        for mid, code in mapping.items():
            w.writerow({"method_id": mid, "blind_code": code, "seed": args.seed, "map_sha256": digest})
    print(f"Wrote blind map to {path}")


if __name__ == "__main__":
    main()
