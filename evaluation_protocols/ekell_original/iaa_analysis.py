from __future__ import annotations

"""IAA analysis scaffold for expert scores. Does not invent ratings."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pairwise_agreement(rows: list[dict[str, str]], dim: str) -> dict[str, float | int | None]:
    # Placeholder: compute simple exact-match rate when >=2 annotators share query/method.
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        val = (row.get(dim) or "").strip()
        if not val:
            continue
        key = (row.get("query_id") or "", row.get("blind_code") or row.get("method_id") or "")
        by_key[key].append(val)
    pairs = 0
    agree = 0
    for vals in by_key.values():
        if len(vals) < 2:
            continue
        pairs += 1
        agree += int(len(set(vals)) == 1)
    return {
        "dimension": dim,
        "comparable_items": pairs,
        "exact_match_rate": (agree / pairs) if pairs else None,
        "note": "Empty until expert scores exist; do not report fabricated IAA.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="evaluation_protocols/ekell_original/expert_evaluation_form.csv")
    args = parser.parse_args()
    rows = load_scores(Path(args.scores))
    for dim in ("comprehensibility", "accuracy", "conciseness", "instructiveness"):
        print(pairwise_agreement(rows, dim))


if __name__ == "__main__":
    main()
