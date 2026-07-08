from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any


def mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def standard_deviation(values: Sequence[float], *, sample: bool = True) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    denom = len(vals) - 1 if sample else len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / denom)


def bootstrap_confidence_interval(values: Sequence[float], *, n_boot: int = 2000, confidence: float = 0.95, seed: int = 7) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(max(1, n_boot)):
        sample = [rng.choice(vals) for _ in vals]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, min(len(boot_means) - 1, int((alpha / 2) * len(boot_means))))
    hi_idx = max(0, min(len(boot_means) - 1, int((1 - alpha / 2) * len(boot_means)) - 1))
    return boot_means[lo_idx], boot_means[hi_idx]


def paired_difference(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b) if x is not None and y is not None]


def win_rate(a: Sequence[float], b: Sequence[float]) -> dict[str, float | int]:
    diffs = paired_difference(a, b)
    if not diffs:
        return {"n": 0, "wins": 0, "ties": 0, "losses": 0, "win_rate": 0.0, "tie_rate": 0.0, "loss_rate": 0.0}
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    losses = sum(1 for d in diffs if d < 0)
    n = len(diffs)
    return {"n": n, "wins": wins, "ties": ties, "losses": losses, "win_rate": wins / n, "tie_rate": ties / n, "loss_rate": losses / n}


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float | None:
    diffs = paired_difference(a, b)
    if not diffs:
        return None
    sd = standard_deviation(diffs, sample=True)
    if not sd:
        return 0.0
    m = mean(diffs)
    return None if m is None else m / sd


def summarize(values: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    ci_lo, ci_hi = bootstrap_confidence_interval(vals) if vals else (None, None)
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": standard_deviation(vals),
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
    }
