from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def normalize_text(text: object) -> str:
    """Lowercase and collapse non-word separators for stable matching.

    Preserves CJK characters so multilingual lexical matching remains useful.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: object) -> list[str]:
    """Multilingual tokenizer: Latin tokens + individual CJK characters."""
    raw = str(text or "")
    tokens: list[str] = [m.group(0).lower() for m in _WORD_RE.finditer(raw)]
    tokens.extend(ch for ch in raw if _CJK_RE.match(ch))
    return tokens


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Reciprocal Rank Fusion over (doc_id, score) ranked lists.

    Returns (doc_id, fused_score, component_scores).
    """
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for list_idx, ranked in enumerate(ranked_lists):
        w = float(weights[list_idx]) if list_idx < len(weights) else 1.0
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            contrib = w * (1.0 / (k + rank))
            fused[doc_id] = fused.get(doc_id, 0.0) + contrib
            components.setdefault(doc_id, {})[f"list_{list_idx}"] = float(score)
            components[doc_id][f"rrf_list_{list_idx}"] = contrib
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score, components.get(doc_id, {})) for doc_id, score in ordered]


def keyword_overlap(a: object, b: object) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def compact_text(text: object, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def bm25_scores(
    query: str,
    docs: Iterable[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    tokenizer: Callable[[object], list[str]] = tokenize,
) -> list[float]:
    """Small deterministic BM25 implementation for lexical baseline retrieval."""
    doc_tokens = [tokenizer(doc) for doc in docs]
    q_tokens = tokenizer(query)
    if not doc_tokens or not q_tokens:
        return [0.0 for _ in doc_tokens]

    n_docs = len(doc_tokens)
    avgdl = sum(len(d) for d in doc_tokens) / max(1, n_docs)
    df: Counter[str] = Counter()
    for toks in doc_tokens:
        for term in set(toks):
            df[term] += 1

    scores: list[float] = []
    for toks in doc_tokens:
        freqs = Counter(toks)
        dl = len(toks) or 1
        score = 0.0
        for term in q_tokens:
            if term not in freqs:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            tf = freqs[term]
            denom = tf + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            score += idf * (tf * (k1 + 1)) / max(denom, 1e-9)
        scores.append(float(score))
    return scores


def extract_json_object(text: str) -> dict:
    """Extract first JSON object from model text; return empty dict on failure."""
    import json

    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Split only obvious semicolon/newline lists; otherwise keep the sentence.
        if "\n" in stripped or ";" in stripped:
            parts = re.split(r"[;\n]+", stripped)
            return [p.strip(" -*\t") for p in parts if p.strip(" -*\t")]
        return [stripped]
    return [value]
