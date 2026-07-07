from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


def normalize_text(text: object) -> str:
    """Lowercase and collapse non-word separators for stable matching."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: object) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(str(text or ""))]


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


def bm25_scores(query: str, docs: Iterable[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Small deterministic BM25 implementation for lexical baseline retrieval."""
    doc_tokens = [tokenize(doc) for doc in docs]
    q_tokens = tokenize(query)
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
