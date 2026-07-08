from __future__ import annotations

from typing import Any, Callable

from external_baselines.common.text_utils import keyword_overlap, normalize_text, tokenize
from external_baselines.ekell_style.kg_loader import entity_aliases, entity_id, entity_name

EmbeddingScorer = Callable[[str, str], float]

BILINGUAL_TERMS: dict[str, list[str]] = {
    "fire": ["火", "火灾", "着火", "燃烧"],
    "smoke": ["烟", "烟雾", "浓烟"],
    "electrical": ["电", "电气", "电力", "配电", "电器"],
    "power": ["电源", "断电", "切电", "电力"],
    "evacuation": ["疏散", "撤离", "逃生"],
    "mall": ["商场", "购物中心", "商业综合体"],
    "warehouse": ["仓库", "库房"],
    "respiratory protection": ["呼吸防护", "空气呼吸器", "防毒面具"],
    "hazmat": ["危化", "危险化学品", "化学品"],
    "gas": ["燃气", "气体", "煤气"],
}


def _expanded_terms(text: str) -> set[str]:
    norm = normalize_text(text)
    terms = set(tokenize(norm))
    if norm:
        terms.add(norm)
    for eng, zh_terms in BILINGUAL_TERMS.items():
        eng_norm = normalize_text(eng)
        if eng_norm in norm or any(zh in text for zh in zh_terms):
            terms.add(eng_norm)
            terms.update(normalize_text(x) for x in zh_terms)
            terms.update(tokenize(eng_norm))
    # Add individual CJK characters only as weak signals for mixed-language matching.
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            terms.add(ch)
    return {t for t in terms if t}


def _query_text(scenario_text: str, parsed: dict[str, Any]) -> str:
    parts = [scenario_text]
    for key in [
        "incident_type",
        "location",
        "building_type",
        "hazards",
        "affected_people",
        "resources_or_equipment",
        "information_gaps",
    ]:
        value = parsed.get(key)
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def _score_alias(query_norm: str, query_terms: set[str], alias: str, embedding_scorer: EmbeddingScorer | None = None) -> tuple[float, str, list[str]]:
    alias_norm = normalize_text(alias)
    if not alias_norm:
        return 0.0, "empty_alias", []
    alias_terms = _expanded_terms(alias)
    matched_terms = sorted((query_terms & alias_terms) | ({alias_norm} if alias_norm and alias_norm in query_norm else set()))

    if alias_norm == query_norm:
        return 1.0, "exact_full_text_match", matched_terms or [alias]
    if alias_norm in query_norm:
        return 0.92 + min(0.05, len(alias_norm) / 300), "exact_or_substring_match", matched_terms or [alias]
    if normalize_text(alias) in query_terms:
        return 0.86, "normalized_term_match", matched_terms or [alias]

    token_score = 0.0
    if alias_terms:
        token_score = len(query_terms & alias_terms) / max(len(alias_terms), 1)
    overlap = keyword_overlap(query_norm, alias_norm)
    score = max(overlap, token_score * 0.78)
    reason = "token_overlap" if token_score >= overlap else "keyword_overlap"

    if embedding_scorer is not None:
        try:
            emb_score = float(embedding_scorer(query_norm, alias_norm))
            if emb_score > score:
                score = emb_score
                reason = "optional_embedding_similarity"
        except Exception:
            pass
    return score, reason, matched_terms


def match_entities(
    scenario_text: str,
    parsed: dict[str, Any],
    entities: list[dict[str, Any]],
    *,
    top_k: int = 8,
    min_score: float = 0.08,
    embedding_scorer: EmbeddingScorer | None = None,
) -> list[dict[str, Any]]:
    """Match scenario terms to KG entities using transparent non-target-project heuristics."""
    query = _query_text(scenario_text, parsed)
    q_norm = normalize_text(query)
    q_terms = _expanded_terms(query)

    scored: list[tuple[float, dict[str, Any]]] = []
    for entity in entities:
        best_score = 0.0
        best_reason = "no_match"
        all_terms: set[str] = set()
        for alias in entity_aliases(entity):
            score, reason, terms = _score_alias(q_norm, q_terms, alias, embedding_scorer)
            all_terms.update(terms)
            if score > best_score:
                best_score = score
                best_reason = reason
        if best_score >= min_score:
            enriched = dict(entity)
            enriched["entity_id"] = entity_id(entity)
            enriched["name"] = entity_name(entity)
            enriched["match_score"] = round(float(best_score), 6)
            enriched["match_reason"] = best_reason
            enriched["matched_terms"] = sorted(all_terms)[:20]
            scored.append((best_score, enriched))

    scored.sort(key=lambda item: (item[0], str(item[1].get("name"))), reverse=True)
    return [entity for _, entity in scored[:top_k]]
