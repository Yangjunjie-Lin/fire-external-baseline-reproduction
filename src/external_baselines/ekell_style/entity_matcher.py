from __future__ import annotations

from typing import Any

from external_baselines.common.text_utils import keyword_overlap, normalize_text, tokenize
from external_baselines.ekell_style.kg_loader import entity_aliases, entity_id, entity_name


def match_entities(scenario_text: str, parsed: dict[str, Any], entities: list[dict[str, Any]], *, top_k: int = 8) -> list[dict[str, Any]]:
    query_terms = [scenario_text]
    for key in ["incident_type", "location", "hazards", "affected_people", "information_gaps"]:
        value = parsed.get(key)
        if isinstance(value, list):
            query_terms.extend(str(v) for v in value)
        elif value:
            query_terms.append(str(value))
    query_text = " ".join(query_terms)
    q_norm = normalize_text(query_text)
    q_tokens = set(tokenize(query_text))

    scored: list[tuple[float, dict[str, Any], str]] = []
    for entity in entities:
        aliases = entity_aliases(entity)
        best_score = 0.0
        best_reason = ""
        for alias in aliases:
            a_norm = normalize_text(alias)
            if not a_norm:
                continue
            if a_norm and a_norm in q_norm:
                score = 1.0 + min(0.5, len(a_norm) / 80)
                reason = "exact_or_substring_match"
            else:
                overlap = keyword_overlap(q_norm, a_norm)
                a_tokens = set(tokenize(a_norm))
                token_hits = len(q_tokens & a_tokens)
                score = overlap + 0.05 * token_hits
                reason = "keyword_overlap"
            if score > best_score:
                best_score, best_reason = score, reason
        if best_score > 0:
            enriched = dict(entity)
            enriched["match_score"] = round(best_score, 6)
            enriched["match_reason"] = best_reason
            enriched["entity_id"] = entity_id(entity)
            enriched["name"] = entity_name(entity)
            scored.append((best_score, enriched, best_reason))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [e for _, e, _ in scored[:top_k]]
