from __future__ import annotations

from typing import Any

from external_baselines.common.schema import RetrievedContext
from external_baselines.common.text_utils import bm25_scores, compact_text, normalize_text, tokenize
from external_baselines.ekell_style.kg_loader import (
    FireKG,
    entity_aliases,
    evidence_chunk_id,
    evidence_citation,
    evidence_source_id,
    evidence_text,
    triple_id,
    triple_parts,
    triple_to_text,
)


def _terms_for_entities(matched_entities: list[dict[str, Any]]) -> set[str]:
    terms: set[str] = set()
    for e in matched_entities:
        for value in [e.get("entity_id"), e.get("name"), e.get("label"), e.get("text"), *entity_aliases(e)]:
            norm = normalize_text(value)
            if norm:
                terms.add(norm)
                terms.update(tokenize(norm))
    return {t for t in terms if t}


def _touches_entity(row: dict[str, Any], entity_terms: set[str]) -> tuple[bool, list[str]]:
    h, r, t = triple_parts(row)
    norm = normalize_text(" ".join([h, r, t, str(row.get("source_id", "")), str(row.get("citation", ""))]))
    hits = sorted(term for term in entity_terms if term and term in norm)
    return bool(hits), hits


def _score_evidence(query: str, entity_terms: set[str], chunk: dict[str, Any], bm25_score: float) -> tuple[float, list[str]]:
    text = " ".join([
        evidence_text(chunk),
        str(chunk.get("source_id", "")),
        str(chunk.get("chunk_id", "")),
        str(chunk.get("title", "")),
    ])
    norm = normalize_text(text)
    hits = sorted(term for term in entity_terms if term and term in norm)
    score = bm25_score + 0.4 * len(hits)
    return score, hits


def retrieve_subgraph(
    scenario_text: str,
    kg: FireKG,
    matched_entities: list[dict[str, Any]],
    *,
    top_k_triples: int = 20,
    top_k_evidence: int = 6,
    top_k_relations: int = 10,
) -> dict[str, Any]:
    entity_terms = _terms_for_entities(matched_entities)
    query_with_entities = scenario_text + " " + " ".join(sorted(entity_terms))

    triple_rows = list(kg.triples or [])
    relation_rows = list(kg.relations or [])
    scored_triples: list[tuple[float, dict[str, Any], str]] = []
    trace_triples: list[str] = []

    for idx, row in enumerate(triple_rows):
        text = triple_to_text(row)
        touches, hits = _touches_entity(row, entity_terms)
        bm25 = bm25_scores(query_with_entities, [text])[0] if text else 0.0
        score = bm25 + (1.0 if touches else 0.0) + 0.1 * len(hits)
        if score > 0:
            enriched = dict(row)
            enriched["triple_id"] = triple_id(row, idx)
            enriched["triple_text"] = text
            enriched["score"] = round(float(score), 6)
            enriched["selection_reason"] = "touching_matched_entity" if touches else "lexical_relevance"
            enriched["matched_terms"] = hits
            scored_triples.append((score, enriched, enriched["selection_reason"]))
            trace_triples.append(f"{enriched['triple_id']}: {enriched['selection_reason']} terms={hits[:8]}")

    scored_triples.sort(key=lambda x: x[0], reverse=True)
    selected_triples = [row for _, row, _ in scored_triples[:top_k_triples]]

    scored_relations: list[tuple[float, dict[str, Any]]] = []
    for idx, row in enumerate(relation_rows):
        text = triple_to_text(row)
        touches, hits = _touches_entity(row, entity_terms)
        bm25 = bm25_scores(query_with_entities, [text])[0] if text else 0.0
        score = bm25 + (0.8 if touches else 0.0) + 0.08 * len(hits)
        if score > 0:
            enriched = dict(row)
            enriched["triple_id"] = str(row.get("relation_id") or row.get("id") or f"relation_{idx}")
            enriched["triple_text"] = text
            enriched["score"] = round(float(score), 6)
            enriched["selection_reason"] = "relation_like_row_touching_entity" if touches else "relation_like_lexical_relevance"
            enriched["matched_terms"] = hits
            scored_relations.append((score, enriched))
    scored_relations.sort(key=lambda x: x[0], reverse=True)
    selected_relations = [row for _, row in scored_relations[:top_k_relations]]

    evidence_texts = [evidence_text(c) for c in kg.evidence_chunks]
    lexical_scores = bm25_scores(query_with_entities, evidence_texts)
    scored_chunks: list[tuple[float, dict[str, Any], list[str]]] = []
    trace_evidence: list[str] = []
    for idx, chunk in enumerate(kg.evidence_chunks):
        score, hits = _score_evidence(query_with_entities, entity_terms, chunk, lexical_scores[idx] if idx < len(lexical_scores) else 0.0)
        if score > 0:
            c = dict(chunk)
            c["chunk_id"] = evidence_chunk_id(chunk, idx)
            c["score"] = round(float(score), 6)
            c["selection_reason"] = "lexical_plus_entity_term_relevance" if hits else "lexical_relevance"
            c["matched_terms"] = hits
            scored_chunks.append((score, c, hits))
            trace_evidence.append(f"{c['chunk_id']}: {c['selection_reason']} terms={hits[:8]}")
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    selected_chunks = [row for _, row, _ in scored_chunks[:top_k_evidence]]

    contexts: list[RetrievedContext] = []
    for row in selected_triples:
        contexts.append(RetrievedContext(
            context_id=str(row.get("triple_id")),
            text=row.get("triple_text") or triple_to_text(row),
            source_id=str(row.get("source_id") or row.get("citation") or "kg_triple"),
            citation=str(row.get("citation") or row.get("source_id") or row.get("triple_id") or "kg_triple"),
            score=float(row.get("score", 0.0)),
            metadata={"kind": "kg_triple", **{k: v for k, v in row.items() if k != "triple_text"}},
        ))
    for row in selected_relations:
        contexts.append(RetrievedContext(
            context_id=str(row.get("triple_id")),
            text=row.get("triple_text") or triple_to_text(row),
            source_id=str(row.get("source_id") or "relation_like_row"),
            citation=str(row.get("citation") or row.get("source_id") or row.get("triple_id") or "relation_like_row"),
            score=float(row.get("score", 0.0)),
            metadata={"kind": "relation_like_row", **{k: v for k, v in row.items() if k != "triple_text"}},
        ))
    for row in selected_chunks:
        contexts.append(RetrievedContext(
            context_id=str(row.get("chunk_id")),
            text=compact_text(evidence_text(row), 1200),
            source_id=evidence_source_id(row),
            citation=evidence_citation(row),
            score=float(row.get("score", 0.0)),
            metadata={"kind": "evidence_chunk", **{k: v for k, v in row.items() if k not in {"text", "content", "chunk", "body"}}},
        ))

    return {
        "matched_entities": matched_entities,
        "triples": selected_triples + selected_relations,
        "evidence_chunks": selected_chunks,
        "contexts": contexts,
        "retrieval_trace": {
            "entity_terms": sorted(entity_terms),
            "triple_selection_reason": trace_triples[:top_k_triples] + [f"relation: {r.get('selection_reason')}" for r in selected_relations],
            "evidence_selection_reason": trace_evidence[:top_k_evidence],
        },
    }
