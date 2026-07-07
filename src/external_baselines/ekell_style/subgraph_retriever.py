from __future__ import annotations

from typing import Any

from external_baselines.common.schema import RetrievedContext
from external_baselines.common.text_utils import bm25_scores, compact_text, normalize_text
from external_baselines.ekell_style.kg_loader import FireKG, triple_parts, triple_to_text


def _chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text") or chunk.get("content") or chunk.get("chunk") or chunk.get("body") or "")


def retrieve_subgraph(
    scenario_text: str,
    kg: FireKG,
    matched_entities: list[dict[str, Any]],
    *,
    top_k_triples: int = 20,
    top_k_evidence: int = 6,
) -> dict[str, Any]:
    entity_terms: set[str] = set()
    for e in matched_entities:
        for value in [e.get("entity_id"), e.get("name"), e.get("label"), e.get("text")]:
            norm = normalize_text(value)
            if norm:
                entity_terms.add(norm)

    triples = list(kg.triples or []) + list(kg.relations or [])
    scored_triples: list[tuple[float, dict[str, Any]]] = []
    for row in triples:
        h, r, t = triple_parts(row)
        triple_text = triple_to_text(row)
        norm = normalize_text(" ".join([h, r, t, row.get("source_id", "")]))
        score = 0.0
        for term in entity_terms:
            if term and term in norm:
                score += 1.0
        # query lexical tie-breaker
        score += bm25_scores(scenario_text, [triple_text])[0] if triple_text else 0.0
        if score > 0:
            enriched = dict(row)
            enriched["triple_text"] = triple_text
            enriched["score"] = round(float(score), 6)
            scored_triples.append((score, enriched))
    scored_triples.sort(key=lambda x: x[0], reverse=True)
    selected_triples = [row for _, row in scored_triples[:top_k_triples]]

    evidence_texts = [_chunk_text(c) for c in kg.evidence_chunks]
    lexical_scores = bm25_scores(scenario_text + " " + " ".join(entity_terms), evidence_texts)
    scored_chunks: list[tuple[float, dict[str, Any]]] = []
    for idx, chunk in enumerate(kg.evidence_chunks):
        text = evidence_texts[idx]
        norm = normalize_text(text + " " + str(chunk.get("source_id", "")) + " " + str(chunk.get("chunk_id", "")))
        score = lexical_scores[idx] if idx < len(lexical_scores) else 0.0
        for term in entity_terms:
            if term and term in norm:
                score += 1.0
        if score > 0:
            c = dict(chunk)
            c["score"] = round(float(score), 6)
            scored_chunks.append((score, c))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    selected_chunks = [row for _, row in scored_chunks[:top_k_evidence]]

    contexts: list[RetrievedContext] = []
    for idx, row in enumerate(selected_triples):
        contexts.append(
            RetrievedContext(
                context_id=str(row.get("triple_id") or row.get("id") or f"triple_{idx}"),
                text=row.get("triple_text") or triple_to_text(row),
                source_id=str(row.get("source_id") or "kg_triple"),
                citation=str(row.get("citation") or row.get("source_id") or "kg_triple"),
                score=float(row.get("score", 0.0)),
                metadata={"kind": "kg_triple", **{k: v for k, v in row.items() if k != "triple_text"}},
            )
        )
    for idx, row in enumerate(selected_chunks):
        contexts.append(
            RetrievedContext(
                context_id=str(row.get("chunk_id") or row.get("id") or f"chunk_{idx}"),
                text=compact_text(_chunk_text(row), 1200),
                source_id=str(row.get("source_id") or row.get("source") or "evidence_chunk"),
                citation=str(row.get("citation") or row.get("url") or row.get("source_id") or row.get("chunk_id") or "evidence_chunk"),
                score=float(row.get("score", 0.0)),
                metadata={"kind": "evidence_chunk", **{k: v for k, v in row.items() if k not in {"text", "content", "chunk", "body"}}},
            )
        )

    return {"triples": selected_triples, "evidence_chunks": selected_chunks, "contexts": contexts}
