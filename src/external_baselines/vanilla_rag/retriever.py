from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from external_baselines.common.io import read_jsonl
from external_baselines.common.schema import RetrievedContext
from external_baselines.common.text_utils import bm25_scores, compact_text, normalize_text


@dataclass
class LexicalRetriever:
    """True BM25 lexical retriever with chunk normalization and duplicate suppression."""

    documents: list[dict[str, Any]]
    max_chunk_chars: int = 1000

    @classmethod
    def from_jsonl(cls, path: str, *, max_chunk_chars: int = 1000) -> "LexicalRetriever":
        docs = read_jsonl(path)
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        for i, doc in enumerate(docs):
            text = str(doc.get("text") or doc.get("content") or doc.get("chunk") or doc.get("body") or "").strip()
            if not text:
                continue
            cid = str(doc.get("chunk_id") or doc.get("id") or doc.get("source_id") or f"chunk_{i}")
            norm = normalize_text(text)
            # Duplicate suppression by id and near-identical normalized text.
            if cid in seen_ids or norm in seen_text:
                continue
            seen_ids.add(cid)
            seen_text.add(norm)
            row = dict(doc)
            row["_normalized_text"] = text
            row["_chunk_id"] = cid
            normalized.append(row)
        return cls(normalized, max_chunk_chars=max_chunk_chars)

    def _doc_text(self, doc: dict[str, Any]) -> str:
        return str(doc.get("_normalized_text") or doc.get("text") or doc.get("content") or doc.get("chunk") or doc.get("body") or "")

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedContext]:
        if not self.documents:
            return []
        texts = [self._doc_text(doc) for doc in self.documents]
        scores = bm25_scores(query, texts)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: (-x[1], str(self.documents[x[0]].get("_chunk_id") or "")),
        )
        contexts: list[RetrievedContext] = []
        for idx, score in ranked:
            if len(contexts) >= top_k:
                break
            if score <= 0 and contexts:
                continue
            if score <= 0 and not contexts:
                # No-result handling: return empty rather than arbitrary zero-score docs.
                return []
            doc = self.documents[idx]
            cid = str(doc.get("_chunk_id") or doc.get("chunk_id") or doc.get("id") or f"chunk_{idx}")
            source_id = doc.get("source_id") or doc.get("source") or doc.get("document_id")
            citation = doc.get("citation") or doc.get("url") or source_id or cid
            meta = {k: v for k, v in doc.items() if not str(k).startswith("_") and k not in {"text", "content", "chunk", "body"}}
            meta["retrieval_backend"] = "deterministic_lexical_bm25"
            meta["bm25_score"] = round(float(score), 6)
            contexts.append(
                RetrievedContext(
                    context_id=cid,
                    text=compact_text(texts[idx], self.max_chunk_chars),
                    source_id=str(source_id) if source_id is not None else None,
                    citation=str(citation) if citation is not None else None,
                    score=round(float(score), 6),
                    metadata=meta,
                )
            )
        return contexts
