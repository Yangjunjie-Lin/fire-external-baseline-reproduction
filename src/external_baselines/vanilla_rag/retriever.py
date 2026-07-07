from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from external_baselines.common.io import read_jsonl
from external_baselines.common.schema import RetrievedContext
from external_baselines.common.text_utils import bm25_scores, compact_text


@dataclass
class LexicalRetriever:
    documents: list[dict[str, Any]]

    @classmethod
    def from_jsonl(cls, path: str) -> "LexicalRetriever":
        return cls(read_jsonl(path))

    def _doc_text(self, doc: dict[str, Any]) -> str:
        return str(doc.get("text") or doc.get("content") or doc.get("chunk") or doc.get("body") or "")

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedContext]:
        texts = [self._doc_text(doc) for doc in self.documents]
        scores = bm25_scores(query, texts)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        contexts: list[RetrievedContext] = []
        for idx, score in ranked[:top_k]:
            if score <= 0 and contexts:
                continue
            doc = self.documents[idx]
            cid = str(doc.get("chunk_id") or doc.get("id") or doc.get("source_id") or f"chunk_{idx}")
            source_id = doc.get("source_id") or doc.get("source") or doc.get("document_id")
            citation = doc.get("citation") or doc.get("url") or source_id or cid
            contexts.append(
                RetrievedContext(
                    context_id=cid,
                    text=compact_text(texts[idx], 1000),
                    source_id=str(source_id) if source_id is not None else None,
                    citation=str(citation) if citation is not None else None,
                    score=round(float(score), 6),
                    metadata={k: v for k, v in doc.items() if k not in {"text", "content", "chunk", "body"}},
                )
            )
        return contexts
