from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from external_baselines.common.io import read_jsonl
from external_baselines.common.schema import RetrievedContext
from external_baselines.common.text_utils import bm25_scores, compact_text, normalize_text, tokenize
from external_baselines.vanilla_rag.bilingual_tokenizer import (
    TOKENIZER_VERSION as BILINGUAL_TOKENIZER_VERSION,
)
from external_baselines.vanilla_rag.bilingual_tokenizer import deterministic_bilingual_lexical_tokens

LEGACY_TOKENIZER_VERSION = "latin_word_cjk_unigram_v1"


@dataclass
class LexicalRetriever:
    """True BM25 lexical retriever with chunk normalization and duplicate suppression."""

    documents: list[dict[str, Any]]
    max_chunk_chars: int = 1000
    tokenizer_version: str = LEGACY_TOKENIZER_VERSION
    k1: float = 1.5
    b: float = 0.75
    minimum_score: float = 0.0
    corpus_checksum: str | None = None

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        *,
        max_chunk_chars: int = 1000,
        tokenizer_version: str = LEGACY_TOKENIZER_VERSION,
        k1: float = 1.5,
        b: float = 0.75,
        minimum_score: float = 0.0,
    ) -> "LexicalRetriever":
        if tokenizer_version not in {LEGACY_TOKENIZER_VERSION, BILINGUAL_TOKENIZER_VERSION}:
            raise ValueError(f"Unsupported lexical tokenizer_version: {tokenizer_version}")
        corpus_path = Path(path)
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
        return cls(
            normalized,
            max_chunk_chars=max_chunk_chars,
            tokenizer_version=tokenizer_version,
            k1=float(k1),
            b=float(b),
            minimum_score=float(minimum_score),
            corpus_checksum=sha256(corpus_path.read_bytes()).hexdigest(),
        )

    def _doc_text(self, doc: dict[str, Any]) -> str:
        return str(doc.get("_normalized_text") or doc.get("text") or doc.get("content") or doc.get("chunk") or doc.get("body") or "")

    def _tokenize(self, text: object) -> list[str]:
        if self.tokenizer_version == BILINGUAL_TOKENIZER_VERSION:
            return deterministic_bilingual_lexical_tokens(text)
        return tokenize(text)

    def retrieve_with_trace(self, query: str, *, top_k: int = 5) -> tuple[list[RetrievedContext], dict[str, Any]]:
        query_tokens = self._tokenize(query)
        trace: dict[str, Any] = {
            "query_text": query,
            "query_tokens": query_tokens,
            "tokenizer_version": self.tokenizer_version,
            "bm25_parameters": {"k1": self.k1, "b": self.b},
            "top_k": int(top_k),
            "minimum_score": self.minimum_score,
            "corpus_checksum": self.corpus_checksum,
            "corpus_document_count": len(self.documents),
            "fallback_used": False,
            "ranked_documents": [],
        }
        if not self.documents:
            return [], trace
        texts = [self._doc_text(doc) for doc in self.documents]
        scores = bm25_scores(query, texts, k1=self.k1, b=self.b, tokenizer=self._tokenize)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: (-x[1], str(self.documents[x[0]].get("_chunk_id") or "")),
        )
        contexts: list[RetrievedContext] = []
        query_token_set = set(query_tokens)
        for rank, (idx, score) in enumerate(ranked, start=1):
            if len(contexts) >= top_k:
                break
            doc = self.documents[idx]
            cid = str(doc.get("_chunk_id") or doc.get("chunk_id") or doc.get("id") or f"chunk_{idx}")
            overlap = sorted(query_token_set.intersection(self._tokenize(texts[idx])))
            selected = float(score) > self.minimum_score
            trace["ranked_documents"].append(
                {
                    "document_id": cid,
                    "rank": rank,
                    "score": round(float(score), 6),
                    "lexical_overlap": overlap,
                    "selected": selected,
                }
            )
            if not selected:
                continue
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
        trace["retrieved_document_ids"] = [context.context_id for context in contexts]
        trace["retrieval_coverage"] = bool(contexts)
        return contexts, trace

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedContext]:
        contexts, _ = self.retrieve_with_trace(query, top_k=top_k)
        return contexts
