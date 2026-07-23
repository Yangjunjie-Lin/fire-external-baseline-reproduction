from __future__ import annotations

import json

from external_baselines.common.text_utils import bm25_scores
from external_baselines.vanilla_rag.bilingual_tokenizer import (
    TOKENIZER_VERSION,
    deterministic_bilingual_lexical_tokens,
)
from external_baselines.vanilla_rag.retriever import LexicalRetriever


def _corpus(tmp_path):
    path = tmp_path / "evidence_chunks.jsonl"
    rows = [
        {
            "chunk_id": "electrical",
            "text": "Electrical fire response requires confirmation of power isolation.",
        },
        {
            "chunk_id": "smoke",
            "text": "High smoke conditions require respiratory protection and controlled entry.",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_bilingual_tokenizer_is_stable_and_preserves_identifiers():
    text = "A-12 配电箱冒烟，确认断电"
    first = deterministic_bilingual_lexical_tokens(text)
    assert first == deterministic_bilingual_lexical_tokens(text)
    assert {"a-12", "electrical", "smoke", "confirmation", "power", "isolation"} <= set(first)
    assert "配电" in first


def test_bilingual_bm25_retrieves_by_lexical_alias_without_fallback(tmp_path):
    retriever = LexicalRetriever.from_jsonl(
        str(_corpus(tmp_path)), tokenizer_version=TOKENIZER_VERSION, minimum_score=0.0
    )
    contexts, trace = retriever.retrieve_with_trace("电气间起火，现场需要确认断电", top_k=2)
    assert contexts
    assert contexts[0].context_id == "electrical"
    assert trace["fallback_used"] is False
    assert trace["ranked_documents"][0]["lexical_overlap"]
    assert trace["corpus_checksum"]


def test_bilingual_bm25_is_deterministic_and_scores_are_recomputable(tmp_path):
    retriever = LexicalRetriever.from_jsonl(str(_corpus(tmp_path)), tokenizer_version=TOKENIZER_VERSION)
    query = "地下通道有浓烟，需要呼吸防护后进入"
    first, first_trace = retriever.retrieve_with_trace(query, top_k=2)
    second, second_trace = retriever.retrieve_with_trace(query, top_k=2)
    assert [item.context_id for item in first] == [item.context_id for item in second]
    assert first_trace == second_trace
    documents = [retriever._doc_text(doc) for doc in retriever.documents]
    recomputed = bm25_scores(query, documents, tokenizer=retriever._tokenize)
    trace_by_id = {row["document_id"]: row["score"] for row in first_trace["ranked_documents"]}
    for doc, score in zip(retriever.documents, recomputed, strict=True):
        assert trace_by_id[doc["_chunk_id"]] == round(score, 6)


def test_unrelated_query_remains_empty_instead_of_returning_zero_score_docs(tmp_path):
    retriever = LexicalRetriever.from_jsonl(str(_corpus(tmp_path)), tokenizer_version=TOKENIZER_VERSION)
    contexts, trace = retriever.retrieve_with_trace("天气晴朗", top_k=2)
    assert contexts == []
    assert trace["fallback_used"] is False
    assert all(row["score"] == 0.0 and not row["selected"] for row in trace["ranked_documents"])
