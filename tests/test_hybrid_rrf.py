"""Hybrid RRF and dense-reuse tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from external_baselines.dense_rag.pipeline import DenseRetriever, build_dense_index
from external_baselines.hybrid_rag.pipeline import hybrid_retrieve, rrf_fuse_deterministic
from external_baselines.retrieval.dense_index import DenseIndexError
from external_baselines.retrieval.embedding_backends import create_embedding_backend
from external_baselines.vanilla_rag.retriever import LexicalRetriever
from tests.test_dense_real_index import FakeEmbeddingModel, _evidence


def test_hybrid_rrf_formula() -> None:
    fused = rrf_fuse_deterministic(
        [("a", 1.0), ("b", 0.5)],
        [("b", 0.9), ("c", 0.1)],
        rrf_k=60,
        lexical_weight=1.0,
        dense_weight=1.0,
    )
    # lexical ranks: a=1, b=2; dense ranks: b=1, c=2
    scores = {doc: score for doc, score, _ in fused}
    assert scores["b"] == pytest.approx(1.0 / 62 + 1.0 / 61)
    assert scores["a"] == pytest.approx(1.0 / 61)
    assert scores["c"] == pytest.approx(1.0 / 62)


def test_hybrid_tie_breaking_is_deterministic() -> None:
    fused1 = rrf_fuse_deterministic([("a", 1.0), ("b", 1.0)], [("a", 1.0), ("b", 1.0)])
    fused2 = rrf_fuse_deterministic([("a", 1.0), ("b", 1.0)], [("a", 1.0), ("b", 1.0)])
    assert [d for d, _, _ in fused1] == [d for d, _, _ in fused2]


def test_hybrid_combines_bm25_and_dense(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "idx"
    fake = FakeEmbeddingModel(8)
    index = build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=fake,
        reject_smoke=True,
    )
    emb = create_embedding_backend(
        "text2vec", model_name="fake/bge", model_version="v-test", dimension=8, reject_smoke=True, model=fake
    )
    lexical = LexicalRetriever.from_jsonl(str(evidence))
    dense = DenseRetriever(index, embedding_backend=emb)
    hits = hybrid_retrieve(
        "alpha fire",
        lexical=lexical,
        dense=dense,
        top_k=3,
        dense_index_checksum=index.checksum,
    )
    assert hits
    assert hits[0].metadata.get("retrieval_backend") == "hybrid_rrf"
    assert "rrf_score" in hits[0].metadata


def test_hybrid_deduplicates_chunks(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "idx"
    fake = FakeEmbeddingModel(8)
    index = build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=fake,
        reject_smoke=True,
    )
    emb = create_embedding_backend(
        "text2vec", model_name="fake/bge", model_version="v-test", dimension=8, reject_smoke=True, model=fake
    )
    hits = hybrid_retrieve(
        "alpha",
        lexical=LexicalRetriever.from_jsonl(str(evidence)),
        dense=DenseRetriever(index, embedding_backend=emb),
        top_k=10,
    )
    ids = [h.context_id for h in hits]
    assert len(ids) == len(set(ids))


def test_hybrid_reuses_dense_index(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    index_dir = tmp_path / "idx"
    fake = FakeEmbeddingModel(8)
    index = build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=fake,
        reject_smoke=True,
    )
    assert index.checksum
    assert index.index_dir == index_dir


def test_hybrid_rejects_missing_dense_index(tmp_path: Path) -> None:
    from external_baselines.hybrid_rag.pipeline import run_scenario

    with pytest.raises(DenseIndexError):
        run_scenario(
            {"scenario_id": "s1", "scenario_text": "fire"},
            config={
                "paper_final": True,
                "paths": {"corpus_dir": str(tmp_path)},
                "hybrid_rag": {"reject_smoke": True},
                "dense_rag": {
                    "backend": "text2vec",
                    "model_name": "fake",
                    "model_version": "v1",
                    "dimension": 8,
                    "reject_smoke": True,
                    "index_path": str(tmp_path / "missing_idx"),
                },
                "llm": {"provider": "heuristic", "model": "h", "model_version": "h"},
            },
        )


def test_hybrid_rejects_dense_checksum_mismatch() -> None:
    from external_baselines.common.fairness import CrossMethodFairnessError, validate_cross_method_fairness

    with pytest.raises(CrossMethodFairnessError):
        validate_cross_method_fairness(
            {
                "dense_rag": {
                    "paper_final": True,
                    "llm": {"provider": "siliconflow", "model": "m", "model_version": "v"},
                    "dense_rag": {"backend": "text2vec", "model_name": "bge", "model_version": "v1", "dimension": 8},
                    "dense_index_checksum": "aaa",
                },
                "hybrid_rag": {
                    "paper_final": True,
                    "llm": {"provider": "siliconflow", "model": "m", "model_version": "v"},
                    "dense_rag": {"backend": "text2vec", "model_name": "bge", "model_version": "v1", "dimension": 8},
                    "dense_index_checksum": "bbb",
                },
            }
        )


def test_hybrid_does_not_silently_fallback_to_bm25(tmp_path: Path) -> None:
    from external_baselines.hybrid_rag.pipeline import run_scenario

    (tmp_path / "evidence_chunks.jsonl").write_text(
        '{"chunk_id":"c1","text":"hose"}\n', encoding="utf-8"
    )
    with pytest.raises(DenseIndexError):
        run_scenario(
            {"scenario_id": "s1", "scenario_text": "hose"},
            config={
                "paper_final": True,
                "paths": {"corpus_dir": str(tmp_path)},
                "hybrid_rag": {"reject_smoke": True, "rrf_k": 60},
                "dense_rag": {
                    "backend": "text2vec",
                    "model_name": "x",
                    "model_version": "y",
                    "dimension": 8,
                    "reject_smoke": True,
                    "index_path": str(tmp_path / "nope"),
                },
                "llm": {"provider": "heuristic"},
            },
        )
