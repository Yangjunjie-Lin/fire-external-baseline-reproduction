"""Method runtime lifecycle tests (fake embedding only; no network)."""

from __future__ import annotations

from pathlib import Path

from external_baselines.common.method_runtime import (
    clear_runtime_cache,
    close_method_runtime,
    prepare_dense_runtime,
    prepare_ekell_runtime,
    prepare_hybrid_runtime,
)
from external_baselines.dense_rag.pipeline import build_dense_index, run_scenario as dense_run
from external_baselines.ekell_style.kg_loader import FireKG
from external_baselines.ekell_style.vector_index import VectorIndex
from external_baselines.hybrid_rag.pipeline import run_scenario as hybrid_run
from external_baselines.retrieval.embedding_backends import Text2VecEmbeddingBackend, create_embedding_backend


class CountingFakeModel:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.load_count = 1
        self.encode_call_count = 0

    def encode(self, texts, show_progress_bar=False):  # noqa: ARG002
        self.encode_call_count += 1
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for i, ch in enumerate(str(text).encode("utf-8")):
                vec[i % self.dim] += (ch % 13) / 13.0
            out.append(vec)
        return out


def _evidence(tmp_path: Path) -> Path:
    path = tmp_path / "evidence_chunks.jsonl"
    path.write_text(
        '{"chunk_id":"c1","text":"alpha fire hose","source_id":"s1"}\n'
        '{"chunk_id":"c2","text":"beta smoke alarm","source_id":"s2"}\n',
        encoding="utf-8",
    )
    return path


def test_text2vec_backend_is_truly_lazy() -> None:
    backend = Text2VecEmbeddingBackend(model_name="should-not-load", model_version="v", model=None, dimension=8)
    # Without encode, model stays None (lazy). Injected path sets load_count.
    assert backend.model is None
    assert backend._load_count == 0
    injected = CountingFakeModel(8)
    backend2 = Text2VecEmbeddingBackend(
        model_name="fake", model_version="v", model=injected, dimension=8
    )
    assert backend2._load_count == 1
    backend2.encode(["a"])
    assert injected.encode_call_count == 1


def test_dense_embedding_model_loaded_once_for_multiple_cases(tmp_path: Path) -> None:
    clear_runtime_cache()
    evidence = _evidence(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "evidence_chunks.jsonl").write_text(evidence.read_text(encoding="utf-8"), encoding="utf-8")
    fake = CountingFakeModel(8)
    index_dir = tmp_path / "dense_idx"
    build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=fake,
        reject_smoke=True,
    )
    config = {
        "paper_final": False,
        "paths": {"corpus_dir": str(corpus)},
        "dense_rag": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(index_dir),
            "reject_smoke": True,
            "allow_index_rebuild": False,
            "injected_model": CountingFakeModel(8),
        },
        "llm": {"provider": "heuristic", "model": "h", "model_version": "h"},
    }
    runtime = prepare_dense_runtime(config)
    query_fake = runtime.embedding_backend.model
    assert isinstance(query_fake, CountingFakeModel)
    before = query_fake.encode_call_count
    for i in range(3):
        dense_run({"scenario_id": f"s{i}", "scenario_text": "hose"}, config=config, runtime=runtime)
    assert runtime.audit.index_load_count == 1
    assert query_fake.encode_call_count > before
    # Same backend object reused
    assert runtime.embedding_backend.model is query_fake


def test_dense_index_loaded_once_for_multiple_cases(tmp_path: Path) -> None:
    clear_runtime_cache()
    evidence = _evidence(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "evidence_chunks.jsonl").write_text(evidence.read_text(encoding="utf-8"), encoding="utf-8")
    index_dir = tmp_path / "dense_idx"
    build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=CountingFakeModel(8),
        reject_smoke=True,
    )
    config = {
        "paths": {"corpus_dir": str(corpus)},
        "dense_rag": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(index_dir),
            "reject_smoke": True,
            "injected_model": CountingFakeModel(8),
        },
        "llm": {"provider": "heuristic"},
    }
    runtime = prepare_dense_runtime(config)
    assert runtime.audit.index_load_count == 1
    dense_run({"scenario_id": "s1", "scenario_text": "hose"}, config=config, runtime=runtime)
    dense_run({"scenario_id": "s2", "scenario_text": "alarm"}, config=config, runtime=runtime)
    assert runtime.audit.index_load_count == 1
    assert runtime.audit.case_count == 2


def test_hybrid_reuses_dense_runtime(tmp_path: Path) -> None:
    clear_runtime_cache()
    evidence = _evidence(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "evidence_chunks.jsonl").write_text(evidence.read_text(encoding="utf-8"), encoding="utf-8")
    index_dir = tmp_path / "dense_idx"
    build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=CountingFakeModel(8),
        reject_smoke=True,
    )
    config = {
        "paths": {"corpus_dir": str(corpus)},
        "dense_rag": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(index_dir),
            "reject_smoke": True,
            "injected_model": CountingFakeModel(8),
        },
        "hybrid_rag": {"rrf_k": 60, "top_k": 2, "candidate_pool": 5},
        "llm": {"provider": "heuristic"},
    }
    dense_rt = prepare_dense_runtime(config)
    hybrid_rt = prepare_hybrid_runtime(config)
    assert hybrid_rt.dense_runtime is dense_rt
    hybrid_run({"scenario_id": "s1", "scenario_text": "hose"}, config=config, runtime=hybrid_rt)
    assert hybrid_rt.audit.case_count == 1


def test_ekell_embedding_model_loaded_once_for_multiple_cases(tmp_path: Path) -> None:
    clear_runtime_cache()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "entities.jsonl").write_text('{"entity_id":"e1","name":"hose"}\n', encoding="utf-8")
    (corpus / "relations.jsonl").write_text('{"relation_id":"r1","name":"r"}\n', encoding="utf-8")
    (corpus / "triples.jsonl").write_text(
        '{"head":"hose","relation":"r","tail":"fire","source_id":"t1"}\n', encoding="utf-8"
    )
    (corpus / "evidence_chunks.jsonl").write_text(
        '{"chunk_id":"c1","text":"hose","source_id":"s1"}\n', encoding="utf-8"
    )
    fake = CountingFakeModel(8)
    backend = create_embedding_backend(
        "text2vec", model_name="fake/bge", model_version="v-test", dimension=8, model=fake
    )
    kg = FireKG(
        entities=[{"entity_id": "e1", "name": "hose"}],
        relations=[{"relation_id": "r1", "name": "r"}],
        triples=[{"head": "hose", "relation": "r", "tail": "fire", "source_id": "t1"}],
        evidence_chunks=[{"chunk_id": "c1", "text": "hose", "source_id": "s1"}],
    )
    index_dir = tmp_path / "ekell"
    VectorIndex.from_kg(kg, backend).save_directory(index_dir)
    query_fake = CountingFakeModel(8)
    config = {
        "paths": {"corpus_dir": str(corpus)},
        "ekell_vector": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(index_dir),
            "injected_model": query_fake,
        },
        "llm": {"provider": "heuristic"},
    }
    runtime = prepare_ekell_runtime(config)
    assert runtime.embedding_backend.model is query_fake
    assert runtime.audit.index_load_count == 1


def test_ekell_index_loaded_once_for_multiple_cases(tmp_path: Path) -> None:
    clear_runtime_cache()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, line in (
        ("entities.jsonl", '{"entity_id":"e1","name":"hose"}\n'),
        ("relations.jsonl", '{"relation_id":"r1","name":"r"}\n'),
        ("triples.jsonl", '{"head":"hose","relation":"r","tail":"fire","source_id":"t1"}\n'),
        ("evidence_chunks.jsonl", '{"chunk_id":"c1","text":"hose","source_id":"s1"}\n'),
    ):
        (corpus / name).write_text(line, encoding="utf-8")
    fake = CountingFakeModel(8)
    backend = create_embedding_backend(
        "text2vec", model_name="fake/bge", model_version="v-test", dimension=8, model=fake
    )
    kg = FireKG(
        entities=[{"entity_id": "e1", "name": "hose"}],
        relations=[{"relation_id": "r1", "name": "r"}],
        triples=[{"head": "hose", "relation": "r", "tail": "fire", "source_id": "t1"}],
        evidence_chunks=[{"chunk_id": "c1", "text": "hose", "source_id": "s1"}],
    )
    index_dir = tmp_path / "ekell"
    VectorIndex.from_kg(kg, backend).save_directory(index_dir)
    config = {
        "paths": {"corpus_dir": str(corpus)},
        "ekell_vector": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(index_dir),
            "injected_model": CountingFakeModel(8),
        },
        "llm": {"provider": "heuristic"},
    }
    rt1 = prepare_ekell_runtime(config)
    rt2 = prepare_ekell_runtime(config)
    assert rt1 is rt2
    assert rt1.audit.index_load_count == 1


def test_runtime_closed_after_method_run(tmp_path: Path) -> None:
    clear_runtime_cache()
    evidence = _evidence(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "evidence_chunks.jsonl").write_text(evidence.read_text(encoding="utf-8"), encoding="utf-8")
    index_dir = tmp_path / "dense_idx"
    build_dense_index(
        evidence,
        model_name="fake/bge",
        model_version="v-test",
        backend="text2vec",
        dim=8,
        cache_path=index_dir,
        embedding_model=CountingFakeModel(8),
        reject_smoke=True,
    )
    config = {
        "paths": {"corpus_dir": str(corpus)},
        "dense_rag": {
            "backend": "text2vec",
            "model_name": "fake/bge",
            "model_version": "v-test",
            "dimension": 8,
            "index_path": str(index_dir),
            "injected_model": CountingFakeModel(8),
        },
        "llm": {"provider": "heuristic"},
    }
    runtime = prepare_dense_runtime(config)
    close_method_runtime(runtime)  # should not raise
