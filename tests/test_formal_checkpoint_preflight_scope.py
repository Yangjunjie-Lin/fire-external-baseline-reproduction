from __future__ import annotations

from external_baselines.common.decision_suite_preflight import _scoped_runtime_freeze


def test_bm25_checkpoint_does_not_require_unaccessed_embedding_indexes():
    freeze = {
        "indexes": {"dense": {"index_checksum": "dense"}},
        "embedding": {"model_name": "embedding-model"},
    }
    scoped, require_complete = _scoped_runtime_freeze(freeze, ["bm25_rag"])
    assert require_complete is False
    assert scoped["indexes"] == {}
    assert scoped["embedding"] == {}
    assert freeze["indexes"]


def test_embedding_checkpoint_retains_complete_index_gate():
    freeze = {"indexes": {"dense": {"index_checksum": "dense"}}}
    scoped, require_complete = _scoped_runtime_freeze(freeze, ["dense_rag"])
    assert require_complete is True
    assert scoped is freeze
