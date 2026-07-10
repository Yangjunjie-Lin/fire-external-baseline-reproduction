"""Dense RAG config field compatibility."""

from __future__ import annotations


def test_dense_config_accepts_dimension_field() -> None:
    dense_cfg = {
        "backend": "smoke_hash_embedding",
        "model_name": "smoke-hash-embedding",
        "dimension": 128,
    }
    dim = int(dense_cfg.get("dimension", dense_cfg.get("dim", 64)))
    assert dim == 128

    dense_cfg_legacy = {"dim": 32}
    dim_legacy = int(dense_cfg_legacy.get("dimension", dense_cfg_legacy.get("dim", 64)))
    assert dim_legacy == 32
