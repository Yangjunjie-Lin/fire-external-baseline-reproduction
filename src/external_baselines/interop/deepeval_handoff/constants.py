"""Frozen identifiers and paths for DeepEval handoff v1."""

from pathlib import Path

CONTRACT_ID = "fireagent-external-prediction-v1"
CONTRACT_SOURCE_REPOSITORY = "Yangjunjie-Lin/fire-agent-demo"
CONTRACT_SOURCE_PATH = Path("schemas/deepeval_external_v1/external_prediction.schema.json")
SNAPSHOT_PATH = Path("schemas/deepeval_handoff/fireagent_external_prediction_v1.snapshot.schema.json")
PROVENANCE_PATH = Path("schemas/deepeval_handoff/contract_provenance.json")
HANDOFF_MANIFEST_VERSION = "deepeval-handoff-manifest-v1"
HANDOFF_PROTOCOL = "fireagent_deepeval_handoff_v1"
VALIDATION_REPORT_VERSION = "deepeval-handoff-validation-report-v1"
BUNDLE_VERSION = "deepeval-handoff-bundle-v1"
DEFAULT_TOP_K = 5
CONTEXT_SELECTION_POLICY = "original_rank_prefix"
DIRECT_METHOD = "direct_llm"
FORMAL_RAG_METHODS = frozenset(
    {
        "bm25_rag",
        "dense_rag",
        "hybrid_rag",
        "ekell_style_controlled_shared_llm",
    }
)
FORBIDDEN_HANDOFF_KEYS = frozenset(
    {
        "gold",
        "expected_output",
        "expected_actions",
        "expected_risk",
        "reference_answer",
        "label",
        "grading",
        "metric_score",
        "deepeval_score",
        "judge_reason",
    }
)
REPOSITORY_NAME = "Yangjunjie-Lin/fire-external-baseline-reproduction"
