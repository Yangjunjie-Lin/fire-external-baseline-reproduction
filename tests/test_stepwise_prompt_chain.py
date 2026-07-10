from __future__ import annotations

from external_baselines.common.llm_client import HeuristicLLMClient
from external_baselines.ekell_style.logical_query import validate_query
from external_baselines.ekell_style.stepwise_prompt_chain import run_stepwise_prompt_chain


def test_stepwise_chain_records_reproducibility_trace() -> None:
    kg = {
        "triples": [
            {
                "triple_id": "t1",
                "subject": "fire",
                "predicate": "requires",
                "object": "evacuation",
                "source_id": "s1",
                "chunk_id": "c1",
            }
        ]
    }
    ast = {"operation": "projection", "entity": "fire", "relation": "requires"}
    validated = validate_query(ast, kg)
    result = run_stepwise_prompt_chain(
        validated_ast=validated,
        kg_contexts=kg["triples"],
        llm=HeuristicLLMClient(),
        query="What does the fire require?",
    )

    assert len(result["steps"]) == 1
    step = result["steps"][0]
    assert step["operation"] == "projection"
    assert step["parsing_status"] == "parsed"
    assert len(step["prompt_template_hash"]) == 64
    assert len(step["rendered_prompt_hash"]) == 64
    assert step["step_dependencies"] == []
    assert result["final"]["step_dependencies"] == [step["step_id"]]
    assert set(result["allowed_evidence_ids"]) >= {"t1", "s1", "c1"}
