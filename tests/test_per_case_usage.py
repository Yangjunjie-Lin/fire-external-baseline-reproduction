from __future__ import annotations

import json

from external_baselines.runner import generate_predictions


def test_per_case_usage_is_not_cumulative(tmp_path):
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(
        json.dumps({
            "scenarios": [
                {"scenario_id": "one", "scenario_text": "Smoke"},
                {"scenario_id": "two", "scenario_text": "Smoke"},
            ]
        }),
        encoding="utf-8",
    )
    rows = generate_predictions(
        methods=["direct_llm"],
        dataset=dataset,
        output_path=tmp_path / "out.jsonl",
        manifest_path=None,
        config={
            "llm": {"provider": "heuristic"},
            "paths": {"corpus_dir": "data/corpus"},
        },
    )
    runtimes = [row["method_specific"]["runtime"] for row in rows]
    assert [runtime["case_llm_calls"] for runtime in runtimes] == [1, 1]
    assert [runtime["llm_calls"] for runtime in runtimes] == [1, 1]
    assert runtimes[1]["case_total_tokens"] == runtimes[1]["token_usage"]["total_tokens"]
