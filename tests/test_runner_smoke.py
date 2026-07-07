import json

from external_baselines.runner import run_methods


def test_run_methods_direct_llm_tiny_dataset(tmp_path):
    dataset = tmp_path / "scenarios.json"
    dataset.write_text(json.dumps({"scenarios": [{"scenario_id": "s1", "scenario_text": "Small fire with smoke."}]}), encoding="utf-8")
    output = tmp_path / "out.jsonl"
    metrics = tmp_path / "metrics.csv"
    report = tmp_path / "report.md"
    manifest = tmp_path / "manifest.json"
    rows = run_methods(methods=["direct_llm"], dataset=dataset, limit=1, output_path=output, metrics_path=metrics, report_path=report, manifest_path=manifest)
    assert len(rows) == 1
    assert output.exists()
    assert metrics.exists()
    assert report.exists()
    assert manifest.exists()
