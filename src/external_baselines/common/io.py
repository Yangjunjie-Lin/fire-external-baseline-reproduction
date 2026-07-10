from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is in requirements, keep fallback explicit.
    yaml = None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_yaml(path: str | Path, default: dict | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        return dict(default or {})
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs. Run `pip install -r requirements.txt`.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: dict, b: dict) -> dict:
    result = dict(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(*paths: str | Path) -> dict:
    config: dict = {}
    for path in paths:
        p = Path(path)
        if p.exists():
            config = deep_merge(config, read_yaml(p))
    return config


GOLD_KEYS = {
    "expected",
    "ground_truth",
    "labels",
    "gold",
    "annotation",
    "annotations",
    "annotation_notes",
    "target_outputs",
    "target_output",
    "evaluator_hints",
    "expert_scores",
    "reference_answer",
}


def _scenario_id(record: dict[str, Any]) -> str:
    return str(
        record.get("scenario_id")
        or record.get("case_id")
        or record.get("id")
        or record.get("name")
        or "unknown_scenario"
    )


def _scenario_text(record: dict[str, Any]) -> str:
    if record.get("scenario_text"):
        return str(record["scenario_text"])
    if record.get("input") and isinstance(record["input"], str):
        return record["input"]
    if record.get("description"):
        return str(record["description"])
    parts = []
    for key in ["incident_type", "location", "hazards", "dynamic_state", "prompt", "task"]:
        value = record.get(key)
        if value is not None:
            parts.append(f"{key}: {value}")
    return ". ".join(parts) or json.dumps(
        {k: v for k, v in record.items() if str(k).lower() not in GOLD_KEYS},
        ensure_ascii=False,
    )


def flatten_scenario(record: dict[str, Any]) -> dict[str, Any]:
    """Convert flexible scenario-matrix records into dataset records.

    Gold/expected is retained only for offline evaluation loaders. Prediction
    generation must call ``to_prediction_input`` so pipelines never see gold.
    """
    expected = record.get("expected") or record.get("ground_truth") or record.get("labels") or {}
    return {
        "scenario_id": _scenario_id(record),
        "case_id": _scenario_id(record),
        "scenario_text": _scenario_text(record),
        "language": record.get("language") or record.get("lang"),
        "input_mode": record.get("input_mode") or record.get("mode"),
        "expected": expected if isinstance(expected, dict) else {"expected": expected},
        "source_record": record,
    }


def to_prediction_input(scenario: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Strip gold/labels/annotations/target outputs before pipeline execution.

    Allowed fields for prediction generation:
    - case_id / scenario_id
    - scenario text
    - language
    - input mode
    - allowed dynamic snapshots (input-only)
    - allowed corpus/config pointers (no gold)
    """
    config = config or {}
    source = scenario.get("source_record") if isinstance(scenario.get("source_record"), dict) else scenario
    case_id = str(scenario.get("case_id") or scenario.get("scenario_id") or _scenario_id(source))
    text = str(scenario.get("scenario_text") or _scenario_text(source))

    allowed_snapshots = None
    for key in ("allowed_dynamic_snapshots", "dynamic_snapshots", "dynamic_state", "input_dynamic_state"):
        if key in source and str(key).lower() not in GOLD_KEYS:
            # Only pass dynamic state when it is an input observation, not a label.
            if key == "dynamic_state" and isinstance(source.get(key), dict):
                # Drop nested gold-like keys if present.
                allowed_snapshots = {
                    k: v for k, v in source[key].items() if str(k).lower() not in GOLD_KEYS
                }
            else:
                allowed_snapshots = source.get(key)
            break

    paths = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    return {
        "case_id": case_id,
        "scenario_id": case_id,
        "scenario_text": text,
        "language": scenario.get("language") or source.get("language") or source.get("lang"),
        "input_mode": scenario.get("input_mode") or source.get("input_mode") or source.get("mode"),
        "allowed_dynamic_snapshots": allowed_snapshots,
        "allowed_corpus_dir": paths.get("corpus_dir"),
        "allowed_config_keys": sorted(
            k for k in ("retrieval", "llm", "paths", "ekell_style", "scenario_parser", "normalization", "dense_rag", "hybrid_rag")
            if k in config
        ),
    }


def assert_no_gold_in_prediction_input(prediction_input: dict[str, Any]) -> None:
    """Raise if forbidden gold/target keys appear in a prediction input."""
    blob = json.dumps(prediction_input, ensure_ascii=False, default=str).lower()
    for key in GOLD_KEYS:
        if f'"{key.lower()}"' in blob or f"'{key.lower()}'" in blob:
            # Allow the word only inside scenario_text narrative, not as structured keys.
            if key.lower() in {str(k).lower() for k in prediction_input.keys()}:
                raise AssertionError(f"Gold/target key leaked into prediction input: {key}")
            nested = prediction_input.get("allowed_dynamic_snapshots")
            if isinstance(nested, dict) and key.lower() in {str(k).lower() for k in nested.keys()}:
                raise AssertionError(f"Gold/target key leaked into allowed_dynamic_snapshots: {key}")


def load_scenarios(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    raw = read_json(path)
    if isinstance(raw, dict):
        for key in ["scenarios", "scenario_matrix", "items", "data", "records"]:
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"Scenario file must contain a list or dict of scenarios: {path}")
    scenarios = [flatten_scenario(x) for x in raw if isinstance(x, dict)]
    if limit is not None and limit >= 0:
        scenarios = scenarios[:limit]
    return scenarios


def load_expected_by_id(path: str | Path, limit: int | None = None) -> dict[str, Any]:
    """Load gold/expected keyed by case id for evaluation only (never for generation)."""
    scenarios = load_scenarios(path, limit=limit)
    return {str(s["scenario_id"]): s.get("expected", {}) for s in scenarios}
