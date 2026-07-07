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


def flatten_scenario(record: dict[str, Any]) -> dict[str, Any]:
    """Convert flexible scenario-matrix records into baseline input records."""
    sid = (
        record.get("scenario_id")
        or record.get("id")
        or record.get("name")
        or record.get("case_id")
        or "unknown_scenario"
    )
    if record.get("scenario_text"):
        text = str(record["scenario_text"])
    elif record.get("input") and isinstance(record["input"], str):
        text = record["input"]
    elif record.get("description"):
        text = str(record["description"])
    else:
        parts = []
        for key in ["incident_type", "location", "hazards", "dynamic_state", "prompt", "task"]:
            value = record.get(key)
            if value is not None:
                parts.append(f"{key}: {value}")
        text = ". ".join(parts) or json.dumps(record, ensure_ascii=False)
    expected = record.get("expected") or record.get("ground_truth") or record.get("labels") or {}
    return {
        "scenario_id": str(sid),
        "scenario_text": text,
        "expected": expected if isinstance(expected, dict) else {"expected": expected},
        "source_record": record,
    }


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
