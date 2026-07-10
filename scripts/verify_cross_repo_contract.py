#!/usr/bin/env python3
"""Cross-repository contract verification (heuristic smoke; no paid API)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "diagnostics" / "cross_repo_contract.json"
MAIN_BUNDLE = ROOT.parent / "fire-agent-demo" / "artifacts" / "firebench_interop_v1" / "runner_seed_curated"


def _git(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> None:
    result: dict[str, object] = {
        "cross_repository_contract_verified": False,
        "reason": "unknown",
    }
    main_repo = ROOT.parent / "fire-agent-demo"
    if not main_repo.is_dir():
        result["reason"] = "main_project_checkout_unavailable"
    else:
        branch = _git(["branch", "--show-current"], main_repo)
        if branch != "evaluation/benchmark-v1":
            result["reason"] = f"main_project_branch_mismatch:{branch}"
        elif not MAIN_BUNDLE.is_dir():
            result["reason"] = "runner_bundle_unavailable"
        else:
            try:
                from external_baselines.interop.bundle import load_runner_bundle, validate_bundle_checksum
                from external_baselines.common.checksums import sha256_file

                bundle = load_runner_bundle(MAIN_BUNDLE)
                checksum = validate_bundle_checksum(bundle)
                if not checksum.get("ok"):
                    result["reason"] = "bundle_checksum_failed"
                else:
                    # Run lightweight smoke via existing script logic
                    import runpy

                    runpy.run_path(str(ROOT / "scripts" / "smoke_main_runner_bundle.py"), run_name="__main__")
                    smoke_report = ROOT / "outputs" / "interop" / "smoke_seed_curated" / "diagnostics" / "smoke_report.json"
                    smoke = json.loads(smoke_report.read_text(encoding="utf-8")) if smoke_report.is_file() else {}
                    if smoke.get("schema_failure_count", 1) == 0:
                        result = {
                            "cross_repository_contract_verified": True,
                            "main_project_commit": _git(["log", "-1", "--format=%H"], main_repo),
                            "baseline_commit": _git(["log", "-1", "--format=%H"], ROOT),
                            "runner_bundle_checksum": bundle.get("consumer_computed_bundle_hash"),
                            "schema_checksum": sha256_file(bundle["prediction_schema_path"])
                            if bundle.get("prediction_schema_path")
                            else None,
                            "case_count": smoke.get("runner_case_count", 0),
                            "prediction_count": smoke.get("canonical_prediction_count", 0),
                            "note": "Contract verification only; not formal experiment verification.",
                        }
                    else:
                        result["reason"] = "smoke_schema_failures"
            except SystemExit as exc:
                result["reason"] = f"smoke_failed:{exc.code}"
            except Exception as exc:  # noqa: BLE001
                result["reason"] = f"error:{type(exc).__name__}:{exc}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result.get("cross_repository_contract_verified"):
        raise SystemExit(0)  # non-fatal when checkout missing


if __name__ == "__main__":
    main()
