#!/usr/bin/env python3
"""Run local CI-equivalent checks and write validation evidence (no fake status)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "outputs" / "diagnostics" / "validation_evidence.json"
OUT_MD = ROOT / "outputs" / "diagnostics" / "validation_evidence.md"
CONTRACT_JSON = ROOT / "outputs/diagnostics/cross_repo_contract.json"
MAIN_REPO = ROOT.parent / "fire-agent-demo"


def _git(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=cwd or ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _run(command: str) -> dict[str, object]:
    proc = subprocess.run(command, shell=True, cwd=ROOT, capture_output=True, text=True)
    status = "passed" if proc.returncode == 0 else "failed"
    return {
        "command": command,
        "exit_code": proc.returncode,
        "status": status,
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def main() -> None:
    commands = [
        "python -m compileall src scripts tests",
        "ruff check src/external_baselines/common/formal_config_validator.py scripts/validate_formal_config.py scripts/collect_validation_evidence.py",
        "python -m pytest -q",
        "python -m build",
        "python scripts/validate_formal_config.py --config configs/experiments/controlled_main_table_v1.yaml.example --allow-placeholders",
        "python scripts/validate_formal_config.py --config configs/experiments/ekell_paper_fidelity_v1.yaml.example --allow-placeholders",
        "python scripts/audit/audit_ekell_fidelity.py",
        "python scripts/check_repository_hygiene.py",
        "python scripts/package_reproducibility_artifact.py --dry-run",
        "python scripts/audit_release_readiness.py",
    ]

    results = [_run(cmd) for cmd in commands]

    cross: dict[str, object] = {
        "executed": False,
        "verified": False,
        "reason": "not_run",
    }
    main_available = MAIN_REPO.is_dir()
    main_branch = _git(["branch", "--show-current"], MAIN_REPO) if main_available else ""
    if main_available and main_branch == "evaluation/benchmark-v1":
        cross_run = _run("python scripts/verify_cross_repo_contract.py")
        results.append(cross_run)
        cross["executed"] = cross_run["exit_code"] == 0
        if CONTRACT_JSON.is_file():
            contract_data = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
            cross["verified"] = bool(contract_data.get("cross_repository_contract_verified"))
            cross["reason"] = contract_data.get("reason") or contract_data.get("note")
            cross.update(
                {
                    k: contract_data.get(k)
                    for k in (
                        "main_project_commit",
                        "baseline_commit",
                        "runner_bundle_checksum",
                        "schema_checksum",
                        "case_count",
                        "prediction_count",
                    )
                }
            )
        else:
            cross["reason"] = "contract_json_missing"
    elif not main_available:
        cross["reason"] = "main_project_checkout_unavailable"
    else:
        cross["reason"] = f"main_project_branch_mismatch:{main_branch}"

    local_passed = all(r["status"] == "passed" for r in results)

    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": _git(["rev-parse", "HEAD"]),
        "baseline_dirty": bool(_git(["status", "--short"])),
        "main_project_available": main_available,
        "main_project_commit": _git(["rev-parse", "HEAD"], MAIN_REPO) if main_available else None,
        "main_project_branch": main_branch or None,
        "commands": results,
        "local_ci_equivalent_passed": local_passed,
        "ci_workflow_configured": (ROOT / ".github/workflows/ci.yml").is_file(),
        "ci_remote_status_known": False,
        "cross_repository_contract": cross,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Validation evidence",
        "",
        f"Generated: `{evidence['generated_at']}`",
        "",
        f"- baseline_commit: `{evidence['baseline_commit']}`",
        f"- local_ci_equivalent_passed: **{local_passed}**",
        "- ci_remote_status_known: **false**",
        f"- cross_repository_contract.executed: **{cross['executed']}**",
        f"- cross_repository_contract.verified: **{cross['verified']}**",
        "",
        "## Commands",
    ]
    for item in results:
        lines.append(f"- `{item['command']}` → {item['status']} (exit {item['exit_code']})")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"written": [str(OUT_JSON), str(OUT_MD)], "local_ci_equivalent_passed": local_passed}, indent=2))
    if not local_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
