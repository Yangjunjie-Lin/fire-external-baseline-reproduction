#!/usr/bin/env python3
"""Release-readiness audit (engineering gates; honest empirical flags)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_JSON = ROOT / "outputs" / "diagnostics" / "release_readiness.json"
OUT_MD = ROOT / "outputs" / "diagnostics" / "release_readiness.md"

ENGINEERING_GATE_KEYS = (
    "method_registry_converged",
    "formal_manifest_unique",
    "canonical_method_ids_only",
    "legacy_isolated",
    "formal_config_validator_present",
    "smoke_config_separated",
    "gold_isolation_tests_present",
    "schema_authority_clear",
    "external_schema_enforcement_components_present",
    "checksum_enforcement_components_present",
    "interop_contract_tests_present",
    "cross_repository_contract_tool_ready",
    "environment_dependency_spec_present",
    "artifact_packager_present",
    "data_card_templates_present",
    "model_card_templates_present",
    "run_card_templates_present",
    "ci_workflow_config_present",
)

EMPIRICAL_GATE_KEYS = (
    "cross_repo_contract_verified",
    "real_shared_llm_run",
    "real_chatglm_run",
    "actual_lightrag",
    "actual_microsoft_graphrag",
    "expert_evaluation",
    "formal_statistics",
)

PAPER_GATE_KEYS = ("paper_ready",)


def _exists(rel: str) -> bool:
    return (ROOT / rel).is_file()


def _read_text(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _manifest_template_flag_exact_true(flag: str) -> bool:
    text = _read_text("configs/experiments/controlled_main_table_v1.yaml.example")
    if not text:
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{flag}:") and stripped.split(":", 1)[1].strip() == "true":
            return True
    return False


def _source_contains(rel: str, needle: str) -> bool:
    return needle in _read_text(rel)


def _external_schema_enforcement_components_present() -> bool:
    manifest_flag = _manifest_template_flag_exact_true("require_external_schema")
    staged_validator = (
        _source_contains(
            "scripts/run_decision_comparison_suite.py",
            "prediction_schema_path: Path",
        )
        and _source_contains(
            "scripts/run_decision_comparison_suite.py",
            "require_external_schema=True",
        )
    )
    contract_tests = _exists("tests/test_external_schema_validation.py") or _exists(
        "tests/interop/test_main_project_contract.py"
    )
    return manifest_flag and staged_validator and contract_tests


def _checksum_enforcement_components_present() -> bool:
    manifest_flag = _manifest_template_flag_exact_true("require_bundle_checksum")
    bundle_validator = _source_contains(
        "src/external_baselines/interop/bundle.py",
        "def validate_bundle_checksum",
    ) and _source_contains(
        "src/external_baselines/interop/bundle.py",
        "_verify_file_checksums",
    )
    staged_rehash = _source_contains(
        "scripts/run_decision_comparison_suite.py",
        "artifact_hashes",
    ) and _source_contains(
        "scripts/run_decision_comparison_suite.py",
        "sha256_file",
    )
    tamper_tests = _exists("tests/test_bundle_integrity.py") and _source_contains(
        "tests/test_bundle_integrity.py",
        "tamper",
    )
    return manifest_flag and bundle_validator and staged_rehash and tamper_tests


def _read_deprecated(path: str) -> bool:
    p = ROOT / path
    if not p.is_file():
        return True
    text = p.read_text(encoding="utf-8")
    return "deprecated: true" in text


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _engineering_gate_values() -> dict[str, bool]:
    return {
        "method_registry_converged": _exists("src/external_baselines/method_registry.py"),
        "formal_manifest_unique": _exists("configs/experiments/controlled_main_table_v1.yaml.example")
        and _read_deprecated("configs/experiments/paper_main_table_v1.yaml.example"),
        "canonical_method_ids_only": _exists("configs/frozen/ekell_controlled_shared_llm_v1.yaml"),
        "legacy_isolated": _exists("scripts/legacy/README.md"),
        "formal_config_validator_present": _exists("src/external_baselines/common/formal_config_validator.py"),
        "smoke_config_separated": _exists("configs/smoke/deterministic_heuristic.yaml"),
        "gold_isolation_tests_present": _exists("tests/test_gold_isolation.py"),
        "schema_authority_clear": _exists("schemas/firebench_interop_v1/README.md"),
        "external_schema_enforcement_components_present": _external_schema_enforcement_components_present(),
        "checksum_enforcement_components_present": _checksum_enforcement_components_present(),
        "interop_contract_tests_present": _exists("tests/interop/test_main_project_contract.py"),
        "cross_repository_contract_tool_ready": _exists("scripts/verify_cross_repo_contract.py"),
        "environment_dependency_spec_present": _exists("constraints.txt") or _exists("requirements.lock"),
        "artifact_packager_present": _exists("scripts/package_reproducibility_artifact.py"),
        "data_card_templates_present": _exists("docs/cards/data_card_template.md"),
        "model_card_templates_present": _exists("docs/cards/model_card_template.md"),
        "run_card_templates_present": _exists("docs/cards/run_card_template.md"),
        "ci_workflow_config_present": _exists(".github/workflows/ci.yml"),
    }


def _empirical_gate_values() -> dict[str, bool]:
    cross_repo_path = ROOT / "outputs/diagnostics/cross_repo_contract.json"
    cross_verified = False
    if cross_repo_path.is_file():
        try:
            value = json.loads(cross_repo_path.read_text(encoding="utf-8")).get(
                "cross_repository_contract_verified",
                False,
            )
            cross_verified = value if type(value) is bool else False
        except json.JSONDecodeError:
            cross_verified = False
    return {
        "cross_repo_contract_verified": cross_verified,
        "real_shared_llm_run": False,
        "real_chatglm_run": False,
        "actual_lightrag": False,
        "actual_microsoft_graphrag": False,
        "expert_evaluation": False,
        "formal_statistics": False,
    }


def build_report() -> dict[str, Any]:
    engineering_gates = _engineering_gate_values()
    empirical_gates = _empirical_gate_values()
    paper_gates = {"paper_ready": False}
    engineering_failed = [key for key, value in engineering_gates.items() if not value]
    engineering_passed = sum(1 for value in engineering_gates.values() if value)
    engineering_total = len(engineering_gates)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git(["log", "-1", "--oneline"]),
        "ci_result_verified_externally": False,
        "assurance_model": {
            "engineering_gates": "structural",
            "behavioral_validation": "pytest_and_formal_e2e",
            "ci_result_verified_externally": False,
        },
        "engineering": {
            "ready": not engineering_failed,
            "passed": engineering_passed,
            "total": engineering_total,
            "failed_gates": engineering_failed,
            "gates": engineering_gates,
        },
        "empirical": {
            "ready": all(empirical_gates.values()),
            "gates": empirical_gates,
        },
        "paper": {
            "ready": all(paper_gates.values()),
            "gates": paper_gates,
        },
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Release readiness",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Engineering",
        f"- ready: **{report['engineering']['ready']}**",
        f"- passed: **{report['engineering']['passed']}/{report['engineering']['total']}**",
        "",
        "## Empirical",
        f"- ready: **{report['empirical']['ready']}**",
        "",
        "## Paper",
        f"- ready: **{report['paper']['ready']}**",
        "",
    ]
    for key, value in report["engineering"]["gates"].items():
        lines.append(f"- engineering `{key}`: **{value}**")
    for key, value in report["empirical"]["gates"].items():
        lines.append(f"- empirical `{key}`: **{value}**")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit release readiness")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write the report but always exit 0.",
    )
    args = parser.parse_args(argv)
    report = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.report_only:
        raise SystemExit(0)
    if not report["engineering"]["ready"]:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
