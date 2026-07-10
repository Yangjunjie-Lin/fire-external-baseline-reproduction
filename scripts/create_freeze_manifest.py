#!/usr/bin/env python3
"""Create a freeze manifest draft from configs + selected DEV evidence (manual confirm still required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_baselines.common.experiment_manifest import load_experiment_manifest  # noqa: E402
from external_baselines.common.freeze_manifest import build_freeze_manifest_payload  # noqa: E402
from external_baselines.common.io import write_json  # noqa: E402
from external_baselines.interop.bundle import load_runner_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create freeze manifest draft (does not auto-freeze).")
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--selected-dev-run", required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    experiment = load_experiment_manifest(args.experiment_manifest)
    raw = experiment.get("raw") or {}
    method_paths = {
        str(e.get("method_id")): str(e.get("config"))
        for e in (raw.get("methods") or [])
        if isinstance(e, dict) and e.get("method_id") and e.get("config")
    }
    bundle_checksum = None
    corpus_checksum = None
    schema_checksum = None
    bundle_path = args.bundle or experiment.get("bundle")
    if bundle_path:
        try:
            bundle = load_runner_bundle(bundle_path)
            bundle_checksum = bundle.get("consumer_computed_bundle_hash") or bundle.get(
                "producer_declared_checksum"
            )
            schema_checksum = bundle.get("prediction_schema_sha256")
            corpus_manifest = bundle.get("corpus_manifest") or {}
            if isinstance(corpus_manifest, dict):
                corpus_checksum = corpus_manifest.get("aggregate_sha256")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not load bundle for checksums: {exc}", file=sys.stderr)

    payload = build_freeze_manifest_payload(
        experiment_manifest_path=args.experiment_manifest,
        experiment_raw=raw,
        selected_dev_run=args.selected_dev_run,
        bundle_checksum=bundle_checksum,
        corpus_checksum=corpus_checksum,
        schema_checksum=schema_checksum,
        method_config_paths=method_paths,
    )
    write_json(args.output, payload)
    print(f"Wrote freeze manifest draft to {args.output}")
    print("Manual confirmation still required before setting freeze_status=frozen.")


if __name__ == "__main__":
    main()
