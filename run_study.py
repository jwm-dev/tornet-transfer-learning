"""Run the real TorNet study in order; identical reruns reuse sealed artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def record(stage, status, **details):
    (ROOT / "results").mkdir(exist_ok=True)
    with (ROOT / "results/pipeline_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                                 "stage": stage, "status": status, **details}) + "\n")


def ensure_data():
    from data_pipeline import digest, prepare_data

    receipt = ROOT / "results/data_audit.json"
    manifest = ROOT / "data/runtime_manifest.csv"
    if receipt.exists() and manifest.exists():
        audit = json.loads(receipt.read_text())
        expected = {"runtime_manifest_sha256": manifest,
                    "planned_manifest_sha256": ROOT / "design/planned_manifest.csv",
                    "planned_config_sha256": ROOT / "planned_config.json"}
        if audit.get("status") == "passed" and all(audit.get(k) == digest(p) for k, p in expected.items()):
            record("data", "reused_verified_audit", valid_samples=audit["valid_samples"])
            return audit
    if (ROOT / "results/final_decision.json").exists():
        raise RuntimeError("Data/configuration no longer matches locked study; preserve its holdout.")
    record("data", "started")
    audit = prepare_data()
    record("data", "passed", valid_samples=audit["valid_samples"])
    return audit


def require_verified_data():
    """A standalone downstream stage must not skip the physical-data audit."""
    from data_pipeline import digest

    receipt = ROOT / "results/data_audit.json"
    if not receipt.exists():
        raise RuntimeError("Run the data stage and pass physical-frame QA before downstream stages.")
    audit = json.loads(receipt.read_text(encoding="utf-8"))
    expected = {"runtime_manifest_sha256": ROOT / "data/runtime_manifest.csv",
                "planned_manifest_sha256": ROOT / "design/planned_manifest.csv",
                "planned_config_sha256": ROOT / "planned_config.json"}
    if audit.get("status") != "passed" or any(not path.exists() or audit.get(key) != digest(path)
                                                for key, path in expected.items()):
        raise RuntimeError("Passed data audit is missing or its hashes no longer match this study.")
    return audit


def ensure_features():
    # Call the public CLI: it validates all hashes before accepting a cache.
    require_verified_data()
    command = [sys.executable, str(ROOT / "features.py"),
               "--manifest", str(ROOT / "data/runtime_manifest.csv"),
               "--data-root", str(ROOT / "data/netcdf"),
               "--checkpoint", str(ROOT / "probe/tornado_detector_baseline.keras"),
               "--batch-size", "16"]
    record("features", "started_or_validating_cache")
    subprocess.run(command, check=True)
    receipt = json.loads((ROOT / "data/feature_provenance.json").read_text())
    record("features", "completed", samples=receipt["samples"])
    return receipt


def run(stage="all", execute_notebook=True):
    if stage in ("data", "all"):
        ensure_data()
    if stage in ("features", "all"):
        ensure_features()
    if stage in ("train", "all"):
        require_verified_data()
        from experiments import train
        decision = train()
        print("Validation-selected configuration:", decision["selected_configuration"], flush=True)
    if stage in ("test", "all"):
        require_verified_data()
        from experiments import test_once
        final = test_once()
        print("Sealed final evaluation:", final["selected_configuration"], flush=True)
    if stage in ("artifacts", "all"):
        from verify_results import verify
        verify()
        from build_artifacts import build_all
        print(build_all(execute_notebook=execute_notebook), flush=True)


def continue_after_data(execute_notebook=True):
    """Wait for the separate acquisition process, then run all downstream work."""
    audit_path = ROOT / "results/data_audit.json"
    print("Waiting for the acquisition process to write the complete physical-data audit.", flush=True)
    record("await_data_audit", "started")
    while not audit_path.exists():
        time.sleep(15)
    require_verified_data()
    record("await_data_audit", "passed")
    for stage in ("features", "train", "test", "artifacts"):
        run(stage, execute_notebook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=("all", "data", "features", "train", "test", "artifacts"))
    parser.add_argument("--no-execute-notebook", action="store_true")
    parser.add_argument("--continue-after-data", action="store_true",
                        help="Wait for another acquisition process's audit, then features/train/test/artifacts")
    args = parser.parse_args()
    try:
        if args.continue_after_data:
            continue_after_data(not args.no_execute_notebook)
        else:
            run(args.stage, not args.no_execute_notebook)
    except Exception as exc:
        record(args.stage, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
