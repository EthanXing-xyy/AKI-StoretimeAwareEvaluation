"""Verify that Steps 1 and 2 remain unchanged during the stress test."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import importlib.util
import json
from pathlib import Path

STRESS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STRESS_ROOT.parent
ROOTS = {
    "baseline": REPO_ROOT / "Step1_Baseline_Reproduction",
    "continuous_monitoring": REPO_ROOT / "Step2_Continuous_Monitoring",
}


def load_integrity_module():
    path = REPO_ROOT / "Step2_Continuous_Monitoring" / "scripts" / "baseline_integrity.py"
    spec = importlib.util.spec_from_file_location("baseline_integrity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LOG_DIR = REPO_ROOT / "work" / "Step4_Storetime_Aware_Stress_Test" / "logs"
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["snapshot", "compare"])
    parser.add_argument("--label")
    parser.add_argument("--label-before")
    parser.add_argument("--label-after")
    args = parser.parse_args()
    integrity = load_integrity_module()
    if args.command == "snapshot":
        for name, root in ROOTS.items():
            integrity.snapshot(f"{name}_{args.label}", root)
        return
    ok = True
    for name in ROOTS:
        try:
            integrity.compare(f"{name}_{args.label_before}", f"{name}_{args.label_after}")
        except SystemExit:
            ok = False
    print(json.dumps({"all_identical": ok}))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
