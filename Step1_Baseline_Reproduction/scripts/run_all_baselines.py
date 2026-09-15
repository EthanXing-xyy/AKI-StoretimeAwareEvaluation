"""Run the requested baseline experiments sequentially to control resources."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import load_paths  # noqa: E402
MODELS = ["logistic_regression", "xgboost", "lstm_attention", "masked_cnn", "ite_transformer"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential, resource-safe baseline runner")
    parser.add_argument("--scope", choices=["smoke", "full"], default="full")
    parser.add_argument("--horizons", type=int, nargs="+", choices=[0, 48, 72], default=[0])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    paths = load_paths()
    for horizon in args.horizons:
        data_dir = paths["processed_data_root"] / "mimiciv" / args.scope / f"horizon_{horizon}"
        if not (data_dir / "master.parquet").exists():
            subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_model_inputs.py"), "--horizon", str(horizon), "--scope", args.scope], check=True, cwd=PROJECT_ROOT)
        for model in MODELS:
            subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "run_baseline.py"), "--model", model, "--horizon", str(horizon), "--scope", args.scope, "--device", args.device], check=True, cwd=PROJECT_ROOT)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_baselines.py"), "--scope", args.scope], check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
