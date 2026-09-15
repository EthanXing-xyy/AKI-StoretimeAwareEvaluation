"""Finalize formal model artifacts without changing fitted parameters."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_paths


MODELS = ["logistic_regression", "xgboost", "lstm_attention", "masked_cnn", "ite_transformer"]
HORIZONS = [0, 48, 72]


def main() -> None:
    paths = load_paths()
    actions: list[dict] = []
    for horizon in HORIZONS:
        data_dir = paths["processed_data_root"] / "mimiciv" / "full" / f"horizon_{horizon}"
        for model_name in MODELS:
            model_dir = paths["model_root"] / model_name / f"horizon_{horizon}" / "full"
            if not model_dir.exists():
                continue
            copied = []
            for name in ["scaling_info.parquet", "feature_list.json"]:
                source = data_dir / name
                if source.exists():
                    shutil.copy2(source, model_dir / name)
                    copied.append(name)

            history_path = model_dir / "training_history.csv"
            history_created = False
            if model_name in ["logistic_regression", "xgboost"] and not history_path.exists():
                model = joblib.load(model_dir / "model_best.joblib")
                if model_name == "logistic_regression":
                    maximum = int(np.max(model.n_iter_))
                    history = [{"iteration": maximum, "converged": maximum < model.max_iter}]
                else:
                    values = model.evals_result().get("validation_0", {}).get("logloss", [])
                    history = [{"iteration": index + 1, "val_logloss": value} for index, value in enumerate(values)]
                pd.DataFrame(history).to_csv(history_path, index=False)
                history_created = True
            actions.append({"model": model_name, "horizon": horizon, "copied": copied, "history_created": history_created})

    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "actions": actions}
    output = paths["logs_root"] / "artifact_finalization.json"
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"model_horizon_directories": len(actions), "log": str(output)}, indent=2))


if __name__ == "__main__":
    main()
