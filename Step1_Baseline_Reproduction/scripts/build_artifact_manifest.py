"""Write the runtime manifest for the 15 formal full-scope experiments."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_baseline_config, load_paths

MODELS = ["logistic_regression", "xgboost", "lstm_attention", "masked_cnn", "ite_transformer"]
HORIZONS = [0, 48, 72]
ML_MODELS = {"logistic_regression", "xgboost"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_sha256(frame: pd.DataFrame) -> str:
    ids = frame[["subject_id", "hadm_id", "stay_id"]].astype("int64").sort_values(["subject_id", "hadm_id", "stay_id"], kind="stable")
    return hashlib.sha256(ids.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def posix(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    paths = load_paths()
    seed = load_baseline_config()["seed"]
    summary = pd.read_csv(paths["results_root"] / "baseline_summary" / "baseline_results.csv")
    final_pairs = set(zip(summary.model, summary.prediction_window.astype(int)))
    rows = []
    for horizon in HORIZONS:
        for model in MODELS:
            model_dir = paths["model_root"] / model / f"horizon_{horizon}" / "full"
            checkpoint = model_dir / ("model_best.joblib" if model in ML_MODELS else "model_best.pt")
            prediction = paths["results_root"] / "predictions" / "full" / f"horizon_{horizon}" / f"{model}.parquet"
            if model in ML_MODELS:
                preprocessing = model_dir / "preprocessing.joblib"
                scaler = dict(scaler_path=posix(preprocessing), scaler_locator="joblib key: scaler (RobustScaler)",
                              imputer_path=posix(preprocessing), imputer_locator="joblib key: imputer (SimpleImputer, median)",
                              preprocessing_notes="ML preprocessing.joblib is authoritative")
            else:
                scaler = dict(scaler_path=posix(model_dir / "scaling_info.parquet"), scaler_locator="Parquet columns: median, iqr; train-only",
                              imputer_path="", imputer_locator="none; CNN normalized missing cells -> 0; irregular models omit absent events",
                              preprocessing_notes="No separately fitted imputer artifact")
            row = {
                "artifact_id": f"full_{model}_h{horizon}_seed{seed}",
                "model": model, "prediction_window_hours": horizon, "scope": "full",
                "checkpoint_path": posix(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                **{k: scaler[k] for k in ("scaler_path", "scaler_locator", "imputer_path", "imputer_locator")},
                "normalization_table_path": posix(model_dir / "scaling_info.parquet"),
                "feature_list_path": posix(model_dir / "feature_list.json"),
                "config_path": posix(model_dir / "config.json"),
                "test_prediction_path": posix(prediction),
                "test_cohort_sha256": cohort_sha256(pd.read_parquet(prediction)),
            }
            required = [row[k] for k in ("checkpoint_path", "scaler_path", "normalization_table_path", "feature_list_path", "config_path", "test_prediction_path") if row[k]]
            config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            row["all_required_paths_exist"] = str(all(Path(p).exists() for p in required)).lower()
            row["used_for_final_evaluation"] = str((model, horizon) in final_pairs and config["scope"] == "full").lower()
            row["preprocessing_notes"] = scaler["preprocessing_notes"]
            rows.append(row)
    manifest = pd.DataFrame(rows)
    if manifest.test_cohort_sha256.nunique() != 1 or manifest.all_required_paths_exist.ne("true").any() or manifest.used_for_final_evaluation.ne("true").any():
        raise SystemExit("manifest integrity failure: cohort hash, missing paths, or non-final artifact")
    out = paths["metadata_root"] / "artifact_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(json.dumps({"rows": len(manifest), "test_cohort_sha256": manifest.test_cohort_sha256.iloc[0], "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
