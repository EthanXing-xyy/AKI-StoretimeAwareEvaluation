# %%
"""
Online Inference (Streaming Simulation) for XGBoost and Logistic Regression baselines.

Simulates real-time prediction: at each hourly timestamp, the model makes a prediction
using all data available up to that point (LOCF + median imputation -> flat features).

Usage:
    cd code/train_ml/scripts
    python experiment_online_inference.py --develop_set ilsan --prediction_window_size 0
    python experiment_online_inference.py --develop_set ilsan --prediction_window_size 48
    python experiment_online_inference.py --develop_set ilsan --prediction_window_size 72
"""

import argparse
import gc
import joblib
import joblib as jl
import numpy as np
import os
import pandas as pd
import sys

from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SCRIPT_DIR.parent))
from data_utils import (
    ALL_FEATURES,
    load_scaling_info, locf_impute, ScalingParams,
)

# Reuse aggregate_online_results from the CNN pipeline
# Import via importlib to avoid package name collision (train_ml/scripts vs train/scripts)
import importlib.util
_online_inf_path = _SCRIPT_DIR.parent.parent / "train" / "scripts" / "online_inference.py"
_spec = importlib.util.spec_from_file_location("cnn_online_inference", str(_online_inf_path))
_online_inf_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_online_inf_module)
aggregate_online_results = _online_inf_module.aggregate_online_results
compute_online_metrics = _online_inf_module.compute_online_metrics

SEED = 1004


def run_online_inference_ml(
    model,
    meta_table: pd.DataFrame,
    data_dir: str,
    scaling_info: pd.DataFrame,
    time_resolution: float = 1.0,
    min_observations: int = 1,
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run online (streaming) inference for an sklearn-compatible model.

    Optimized: LOCF is computed once per patient, then each timestamp
    just indexes into the pre-computed LOCF array.
    Batched: all timestamps for a patient are predicted in a single batch call.
    """
    meta_table = meta_table.copy().reset_index(drop=True)
    if meta_table["sex"].dtype == object:
        meta_table["sex"] = meta_table["sex"].map({"M": 0, "F": 1})

    params = ScalingParams(scaling_info)
    results = {}

    iterator = range(len(meta_table))
    if show_progress:
        iterator = tqdm(iterator, desc="Online Inference (ML)")

    for idx in iterator:
        row = meta_table.iloc[idx]
        visit_id = row["visit_id"]
        label = row["label"]
        meta = np.array([row["age"], row["sex"], row["bmi"]], dtype=np.float64)

        data_path = f"{data_dir}/{visit_id}.gz"
        try:
            data_dict = joblib.load(data_path)
        except FileNotFoundError:
            continue

        data = data_dict["data"]  # (seq_len, 48)
        total_steps = len(data)

        # Pre-compute LOCF for entire sequence ONCE
        locf_data = locf_impute(data)

        # Pre-compute normalized demographics ONCE
        demo_norm = (meta - params.demo_medians) / params.demo_iqrs

        # Determine unique timestamps
        timestamps = np.arange(total_steps, dtype=float)
        if time_resolution > 0:
            rounded_timestamps = np.round(timestamps / time_resolution) * time_resolution
            unique_timestamps = np.unique(rounded_timestamps)
        else:
            unique_timestamps = np.unique(timestamps)

        # Collect all timestamp feature vectors into a batch
        batch_features = []
        valid_timestamps = []
        observation_counts = []

        for current_time in unique_timestamps:
            if time_resolution > 0:
                time_mask = rounded_timestamps <= current_time
            else:
                time_mask = timestamps <= current_time

            n_obs = int(time_mask.sum())
            if n_obs < min_observations:
                continue

            # Take last row of LOCF data up to this point
            last_row = locf_data[n_obs - 1, :].copy()

            # Median imputation for remaining NaNs
            still_nan = np.isnan(last_row)
            if still_nan.any():
                last_row[still_nan] = params.ts_medians[still_nan]

            # Normalize time-series
            last_row = (last_row - params.ts_medians) / params.ts_iqrs

            features = np.concatenate([last_row, demo_norm])
            batch_features.append(features)
            valid_timestamps.append(current_time)
            observation_counts.append(n_obs)

        if not batch_features:
            continue

        # Batched prediction: single call for all timestamps
        X_batch = np.array(batch_features, dtype=np.float32)
        proba_batch = model.predict_proba(X_batch)  # (n_timestamps, n_classes)

        # Build result DataFrame
        patient_results = []
        for i in range(len(valid_timestamps)):
            result = {
                "timestamp": valid_timestamps[i],
                "n_observations": observation_counts[i],
                "visit_id": visit_id,
                "label": label,
            }
            for c in range(proba_batch.shape[1]):
                result[f"pred_proba_{c}"] = proba_batch[i, c]
            patient_results.append(result)

        results[visit_id] = pd.DataFrame(patient_results)

    return results


def experiment_online_inference(config: dict):
    """Run online inference for ML baselines."""
    np.random.seed(SEED)

    data_dir = config["data_dir"]
    scaling_info = load_scaling_info(data_dir)

    for model_name in ["xgboost", "logistic_regression"]:
        print(f"\n{'='*60}")
        print(f"Online Inference: {model_name} | PW: {config['prediction_window_size']} | Dev: {config['develop_set']}")
        print(f"{'='*60}")

        pw_version = f"prediction_window_{config['prediction_window_size']}"
        version = f"{pw_version}/resolution_control/apply_prob_0.0"

        # Load trained model
        ckpt_path = f"{config['save_dir_checkpoints']}/{config['develop_set']}/{model_name}/{version}/model_best.joblib"
        print(f"Loading model from: {ckpt_path}")
        model = jl.load(ckpt_path)

        output_dir = f"{config['save_dir_predictions']}/{config['develop_set']}/{model_name}/{version}/online"
        os.makedirs(output_dir, exist_ok=True)

        # IMPORTANT: For online prediction, always use prediction_window_0 data
        data_version = "prediction_window_0"

        # =====================================================================
        # Development test set
        # =====================================================================
        print(f"\n--- Development Set ({config['develop_set']}) ---")
        dev_data_dir = f"{data_dir}/{config['develop_set']}/model/{data_version}"

        meta_table = (
            pd.read_parquet(f"{dev_data_dir}/master.parquet")
            .query("inclusion_yn == 1 & vital_yn == 1")
            .reset_index(drop=True)
        )

        _, test_meta = train_test_split(
            meta_table, test_size=0.1, random_state=SEED, stratify=meta_table["label"]
        )
        test_meta = test_meta.reset_index(drop=True)
        print(f"Test set size: {len(test_meta)}")

        results = run_online_inference_ml(
            model=model,
            meta_table=test_meta,
            data_dir=f"{dev_data_dir}/timeseries",
            scaling_info=scaling_info,
            time_resolution=config["time_resolution"],
            min_observations=config["min_observations"],
        )

        results_df = aggregate_online_results(results, meta_table=test_meta)
        results_df.to_parquet(f"{output_dir}/{config['develop_set']}_test_online.parquet")
        print(f"Saved online predictions: {len(results_df)} rows")

        metrics_df = compute_online_metrics(results_df, positive_class=1, thresholds=config.get("eval_thresholds"))
        metrics_df.to_csv(f"{output_dir}/{config['develop_set']}_test_online_metrics.csv", index=False)
        print("\nOnline Metrics (Development Test Set):")
        print(metrics_df.to_string(index=False))

        del results
        gc.collect()

        # =====================================================================
        # External validation set
        # =====================================================================
        ext_set = config["external_set"]
        print(f"\n--- External Validation Set ({ext_set}) ---")
        ext_data_dir = f"{data_dir}/{ext_set}/model/{data_version}"

        ext_meta = (
            pd.read_parquet(f"{ext_data_dir}/master.parquet")
            .query("inclusion_yn == 1 & vital_yn == 1")
            .reset_index(drop=True)
        )
        print(f"External set size: {len(ext_meta)}")

        results = run_online_inference_ml(
            model=model,
            meta_table=ext_meta,
            data_dir=f"{ext_data_dir}/timeseries",
            scaling_info=scaling_info,
            time_resolution=config["time_resolution"],
            min_observations=config["min_observations"],
        )

        results_df = aggregate_online_results(results, meta_table=ext_meta)
        results_df.to_parquet(f"{output_dir}/{ext_set}_external_online.parquet")
        print(f"Saved online predictions: {len(results_df)} rows")

        metrics_df = compute_online_metrics(results_df, positive_class=1, thresholds=config.get("eval_thresholds"))
        metrics_df.to_csv(f"{output_dir}/{ext_set}_external_online_metrics.csv", index=False)
        print("\nOnline Metrics (External Validation):")
        print(metrics_df.to_string(index=False))

        del results
        gc.collect()

        # =====================================================================
        # MIMIC-IV
        # =====================================================================
        print(f"\n--- MIMIC-IV External Validation ---")
        mimic_data_dir = f"{data_dir}/mimic-iv/model/{data_version}"

        try:
            mimic_meta = (
                pd.read_parquet(f"{mimic_data_dir}/master.parquet")
                .query("inclusion_yn == 1 & vital_yn == 1")
                .reset_index(drop=True)
            )
            print(f"MIMIC-IV set size: {len(mimic_meta)}")

            results = run_online_inference_ml(
                model=model,
                meta_table=mimic_meta,
                data_dir=f"{mimic_data_dir}/timeseries",
                scaling_info=scaling_info,
                time_resolution=config["time_resolution"],
                min_observations=config["min_observations"],
            )

            results_df = aggregate_online_results(results, meta_table=mimic_meta)
            results_df.to_parquet(f"{output_dir}/mimic-iv_external_online.parquet")
            print(f"Saved online predictions: {len(results_df)} rows")

            metrics_df = compute_online_metrics(results_df, positive_class=1, thresholds=config.get("eval_thresholds"))
            metrics_df.to_csv(f"{output_dir}/mimic-iv_external_online_metrics.csv", index=False)
            print("\nOnline Metrics (MIMIC-IV):")
            print(metrics_df.to_string(index=False))

        except FileNotFoundError:
            print("MIMIC-IV data not found, skipping...")

        del model
        gc.collect()

    print(f"\n{'='*60}")
    print("All online inference experiments completed!")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Online inference for ML baselines")
    parser.add_argument("--develop_set", type=str, choices=["ilsan", "cchlmc"], required=True)
    parser.add_argument("--prediction_window_size", type=int, choices=[0, 48, 72], required=True)
    parser.add_argument("--data_dir", type=str, default="../../../data/processed")
    parser.add_argument("--save_dir_predictions", type=str, default="../../../result/predictions")
    parser.add_argument("--save_dir_checkpoints", type=str, default="../../../result/checkpoints")
    parser.add_argument("--time_resolution", type=float, default=1.0)
    parser.add_argument("--min_observations", type=int, default=1)
    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()

    config = {
        "develop_set": args.develop_set,
        "external_set": "cchlmc" if args.develop_set == "ilsan" else "ilsan",
        "prediction_window_size": args.prediction_window_size,
        "data_dir": args.data_dir,
        "save_dir_predictions": args.save_dir_predictions,
        "save_dir_checkpoints": args.save_dir_checkpoints,
        "time_resolution": args.time_resolution,
        "min_observations": args.min_observations,
        "eval_thresholds": [6, 12, 24, 48, 72, 96, 120, 144, 168],
    }

    experiment_online_inference(config)
# %%
