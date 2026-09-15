# %%
"""
Train XGBoost and Logistic Regression baselines and run holdout inference.

Usage:
    cd code/train_ml/scripts
    python experiment_holdout.py --develop_set ilsan --prediction_window_size 0 --device 0
    python experiment_holdout.py --develop_set ilsan --prediction_window_size 48 --device 0
    python experiment_holdout.py --develop_set ilsan --prediction_window_size 72 --device 0
"""

import argparse
import gc
import numpy as np
import os
import pandas as pd
import sys
import joblib as jl

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_utils import (
    DEMO_COL, ALL_FEATURES,
    load_scaling_info, build_dataset,
)

SEED = 1004


def experiment_holdout(config: dict):
    """Train ML baselines and run holdout inference on dev test + external sets."""
    np.random.seed(SEED)

    data_dir = config["data_dir"]
    scaling_info = load_scaling_info(data_dir)
    pw_dir = f"prediction_window_{config['prediction_window_size']}"

    for model_name in ["xgboost", "logistic_regression"]:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | PW: {config['prediction_window_size']} | Dev: {config['develop_set']}")
        print(f"{'='*60}")

        version = f"{pw_dir}/resolution_control/apply_prob_0.0"
        save_dir = f"{config['save_dir_predictions']}/{config['develop_set']}/{model_name}/{version}"
        ckpt_dir = f"{config['save_dir_checkpoints']}/{config['develop_set']}/{model_name}/{version}"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # =====================================================================
        # Load development set
        # =====================================================================
        dev_data_dir = f"{data_dir}/{config['develop_set']}/model/{pw_dir}"
        meta_table = (
            pd.read_parquet(f"{dev_data_dir}/master.parquet")
            .assign(age_raw=lambda df: df["age"], bmi_raw=lambda df: df["bmi"])
            .query("inclusion_yn == 1 & vital_yn == 1 & error_yn == 0")
            .reset_index(drop=True)
        )

        # Same split as CNN pipeline: test_size=0.1 first, then val from remainder
        train_val_meta, test_meta = train_test_split(
            meta_table, test_size=0.1, random_state=SEED, stratify=meta_table["label"]
        )
        train_meta, val_meta = train_test_split(
            train_val_meta, test_size=0.1 / 0.9, random_state=SEED, stratify=train_val_meta["label"]
        )

        print(f"Train: {len(train_meta)} | Val: {len(val_meta)} | Test: {len(test_meta)}")
        print(f"Train positive rate: {train_meta['label'].mean():.3f}")

        # Build feature matrices
        print("Building training features...")
        X_train, y_train, _ = build_dataset(train_meta, f"{dev_data_dir}/timeseries", scaling_info)
        print("Building validation features...")
        X_val, y_val, _ = build_dataset(val_meta, f"{dev_data_dir}/timeseries", scaling_info)
        print("Building test features...")
        X_test, y_test, df_test = build_dataset(test_meta, f"{dev_data_dir}/timeseries", scaling_info)

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # =====================================================================
        # Train model
        # =====================================================================
        if model_name == "xgboost":
            scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            model = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                early_stopping_rounds=20,
                random_state=SEED,
                n_jobs=-1,
                tree_method="hist",
                device=f"cuda:{config['device']}" if config["device"] != "cpu" else "cpu",
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50,
            )
        else:  # logistic_regression
            model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

        # Save model
        jl.dump(model, f"{ckpt_dir}/model_best.joblib")
        print(f"Model saved to {ckpt_dir}/model_best.joblib")

        # =====================================================================
        # Inference: development test set
        # =====================================================================
        pred_proba = model.predict_proba(X_test)[:, 1]
        df_test["pred_proba"] = pred_proba
        df_test.to_parquet(f"{save_dir}/{config['develop_set']}_test.parquet")
        print(f"Dev test predictions saved ({len(df_test)} samples)")

        # =====================================================================
        # Inference: external validation (cchlmc or ilsan)
        # =====================================================================
        ext_set = config["external_set"]
        ext_data_dir = f"{data_dir}/{ext_set}/model/{pw_dir}"
        ext_meta = (
            pd.read_parquet(f"{ext_data_dir}/master.parquet")
            .assign(age_raw=lambda df: df["age"], bmi_raw=lambda df: df["bmi"])
            .query("inclusion_yn == 1 & vital_yn == 1 & error_yn == 0")
            .reset_index(drop=True)
        )
        print(f"\nExternal set ({ext_set}): {len(ext_meta)} samples")

        X_ext, y_ext, df_ext = build_dataset(ext_meta, f"{ext_data_dir}/timeseries", scaling_info)
        pred_proba = model.predict_proba(X_ext)[:, 1]
        df_ext["pred_proba"] = pred_proba
        df_ext.to_parquet(f"{save_dir}/{ext_set}_external.parquet")
        print(f"External predictions saved ({len(df_ext)} samples)")

        # =====================================================================
        # Inference: MIMIC-IV
        # =====================================================================
        mimic_data_dir = f"{data_dir}/mimic-iv/model/{pw_dir}"
        try:
            mimic_meta = (
                pd.read_parquet(f"{mimic_data_dir}/master.parquet")
                .assign(age_raw=lambda df: df["age"], bmi_raw=lambda df: df["bmi"])
                .query("inclusion_yn == 1 & vital_yn == 1 & error_yn == 0")
                .reset_index(drop=True)
            )
            print(f"\nMIMIC-IV: {len(mimic_meta)} samples")

            X_mimic, y_mimic, df_mimic = build_dataset(mimic_meta, f"{mimic_data_dir}/timeseries", scaling_info)
            pred_proba = model.predict_proba(X_mimic)[:, 1]
            df_mimic["pred_proba"] = pred_proba
            df_mimic.to_parquet(f"{save_dir}/mimic-iv_external.parquet")
            print(f"MIMIC-IV predictions saved ({len(df_mimic)} samples)")
        except FileNotFoundError:
            print("MIMIC-IV data not found, skipping...")

        del model
        gc.collect()

    print(f"\n{'='*60}")
    print("All holdout experiments completed!")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML baselines (XGBoost, LR) with holdout setting")
    parser.add_argument("--develop_set", type=str, choices=["ilsan", "cchlmc"], required=True)
    parser.add_argument("--prediction_window_size", type=int, choices=[0, 48, 72], required=True)
    parser.add_argument("--device", type=str, default="0", help="GPU device or 'cpu'")
    parser.add_argument("--data_dir", type=str, default="../../../data/processed")
    parser.add_argument("--save_dir_predictions", type=str, default="../../../result/predictions")
    parser.add_argument("--save_dir_checkpoints", type=str, default="../../../result/checkpoints")
    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()

    config = {
        "develop_set": args.develop_set,
        "external_set": "cchlmc" if args.develop_set == "ilsan" else "ilsan",
        "prediction_window_size": args.prediction_window_size,
        "device": args.device if args.device.lower() != "cpu" else "cpu",
        "data_dir": args.data_dir,
        "save_dir_predictions": args.save_dir_predictions,
        "save_dir_checkpoints": args.save_dir_checkpoints,
    }

    experiment_holdout(config)
# %%
