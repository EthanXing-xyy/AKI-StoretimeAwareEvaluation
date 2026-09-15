# %%
"""
Experiment Script for Online Inference with 1D CNN Model

This script runs online inference to simulate real-time prediction scenarios
where predictions are generated at each timestamp as new data arrives.

Usage:
    python experiment_online_inference.py --develop_set ilsan --prediction_window_size 48
"""

import argparse
import gc
import numpy as np
import os
import pandas as pd
import torch
import sys

from pathlib import Path
from IPython.display import display

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.callbacks import get_progress_bar
from shared.utils import set_seed
from data.datamodule import AKIDataModule, TestDataModule
from lightning_modules.classifier_module import AKI_Simple_TrainModule
from models.mask_rms_1d_cnn import Timeseries_CNN_Model
from scripts.online_inference import (
    run_online_inference,
    run_online_inference_batched_patients,
    run_online_inference_dataloader,
    aggregate_online_results,
    compute_online_metrics,
)

torch.set_float32_matmul_precision("high")

# Feature configuration (defined at module level for use in experiment_online_inference)
DEMO_COL = ["age", "sex", "bmi"]
VITAL_COL = ['sbp', 'dbp', 'pulse', 'resp', 'spo2', 'temp']
LAB_COL = [
    'albumin', 'alp', 'alt', 'aptt', 'ast', 'bilirubin',
    'bnp', 'bun', 'calcium', 'chloride', 'cholesterol',
    'ck', 'creatinine', 'crp', 'esr', 'ggt', 'glucose',
    'hba1c', 'hco3', 'hdl', 'hematocrit', 'hemoglobin',
    'lactate', 'ldh', 'magnesium', 'pco2', 'pdw', 'ph',
    'phosphorus', 'platelet', 'po2', 'potassium',
    'procalcitonin', 'protein', 'pt_inr', 'rbc',
    'rdw', 'sodium', 'tco2', 'triglyceride',
    'uric_acid', 'wbc'
]
ALL_FEATURES = VITAL_COL + LAB_COL


def experiment_online_inference(config: dict):
    """
    Run online inference experiment for 1D CNN model.
    
    This function:
    1. Loads trained model checkpoint
    2. Runs online inference on test set (generates predictions at each timestamp)
    3. Computes metrics at different time horizons
    4. Saves results
    """
    set_seed(1004)
    
    # Create output directory
    output_dir = f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}/online"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Online Inference Experiment (1D CNN)")
    print(f"{'='*60}")
    print(f"Develop Set: {config['develop_set']}")
    print(f"Experiment: {config['exp_nm']}")
    print(f"Model Version: {config['version']}")
    print(f"Data Version: prediction_window_0 (always use full time series)")
    print(f"Time Resolution: {config['time_resolution']} hours")
    print(f"Sequence Length: {config['seq_len']}")
    print(f"Output Dir: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load scaling info
    scaling_info_path = f"{config['data_dir']}/scaling_info/ilsan/scaling_info.parquet"
    try:
        scaling_info = pd.read_parquet(scaling_info_path).query("feature in @ALL_FEATURES").set_index("feature").loc[ALL_FEATURES].reset_index()
        print(f"Loaded scaling info from: {scaling_info_path}")
    except Exception as e:
        print(f"Warning: Could not load scaling info: {e}")
        scaling_info = pd.DataFrame()
    
    # Initialize model architecture
    backbone_model = Timeseries_CNN_Model(
        num_demo_features=config['num_demo_features'],
        num_timeseries_features=config['num_timeseries_features'],
        num_classes=config['num_classes'],
        seq_len=config['seq_len'],
    )
    
    # Load trained model checkpoint
    checkpoint_path = f"../../../result/checkpoints/{config['develop_set']}/{config['exp_nm']}/{config['version']}/model_best.ckpt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    lightning_module = AKI_Simple_TrainModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone_model=backbone_model,
        num_class=config['num_classes'],
        ordinal_class=False,
    )
    
    # Determine device
    if config['device'] == 'cpu':
        device = 'cpu'
    elif config['device'] == 'mps':
        device = 'mps'
    else:
        device = 'cuda'
    
    # =========================================================================
    # Run online inference on development test set
    # =========================================================================
    print(f"\n--- Development Set ({config['develop_set']}) ---")
    
    # IMPORTANT: For online prediction, always use prediction_window_0 data
    # This gives us the full time series regardless of which model we're testing
    # The model may have been trained on 0, 48, or 72 hour horizon, but for
    # online prediction we need the complete patient trajectory
    data_version = "prediction_window_0"
    data_dir = f"{config['data_dir']}/{config['develop_set']}/model/{data_version}"
    print(f"Using data version: {data_version} (full time series for online prediction)")
    
    meta_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .query("inclusion_yn == 1 & vital_yn == 1")
        .reset_index(drop=True)
    )
    
    # Split to get test set (same as training split)
    from sklearn.model_selection import train_test_split
    train_val_meta, test_meta = train_test_split(
        meta_table,
        test_size=0.1,
        random_state=1004,
        stratify=meta_table['label']
    )
    test_meta = test_meta.reset_index(drop=True)
    
    print(f"Test set size: {len(test_meta)}")
    print(f"Positive rate: {test_meta['label'].mean():.3f}")
    
    # Run online inference (optimized with batching)
    if config.get('use_dataloader', True):
        # RECOMMENDED: DataLoader-based inference with parallel data loading
        results = run_online_inference_dataloader(
            lightning_module=lightning_module,
            meta_table=test_meta,
            data_dir=f"{data_dir}/timeseries",
            scaling_info=scaling_info,
            device=device,
            seq_len=config['seq_len'],
            time_column_idx=config['time_column_idx'],
            time_resolution=config['time_resolution'],
            min_observations=config['min_observations'],
            show_progress=True,
            batch_size=config.get('dataloader_batch_size', 16),
            gpu_batch_size=config.get('gpu_batch_size', 32),
            num_workers=config.get('num_workers', 4),
            prefetch_factor=config.get('prefetch_factor', 2),
        )
    else:
        # Alternative: Manual batching
        inference_fn = run_online_inference_batched_patients if config.get('use_batched_patients', True) else run_online_inference
        results = inference_fn(
            lightning_module=lightning_module,
            meta_table=test_meta,
            data_dir=f"{data_dir}/timeseries",
            scaling_info=scaling_info,
            device=device,
            seq_len=config['seq_len'],
            time_column_idx=config['time_column_idx'],
            time_resolution=config['time_resolution'],
            min_observations=config['min_observations'],
            show_progress=True,
            batch_size=config.get('batch_size', 256),
            patient_batch_size=config.get('patient_batch_size', 64),
        )
    
    # Aggregate and save results (with time difference calculations)
    results_df = aggregate_online_results(results, meta_table=test_meta)
    results_df.to_parquet(f"{output_dir}/{config['develop_set']}_test_online.parquet")
    print(f"Saved online predictions: {len(results_df)} rows")
    print(f"Columns: {results_df.columns.tolist()}")
    
    # Compute and display metrics at different time horizons
    metrics_df = compute_online_metrics(
        results_df,
        positive_class=1,
        thresholds=config.get('eval_thresholds'),
    )
    metrics_df.to_csv(f"{output_dir}/{config['develop_set']}_test_online_metrics.csv", index=False)
    print("\nOnline Metrics (Development Test Set):")
    print(metrics_df.to_string(index=False))
    
    del results
    gc.collect()
    
    # =========================================================================
    # Run online inference on external validation set
    # =========================================================================
    print(f"\n--- External Validation Set ({config['external_set']}) ---")
    
    # Use prediction_window_0 for online prediction
    data_dir = f"{config['data_dir']}/{config['external_set']}/model/{data_version}"
    
    meta_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .query("inclusion_yn == 1 & vital_yn == 1")
        .reset_index(drop=True)
    )
    
    print(f"External set size: {len(meta_table)}")
    print(f"Positive rate: {meta_table['label'].mean():.3f}")
    
    if config.get('use_dataloader', True):
        results = run_online_inference_dataloader(
            lightning_module=lightning_module,
            meta_table=meta_table,
            data_dir=f"{data_dir}/timeseries",
            scaling_info=scaling_info,
            device=device,
            seq_len=config['seq_len'],
            time_column_idx=config['time_column_idx'],
            time_resolution=config['time_resolution'],
            min_observations=config['min_observations'],
            show_progress=True,
            batch_size=config.get('dataloader_batch_size', 16),
            gpu_batch_size=config.get('gpu_batch_size', 32),
            num_workers=config.get('num_workers', 4),
            prefetch_factor=config.get('prefetch_factor', 2),
        )
    else:
        results = inference_fn(
            lightning_module=lightning_module,
            meta_table=meta_table,
            data_dir=f"{data_dir}/timeseries",
            scaling_info=scaling_info,
            device=device,
            seq_len=config['seq_len'],
            time_column_idx=config['time_column_idx'],
            time_resolution=config['time_resolution'],
            min_observations=config['min_observations'],
            show_progress=True,
            batch_size=config.get('batch_size', 256),
            patient_batch_size=config.get('patient_batch_size', 64),
        )
    
    # Aggregate with time difference calculations
    results_df = aggregate_online_results(results, meta_table=meta_table)
    results_df.to_parquet(f"{output_dir}/{config['external_set']}_external_online.parquet")
    print(f"Saved online predictions: {len(results_df)} rows")
    
    metrics_df = compute_online_metrics(
        results_df,
        positive_class=1,
        thresholds=config.get('eval_thresholds'),
    )
    metrics_df.to_csv(f"{output_dir}/{config['external_set']}_external_online_metrics.csv", index=False)
    print("\nOnline Metrics (External Validation Set):")
    print(metrics_df.to_string(index=False))
    
    del results
    gc.collect()
    
    # =========================================================================
    # Run online inference on MIMIC-IV
    # =========================================================================
    print(f"\n--- MIMIC-IV External Validation ---")
    
    # Use prediction_window_0 for online prediction
    data_dir = f"{config['data_dir']}/mimic-iv/model/{data_version}"
    
    try:
        meta_table = (
            pd.read_parquet(f"{data_dir}/master.parquet")
            .query("inclusion_yn == 1 & vital_yn == 1")
            .reset_index(drop=True)
        )
        
        print(f"MIMIC-IV set size: {len(meta_table)}")
        print(f"Positive rate: {meta_table['label'].mean():.3f}")
        
        if config.get('use_dataloader', True):
            results = run_online_inference_dataloader(
                lightning_module=lightning_module,
                meta_table=meta_table,
                data_dir=f"{data_dir}/timeseries",
                scaling_info=scaling_info,
                device=device,
                seq_len=config['seq_len'],
                time_column_idx=config['time_column_idx'],
                time_resolution=config['time_resolution'],
                min_observations=config['min_observations'],
                show_progress=True,
                batch_size=config.get('dataloader_batch_size', 16),
                gpu_batch_size=config.get('gpu_batch_size', 32),
                num_workers=config.get('num_workers', 4),
                prefetch_factor=config.get('prefetch_factor', 2),
            )
        else:
            results = inference_fn(
                lightning_module=lightning_module,
                meta_table=meta_table,
                data_dir=f"{data_dir}/timeseries",
                scaling_info=scaling_info,
                device=device,
                seq_len=config['seq_len'],
                time_column_idx=config['time_column_idx'],
                time_resolution=config['time_resolution'],
                min_observations=config['min_observations'],
                show_progress=True,
                batch_size=config.get('batch_size', 256),
                patient_batch_size=config.get('patient_batch_size', 64),
            )
        
        # Aggregate with time difference calculations
        results_df = aggregate_online_results(results, meta_table=meta_table)
        results_df.to_parquet(f"{output_dir}/mimic-iv_external_online.parquet")
        print(f"Saved online predictions: {len(results_df)} rows")
        
        metrics_df = compute_online_metrics(
            results_df,
            positive_class=1,
            thresholds=config.get('eval_thresholds'),
        )
        metrics_df.to_csv(f"{output_dir}/mimic-iv_external_online_metrics.csv", index=False)
        print("\nOnline Metrics (MIMIC-IV):")
        print(metrics_df.to_string(index=False))
        
    except FileNotFoundError:
        print("MIMIC-IV data not found, skipping...")
    
    del lightning_module
    gc.collect()
    
    print(f"\n{'='*60}")
    print("Online inference completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run online inference for 1D CNN model")
    parser.add_argument(
        "--develop_set",
        type=str,
        choices=["ilsan", "cchlmc"],
        required=True,
        help="Development set hospital"
    )
    parser.add_argument(
        "--prediction_window_size",
        type=int,
        choices=[0, 48, 72],
        required=True,
        help="Prediction window size (hours)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device(s) to use, e.g. '0', 'mps', or 'cpu'"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../data/processed",
        help="Data directory"
    )
    parser.add_argument(
        "--save_dir_predictions",
        type=str,
        default="../../../result/predictions",
        help="Prediction save directory"
    )
    parser.add_argument(
        "--exp_nm",
        type=str,
        default="mask_rms_cnn",
        help="Experiment name"
    )
    parser.add_argument(
        "--time_resolution",
        type=float,
        default=1.0,
        help="Time resolution for predictions in hours (default: 1.0 = hourly)"
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=1,
        help="Minimum observations required to make prediction"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Expected sequence length for 1D CNN model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for GPU inference (timestamps per batch)"
    )
    parser.add_argument(
        "--patient_batch_size",
        type=int,
        default=64,
        help="Number of patients to process together (for cross-patient batching)"
    )
    parser.add_argument(
        "--no_batched_patients",
        action="store_true",
        help="Disable cross-patient batching (process each patient separately)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for parallel data loading"
    )
    parser.add_argument(
        "--dataloader_batch_size",
        type=int,
        default=16,
        help="Number of patients per DataLoader batch"
    )
    parser.add_argument(
        "--gpu_batch_size",
        type=int,
        default=32,
        help="Max samples per GPU forward pass (to avoid OOM)"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker"
    )
    parser.add_argument(
        "--no_dataloader",
        action="store_true",
        help="Disable DataLoader (use manual batching instead)"
    )
    
    args = parser.parse_args()
    
    # Handle device argument
    if args.device.lower() == "cpu":
        args.device = "cpu"
    elif args.device.lower() == "mps":
        args.device = "mps"
    else:
        if isinstance(args.device, str):
            device_list = [d.strip() for d in args.device.split(",") if d.strip() != ""]
            if len(device_list) == 1:
                try:
                    args.device = [int(device_list[0])]
                except ValueError:
                    args.device = device_list[0]
            else:
                args.device = [int(d) for d in device_list]
    
    return args


# %%
if __name__ == "__main__":
    
    args = parse_args()
    
    config = {
        "exp_nm": args.exp_nm,
        "version": f"prediction_window_{args.prediction_window_size}/resolution_control/apply_prob_0.0",
        "device": args.device,
        "data_dir": args.data_dir,
        "save_dir_predictions": args.save_dir_predictions,
        "develop_set": args.develop_set,
        "external_set": "cchlmc" if args.develop_set == "ilsan" else "ilsan",
        "time_resolution": args.time_resolution,
        "min_observations": args.min_observations,
        "seq_len": args.seq_len,
        # Batching config (for optimized inference)
        "batch_size": args.batch_size,
        "patient_batch_size": args.patient_batch_size,
        "use_batched_patients": not args.no_batched_patients,
        # DataLoader config (recommended)
        "use_dataloader": not args.no_dataloader,
        "num_workers": args.num_workers,
        "dataloader_batch_size": args.dataloader_batch_size,
        "gpu_batch_size": args.gpu_batch_size,
        "prefetch_factor": args.prefetch_factor,
        # Model config (same as training)
        "num_demo_features": len(DEMO_COL),
        "num_timeseries_features": len(ALL_FEATURES),
        "num_classes": 2,
        # Time column index (-1 means use row index)
        "time_column_idx": -1,
        # Evaluation thresholds (in row indices if time_column_idx=-1, else hours)
        "eval_thresholds": [6, 12, 24, 48, 72, 96, 120, 144, 168],
    }
    
    experiment_online_inference(config)
# %%
