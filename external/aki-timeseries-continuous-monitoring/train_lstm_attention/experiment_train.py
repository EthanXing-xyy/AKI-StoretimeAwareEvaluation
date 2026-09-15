# %%
"""
Training Experiment Script for LSTM with Attention Model

This script trains an LSTM with Attention model for AKI prediction
using the same data format as the Transformer model.

Usage:
    python experiment_train.py --develop_set ilsan --prediction_window_size 48
"""

import argparse
import gc
import numpy as np
import os
import pandas as pd
import torch
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.callbacks import get_early_stopping_callback, get_check_point_callback, get_progress_bar, get_tensor_board_logger
from shared.utils import set_seed
from src.data.datamodule import AKIDataModule, TestDataModule
from src.lightning_modules.classifier_module import AKI_Simple_TrainModule
from src.models.lstm_attention import LSTMAttentionModel
from src.scripts.train import train_model

torch.set_float32_matmul_precision("high")

# %%
def experiment_holdout(
    config:dict,
):
    """
    Run training experiment with holdout validation.
    
    Args:
        config: Configuration dictionary with model and training parameters
    """
    set_seed(1004)
    
    os.makedirs(f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}", exist_ok=True)
    
    tb_logger = get_tensor_board_logger(
        save_path=f"../../result/logs/{config['develop_set']}",
        log_name=f"{config['exp_nm']}/{config['version']}",
    )
    
    check_point_callback = get_check_point_callback(
        save_path=f"../../result/checkpoints/{config['develop_set']}/{config['exp_nm']}/{config['version']}",
        cv_num=None,
    )
    
    early_stopping_callback = get_early_stopping_callback(
        monitor="val/loss",
        patience=10,
        mode="min",
        verbose=False,
    )
    
    prog_bar = get_progress_bar()
    
    # Initialize LSTM Attention model
    lightning_module = AKI_Simple_TrainModule(
        backbone_model=LSTMAttentionModel(
            config=config
        ),
        num_classes=config.get("num_classes"),
        ordinal_class=False,
        learning_rate=config.get("learning_rate", 0.001),
    )
    
    _version = "/".join(config['version'].split('/')[0:1])
    data_dir = f"{config['data_dir']}/{config['develop_set']}/transformer/{_version}"
    
    master_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .query(
            "inclusion_yn == 1 & vital_yn == 1"
        )
        .reset_index(drop=True)
    )
    
    # Load scaling info from ilsan dataset
    scaling_info_path = f"{config['data_dir']}/scaling_info/ilsan/scaling_info.parquet"
    try:
        scaling_info = pd.read_parquet(scaling_info_path)
        print(f"Loaded scaling info from: {scaling_info_path}")
        print(f"Scaling info shape: {scaling_info.shape}")
    except Exception as e:
        print(f"Warning: Could not load scaling info from {scaling_info_path}: {e}")
        scaling_info = None
    
    datamodule = AKIDataModule(
        master_table=master_table,
        data_dir=f"{data_dir}/timeseries",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        val_split=0.1,
        test_split=0.1,
        random_state=1004,
        scaling_info=scaling_info,
    )
    
    print(f"\n{'='*60}")
    print(f"Training LSTM with Attention Model")
    print(f"{'='*60}")
    print(f"Develop Set: {config['develop_set']}")
    print(f"Experiment: {config['exp_nm']}")
    print(f"Version: {config['version']}")
    print(f"Model Config:")
    print(f"  - d_model: {config['d_model']}")
    print(f"  - num_lstm_layers: {config.get('num_lstm_layers', 2)}")
    print(f"  - num_heads: {config['num_heads']}")
    print(f"  - dropout: {config['dropout']}")
    print(f"  - bidirectional: {config.get('bidirectional', True)}")
    print(f"{'='*60}\n")
    
    train_model(
        lightning_module,
        datamodule, 
        config,
        tb_logger,
        prog_bar,
        check_point_callback,
        early_stopping_callback,
    )
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best model saved to: {check_point_callback.best_model_path}")
    print(f"{'='*60}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM with Attention model")
    parser.add_argument("--develop_set", type=str, choices=["ilsan", "cchlmc"], required=True, help="Development set hospital")
    parser.add_argument("--prediction_window_size", type=int, choices=[0, 48, 72], required=True, help="Prediction window size")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device(s) to use, e.g. '0', '0,1', 'mps', or 'cpu'"
    )
    parser.add_argument("--max_epoch", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=2**7, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--data_dir", type=str, default="../../data/processed", help="Data directory")
    parser.add_argument("--save_dir_predictions", type=str, default="../../result/predictions", help="Prediction save directory")
    parser.add_argument("--exp_nm", type=str, default="lstm_attention", help="Experiment name")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension (reduced for memory)")
    parser.add_argument("--num_lstm_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length (truncate longer)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    args = parser.parse_args()

    # Handle device argument: convert comma-separated string to list of ints or 'cpu'
    if args.device.lower() == "cpu":
        args.device = "cpu"
    elif args.device.lower() == "mps":
        args.device = "mps"
    else:
        # Accept both comma-separated string and single int as string
        if isinstance(args.device, str):
            device_list = [d.strip() for d in args.device.split(",") if d.strip() != ""]
            
            if len(device_list) == 1:
                try:
                    args.device = [int(device_list[0])]
                except ValueError:
                    args.device = [device_list[0]]
            else:
                args.device = [int(d) for d in device_list]
                
    return args

# %%
if __name__ == "__main__":
    
    demo_col = ["age", "sex", "bmi"]

    vital_col = [
        'sbp', 'dbp', 'pulse', 'resp', 'spo2', 'temp'
    ]
    lab_col = [
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

    feature_encoding = {
        **{col: i for i, col in enumerate(demo_col)},
        **{col: i + len(demo_col) for i, col in enumerate(vital_col)},
        **{col: i + len(demo_col) + len(vital_col) for i, col in enumerate(lab_col)}
    }

    args = parse_args()

    config = {
        "exp_nm": args.exp_nm,
        "version": f"prediction_window_{args.prediction_window_size}",
        "max_epoch": args.max_epoch,
        "device": args.device,
        "batch_size": args.batch_size,
        "data_dir": args.data_dir,
        "save_dir_predictions": args.save_dir_predictions,
        "num_workers": args.num_workers,
        "develop_set": args.develop_set,
        "external_set": "cchlmc" if args.develop_set == "ilsan" else "ilsan",
        # Model configuration
        "d_model": args.d_model,
        "num_layers": args.num_lstm_layers,
        "num_lstm_layers": args.num_lstm_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "bidirectional": True,
        "type_dict": feature_encoding,
        "num_classes": 2,
        "learning_rate": args.learning_rate,
        # Memory optimization
        "max_seq_len": args.max_seq_len,
        "gradient_checkpointing": args.gradient_checkpointing,
        "ffn_expansion": 2,  # Reduced from 4
    }

    experiment_holdout(config)
# %%

