# %%
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

from shared.callbacks import get_check_point_callback, get_progress_bar, get_tensor_board_logger
from shared.utils import set_seed
from src.data.datamodule import AKIDataModule, TestDataModule
from src.lightning_modules.classifier_module import AKI_Simple_TrainModule
from src.models.ite_transformer import ITETransformer
from src.scripts.inference import inference_model

torch.set_float32_matmul_precision("high")

# %%
def experiment_holdout(
    config:dict,
):
    
    set_seed(1004)
    
    os.makedirs(f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}", exist_ok=True)
    
    prog_bar = get_progress_bar()
    
    lightning_module = AKI_Simple_TrainModule(
        backbone_model=ITETransformer(
            config=config
        ),
        num_classes=config.get("num_classes"),
        ordinal_class=False,
    )

    # Load scaling info from ilsan dataset (used for all datasets)
    scaling_info_path = f"{config['data_dir']}/scaling_info/ilsan/scaling_info.parquet"
    try:
        scaling_info = pd.read_parquet(scaling_info_path)
        print(f"Loaded scaling info from: {scaling_info_path}")
        print(f"Scaling info shape: {scaling_info.shape}")
    except Exception as e:
        print(f"Warning: Could not load scaling info from {scaling_info_path}: {e}")
        scaling_info = None
    
    _version = "/".join(config['version'].split('/')[0:1])
    data_dir = f"{config['data_dir']}/{config['develop_set']}/transformer/{_version}"
    
    master_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .query(
            "inclusion_yn == 1 & vital_yn == 1"
        )
        .reset_index(drop=True)
    )
    
    
    
    lightning_module = AKI_Simple_TrainModule.load_from_checkpoint(
        checkpoint_path=f"../../result/checkpoints/{config['develop_set']}/{config['exp_nm']}/{config['version']}/model_best.ckpt",
        backbone_model=ITETransformer(
            config=config
        ),
        num_classes=config.get("num_classes"),
        ordinal_class=False,
    )
    
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
    
    datamodule.setup(stage="predict")
    df_test = datamodule.predict_dataset.master_table
    
    test_pred_proba = inference_model(
        lightning_module=lightning_module,
        datamodule=datamodule,
        prog_bar=prog_bar,
        config=config,
    )
    
    # df_test = datamodule.predict_dataset.meta_table
    df_test["pred_proba"] = test_pred_proba.detach().cpu().numpy()[:, 1]
    df_test.to_parquet(f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}/{config['develop_set']}_test.parquet")
    
    datamodule.teardown()
    del datamodule
    gc.collect()
    
    ## cchlmc
    data_dir = f"{config['data_dir']}/{config['external_set']}/transformer/{"/".join(config['version'].split('/')[0:1])}"
    
    master_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .query(
            "inclusion_yn == 1 & vital_yn == 1"
        )
        .reset_index(drop=True)
    )
    
    datamodule = TestDataModule(
        master_table=master_table,
        data_dir=f"{data_dir}/timeseries",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        scaling_info=scaling_info,
    )
    
    datamodule.setup(stage="predict")
    df_external = datamodule.dataset.master_table
    
    test_pred_proba = inference_model(
        lightning_module=lightning_module,
        datamodule=datamodule,
        prog_bar=prog_bar,
        config=config,
    )
    
    df_external["pred_proba"] = test_pred_proba.detach().cpu().numpy()[:, 1]
    df_external.to_parquet(f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}/{config['external_set']}_external.parquet")
    
    datamodule.teardown()
    del datamodule
    gc.collect()
    
    ## mimic-iv
    data_dir = f"{config['data_dir']}/mimic-iv/transformer/{"/".join(config['version'].split('/')[0:1])}"
    
    master_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .query(
            "inclusion_yn == 1 & vital_yn == 1"
        )
        .reset_index(drop=True)
    )
    
    datamodule = TestDataModule(
        master_table=master_table,
        data_dir=f"{data_dir}/timeseries",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        scaling_info=scaling_info,
    )
    
    datamodule.setup(stage="predict")
    df_external = datamodule.dataset.master_table
    
    test_pred_proba = inference_model(
        lightning_module=lightning_module,
        datamodule=datamodule,
        prog_bar=prog_bar,
        config=config,
    )
    
    df_external["pred_proba"] = test_pred_proba.detach().cpu().numpy()[:, 1]
    df_external.to_parquet(f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}/mimic-iv_external.parquet")
    
    datamodule.teardown()
    del datamodule
    del lightning_module
    gc.collect()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train time series CNN model with holdout setting")
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
    parser.add_argument("--exp_nm", type=str, default="ite_transformer", help="Experiment name")
    args = parser.parse_args()

    # Handle device argument: accept single GPU as int (e.g., 0), or comma-separated list, or 'cpu'/'mps'
    if args.device.lower() == "cpu":
        args.device = "cpu"
    elif args.device.lower() == "mps":
        args.device = "mps"
    else:
        # Accept both single int (e.g., "0") and comma-separated string (e.g., "0,1")
        if isinstance(args.device, str):
            device_list = [d.strip() for d in args.device.split(",") if d.strip() != ""]
            if len(device_list) == 1:
                try:
                    # Accept single GPU as int, e.g., device 0
                    args.device = [int(device_list[0])]
                except ValueError:
                    args.device = device_list[0]
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
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "type_dict": feature_encoding,
        "num_classes": 2,
    }

    experiment_holdout(config)
# %%
