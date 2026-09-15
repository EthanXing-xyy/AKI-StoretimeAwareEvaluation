# %%
import argparse
import gc
import numpy as np
import os
import pandas as pd
import torch
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.callbacks import get_check_point_callback, get_progress_bar, get_tensor_board_logger
from shared.utils import set_seed
from data.datamodule import AKIDataModule, TestDataModule
from lightning_modules.classifier_module import AKI_Simple_TrainModule
from models.mask_rms_1d_cnn import Timeseries_CNN_Model
from inference import inference_model

torch.set_float32_matmul_precision("high")

# %%
def experiment_holdout(
    config:dict,
):
    
    set_seed(1004)
    
    os.makedirs(f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}", exist_ok=True)
    
    prog_bar = get_progress_bar()
    
    lightning_module = AKI_Simple_TrainModule(
        backbone_model=Timeseries_CNN_Model(
            num_demo_features=3,
            num_timeseries_features=48,
            num_classes=2,
            seq_len=config.get("input_seq_len"),
        ),
        num_class=2,
        ordinal_class=False,
    )
    
    data_dir = f"{config['data_dir']}/{config['develop_set']}/model/{"/".join(config['version'].split('/')[0:1])}"
    
    meta_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .assign(
            age_raw = lambda df: df["age"],
            bmi_raw = lambda df: df["bmi"],
        )
        .query(
            "inclusion_yn == 1 & vital_yn == 1 & error_yn == 0"
        )
        .reset_index(drop=True)
    )
    
    lightning_module = AKI_Simple_TrainModule.load_from_checkpoint(
        checkpoint_path=f"../../../result/checkpoints/{config['develop_set']}/{config['exp_nm']}/{config['version']}/model_best.ckpt",
        backbone_model=Timeseries_CNN_Model(
            num_demo_features=3,
            num_timeseries_features=48,
            num_classes=2,
            seq_len=config.get("input_seq_len"),
        ),
        num_class=2,
        ordinal_class=False,
    )
    
    datamodule = AKIDataModule(
        meta_table=meta_table,
        data_dir=f"{data_dir}/timeseries",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        val_split=0.1,
        test_split=0.1,
        random_state=1004,
        resolution_control_method=config.get("resolution_control_method"),
        resolution=config.get("resolution"),
        resolution_control_features=config.get("resolution_control_features"),
        scaling_info=config.get("scaling_info"),
        apply_prob=0.0,
        all_features=config.get("all_features"),
        seq_len=config.get("input_seq_len"),
    )
    
    datamodule.setup(stage="predict")
    df_test = datamodule.predict_dataset.meta_table
    
    test_pred_proba = inference_model(
        lightning_module=lightning_module,
        datamodule=datamodule,
        prog_bar=prog_bar,
        devices=config.get("device"),
    )
    
    # df_test = datamodule.predict_dataset.meta_table
    df_test["pred_proba"] = test_pred_proba.detach().cpu().numpy()[:, 1]
    df_test.to_parquet(f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}/{config['develop_set']}_test.parquet")
    
    datamodule.teardown()
    del datamodule
    gc.collect()
    
    ## cchlmc
    data_dir = f"{config['data_dir']}/{config['external_set']}/model/{"/".join(config['version'].split('/')[0:1])}"
    
    meta_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .assign(
            age_raw = lambda df: df["age"],
            bmi_raw = lambda df: df["bmi"],
        )
        .query(
            "inclusion_yn == 1 & vital_yn == 1 & error_yn == 0"
        )
        .reset_index(drop=True)
    )
    
    datamodule = TestDataModule(
        meta_table=meta_table,
        data_dir=f"{data_dir}/timeseries",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        resolution_control_method=config.get("resolution_control_method"),
        resolution=config.get("resolution"),
        resolution_control_features=config.get("resolution_control_features"),
        scaling_info=config.get("scaling_info"),
        apply_prob=0.0,
        all_features=config.get("all_features"),
        seq_len=config.get("input_seq_len"),
    )
    
    datamodule.setup(stage="predict")
    df_external = datamodule.dataset.meta_table
    
    test_pred_proba = inference_model(
        lightning_module=lightning_module,
        datamodule=datamodule,
        prog_bar=prog_bar,
        devices=config.get("device"),
    )
    
    df_external["pred_proba"] = test_pred_proba.detach().cpu().numpy()[:, 1]
    df_external.to_parquet(f"{config['save_dir_predictions']}/{config['develop_set']}/{config['exp_nm']}/{config['version']}/{config['external_set']}_external.parquet")
    
    datamodule.teardown()
    del datamodule
    gc.collect()
    
    ## mimic-iv
    data_dir = f"{config['data_dir']}/mimic-iv/model/{"/".join(config['version'].split('/')[0:1])}"
    
    meta_table = (
        pd.read_parquet(f"{data_dir}/master.parquet")
        .assign(
            age_raw = lambda df: df["age"],
            bmi_raw = lambda df: df["bmi"],
        )
        .query(
            "inclusion_yn == 1 & vital_yn == 1 & error_yn == 0"
        )
        .reset_index(drop=True)
    )
    
    datamodule = TestDataModule(
        meta_table=meta_table,
        data_dir=f"{data_dir}/timeseries",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        resolution_control_method=config.get("resolution_control_method"),
        resolution=config.get("resolution"),
        resolution_control_features=config.get("resolution_control_features"),
        scaling_info=config.get("scaling_info"),
        apply_prob=0.0,
        all_features=config.get("all_features"),
        seq_len=config.get("input_seq_len"),
    )
    
    datamodule.setup(stage="predict")
    df_external = datamodule.dataset.meta_table
    
    test_pred_proba = inference_model(
        lightning_module=lightning_module,
        datamodule=datamodule,
        prog_bar=prog_bar,
        devices=config.get("device"),
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
    parser.add_argument("--input_seq_len", type=int, choices=[48, 256], required=True, help="Input sequence length")
    parser.add_argument("--apply_prob", type=float, choices=[0.0, 0.25, 0.5, 0.75, 1.0], required=True, help="Resolution control apply probability")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device(s) to use, e.g. '0', '0,1', or 'mps'"
    )
    parser.add_argument("--max_epoch", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=2**12, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--data_dir", type=str, default="../../../data/processed", help="Data directory")
    parser.add_argument("--save_dir_predictions", type=str, default="../../../result/predictions", help="Prediction save directory")
    parser.add_argument("--exp_nm", type=str, default="time_series_cnn", help="Experiment name")
    args = parser.parse_args()

    # Handle device argument: convert comma-separated string to list of ints or 'cpu'
    if args.device.lower() == "mps":
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
                
            # print(args.device)
        # If argparse gives a list (from nargs), keep as is
    return args

# %%
if __name__ == "__main__":
    
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
    
    feat_col = vital_col + lab_col
    demo_col = ['age', 'bmi']  # demographic features for normalization
    all_scale_features = feat_col + demo_col
    
    scale_info = (
        pd.read_parquet(
            "../../../data/processed/scaling_info/ilsan/scaling_info.parquet"
        )
        .query(
            "feature in @all_scale_features"
        )
    )
    # Reorder time-series features to match feat_col order, keep demographics at end
    ts_scale = scale_info.query("feature in @feat_col").set_index("feature").loc[feat_col].reset_index()
    demo_scale = scale_info.query("feature in @demo_col")
    scale_info = pd.concat([ts_scale, demo_scale], ignore_index=True)

    args = parse_args()

    config = {
        "exp_nm": args.exp_nm,
        "version": f"prediction_window_{args.prediction_window_size}/resolution_control/apply_prob_{args.apply_prob}",
        "max_epoch": args.max_epoch,
        "device": args.device,
        "batch_size": args.batch_size,
        "data_dir": args.data_dir,
        "save_dir_predictions": args.save_dir_predictions,
        "num_workers": args.num_workers,
        "develop_set": args.develop_set,
        "external_set": "cchlmc" if args.develop_set == "ilsan" else "ilsan",
        "resolution_control_method": "round",
        "resolution": {'sbp': 10, 'dbp': 10, 'bnp': 1, 'magnesium': 0.1, "crp": 0.1, "creatinine": 0.1},
        "resolution_control_features": None,
        "scaling_info": scale_info,
        "apply_prob": args.apply_prob,
        "all_features": feat_col,
        "input_seq_len": args.input_seq_len,
    }

    experiment_holdout(config)
# %%
