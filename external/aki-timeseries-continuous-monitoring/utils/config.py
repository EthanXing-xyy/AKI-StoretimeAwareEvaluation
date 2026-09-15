# %%
"""
Shared configuration for revision round 1 experiments.
Centralizes paths, model configs, feature definitions, and dataset settings.
"""

import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project paths for importing existing code
sys.path.insert(0, str(PROJECT_ROOT / "train_ml"))
sys.path.insert(0, str(PROJECT_ROOT / "train_cnn" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "train_cnn"))
sys.path.insert(0, str(PROJECT_ROOT / "train_transformer" / "src" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "train_transformer" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "train_lstm_attention" / "src" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "train_lstm_attention" / "src"))

# Random seed
SEED = 42

# Feature definitions
DEMO_COL = ["age", "sex", "bmi"]
VITAL_COL = ["sbp", "dbp", "pulse", "resp", "spo2", "temp"]
LAB_COL = [
    "albumin", "alp", "alt", "aptt", "ast", "bilirubin",
    "bnp", "bun", "calcium", "chloride", "cholesterol",
    "ck", "creatinine", "crp", "esr", "ggt", "glucose",
    "hba1c", "hco3", "hdl", "hematocrit", "hemoglobin",
    "lactate", "ldh", "magnesium", "pco2", "pdw", "ph",
    "phosphorus", "platelet", "po2", "potassium",
    "procalcitonin", "protein", "pt_inr", "rbc",
    "rdw", "sodium", "tco2", "triglyceride",
    "uric_acid", "wbc",
]
ALL_TS_FEATURES = VITAL_COL + LAB_COL  # 48 time-series features
ALL_FEATURES = ALL_TS_FEATURES + DEMO_COL  # 51 total

# Display names for features (British English, first word capitalised, no units)
FEATURE_DISPLAY_NAMES = {
    # Demographics
    "age": "Age",
    "sex": "Sex",
    "bmi": "BMI",
    # Vital signs
    "sbp": "Systolic blood pressure",
    "dbp": "Diastolic blood pressure",
    "pulse": "Heart rate",
    "resp": "Respiratory rate",
    "spo2": "Oxygen saturation",
    "temp": "Temperature",
    # Laboratory
    "alt": "ALT",
    "ast": "AST",
    "albumin": "Albumin",
    "alp": "Alkaline phosphatase",
    "bnp": "BNP",
    "bun": "BUN",
    "hco3": "Bicarbonate",
    "ck": "CK",
    "crp": "CRP",
    "calcium": "Calcium",
    "chloride": "Chloride",
    "creatinine": "Creatinine",
    "esr": "ESR",
    "ggt": "GGT",
    "glucose": "Glucose",
    "hdl": "HDL",
    "hematocrit": "Haematocrit",
    "hemoglobin": "Haemoglobin",
    "hba1c": "HbA1c",
    "ldh": "LDH",
    "lactate": "Lactate",
    "magnesium": "Magnesium",
    "pdw": "PDW",
    "pt_inr": "PT INR",
    "phosphorus": "Phosphate",
    "platelet": "Platelet",
    "potassium": "Potassium",
    "procalcitonin": "Procalcitonin",
    "rbc": "RBC",
    "rdw": "RDW",
    "sodium": "Sodium",
    "bilirubin": "Total bilirubin",
    "cholesterol": "Total cholesterol",
    "protein": "Total protein",
    "triglyceride": "Triglyceride",
    "uric_acid": "Uric acid",
    "wbc": "WBC",
    "aptt": "aPTT",
    "pco2": "pCO2",
    "ph": "pH",
    "po2": "pO2",
    "tco2": "tCO2",
}


def feature_display(name: str) -> str:
    """Convert internal feature name to display name."""
    return FEATURE_DISPLAY_NAMES.get(name, name)

# Prediction horizons
PREDICTION_HORIZONS = [0, 48, 72]

# Time points for online simulation analysis (hours before event)
TIME_POINTS_ONLINE = [0, 12, 24, 36, 48, 60, 72]

# Dataset configurations
DATASETS = {
    "ilsan_test": {"display_name": "NHIS (Internal)", "hospital": "ilsan", "type": "internal"},
    "cchlmc_external": {"display_name": "CSHH (External)", "hospital": "cchlmc", "type": "external"},
    "mimic-iv_external": {"display_name": "MIMIC-IV (External)", "hospital": "mimic-iv", "type": "external"},
}

# Hospital list for cohort data
HOSPITALS = ["ilsan", "cchlmc", "mimic-iv"]

# Model configurations for loading predictions
DL_MODEL_CONFIGS = {
    "ITE Transformer": {
        "key": "ite_transformer",
        "pred_path_template": "result/predictions/ilsan/ite_transformer/prediction_window_{pw}/{dataset}.parquet",
        "online_path_template": "result/predictions/ilsan/ite_transformer/prediction_window_{pw}/online/{dataset}_online.parquet",
        "checkpoint_template": "result/checkpoints/ilsan/ite_transformer/prediction_window_{pw}/model_best.ckpt",
    },
    "LSTM-Attention": {
        "key": "lstm_attention",
        "pred_path_template": "result/predictions/ilsan/lstm_attention/prediction_window_{pw}/{dataset}.parquet",
        "online_path_template": "result/predictions/ilsan/lstm_attention/prediction_window_{pw}/online/{dataset}_online.parquet",
        "checkpoint_template": "result/checkpoints/ilsan/lstm_attention/prediction_window_{pw}/model_best.ckpt",
    },
    "Masked CNN": {
        "key": "mask_rms_cnn",
        "pred_path_template": "result/predictions/ilsan/mask_rms_cnn/prediction_window_{pw}/resolution_control/apply_prob_0.0/{dataset}.parquet",
        "online_path_template": "result/predictions/ilsan/mask_rms_cnn/prediction_window_{pw}/resolution_control/apply_prob_0.0/online/{dataset}_online.parquet",
        "checkpoint_template": "result/checkpoints/ilsan/mask_rms_cnn/prediction_window_{pw}/resolution_control/apply_prob_0.0/model_best.ckpt",
    },
}

BASELINE_MODEL_CONFIGS = {
    "XGBoost": {
        "key": "xgboost",
        "pred_path_template": "result/predictions/ilsan/xgboost/prediction_window_{pw}/resolution_control/apply_prob_0.0/{dataset}.parquet",
        "online_path_template": "result/predictions/ilsan/xgboost/prediction_window_{pw}/resolution_control/apply_prob_0.0/online/{dataset}_online.parquet",
        "checkpoint_template": "result/checkpoints/ilsan/xgboost/prediction_window_{pw}/resolution_control/apply_prob_0.0/model_best.joblib",
    },
    "Logistic Regression": {
        "key": "logistic_regression",
        "pred_path_template": "result/predictions/ilsan/logistic_regression/prediction_window_{pw}/resolution_control/apply_prob_0.0/{dataset}.parquet",
        "online_path_template": "result/predictions/ilsan/logistic_regression/prediction_window_{pw}/resolution_control/apply_prob_0.0/online/{dataset}_online.parquet",
        "checkpoint_template": "result/checkpoints/ilsan/logistic_regression/prediction_window_{pw}/resolution_control/apply_prob_0.0/model_best.joblib",
    },
}

ALL_MODEL_CONFIGS = {**DL_MODEL_CONFIGS, **BASELINE_MODEL_CONFIGS}

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
SCALING_INFO_DIR = PROCESSED_DIR / "scaling_info"

# Result paths
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Display names and figure styling ──────────────────────────────────
MODEL_DISPLAY_NAMES = {
    "ITE Transformer": "TF",
    "LSTM-Attention": "LSTM",
    "Masked CNN": "CNN",
    "XGBoost": "XGB",
    "Logistic Regression": "LR",
}

# Reverse mapping (short → internal name)
MODEL_INTERNAL_NAMES = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}

# Consistent model colors (used across all figures)
MODEL_COLORS = {
    "TF":   "#1f77b4",  # Blue
    "LSTM": "#ff7f0e",  # Orange
    "CNN":  "#2ca02c",  # Green
    "XGB":  "#d62728",  # Red
    "LR":   "#9467bd",  # Purple
}

# Model line styles
MODEL_LINESTYLES = {
    "TF":   "-",   # solid
    "LSTM": "--",  # dashed
    "CNN":  ":",   # dotted
    "XGB":  "-.",  # dash-dot
    "LR":   (0, (3, 1, 1, 1)),  # densely dash-dotted
}

# Model markers
MODEL_MARKERS = {
    "TF":   "^",  # triangle
    "LSTM": "o",  # circle
    "CNN":  "s",  # square
    "XGB":  "D",  # diamond
    "LR":   "v",  # inverted triangle
}

# Dataset order (NHIS → CSHH → MIMIC-IV)
DATASET_ORDER = ["ilsan_test", "cchlmc_external", "mimic-iv_external"]

DATASET_DISPLAY = {
    "ilsan_test": "NHIS (Internal)",
    "cchlmc_external": "CSHH (External)",
    "mimic-iv_external": "MIMIC-IV (External)",
}

DATASET_COLORS = {
    "ilsan_test":        "#1565C0",  # Blue
    "cchlmc_external":   "#E65100",  # Orange
    "mimic-iv_external": "#2E7D32",  # Green
}

DATASET_PANEL_LABELS = {
    "ilsan_test": "A",
    "cchlmc_external": "B",
    "mimic-iv_external": "C",
}

# Prediction horizon styling (for overlay plots)
PH_COLORS = {0: "#1565C0", 48: "#E57373", 72: "#9E9E9E"}
PH_ALPHAS = {0: 1.0, 48: 0.7, 72: 0.55}
PH_LW_MULT = {0: 1.3, 48: 1.0, 72: 0.9}

# Model render order (DL first, then baselines)
MODEL_ORDER = ["TF", "LSTM", "CNN", "XGB", "LR"]

# Figure defaults
FIGURE_DPI_DISPLAY = 300
FIGURE_DPI_EXPORT = 500
FIGURE_FONT_FAMILY = "DejaVu Sans"


def model_display(internal_name: str) -> str:
    """Convert internal model name to display name."""
    return MODEL_DISPLAY_NAMES.get(internal_name, internal_name)


def model_internal(display_name: str) -> str:
    """Convert display name to internal model name."""
    return MODEL_INTERNAL_NAMES.get(display_name, display_name)


def get_master_path(hospital: str) -> Path:
    return PROCESSED_DIR / hospital / "cohort" / "master_include.parquet"


def get_pred_path(model_config: dict, pw: int, dataset: str) -> Path:
    return PROJECT_ROOT / model_config["pred_path_template"].format(pw=pw, dataset=dataset)


def get_online_path(model_config: dict, pw: int, dataset: str) -> Path:
    return PROJECT_ROOT / model_config["online_path_template"].format(pw=pw, dataset=dataset)


def get_checkpoint_path(model_config: dict, pw: int) -> Path:
    return PROJECT_ROOT / model_config["checkpoint_template"].format(pw=pw)
