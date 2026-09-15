# %%
"""
Step 3: Permutation Feature Importance (Single-Point Evaluation)

For each DL model × horizon × dataset, compute permutation-based feature importance.
- Shuffle each feature across patients (patient-level permutation for time series)
- Measure AUROC drop
- Repeat 10 times, report mean ± std

Also compute permutation FI for baseline models (XGBoost, LR).
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gc
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils.config import (
    SEED, PROJECT_ROOT, PREDICTION_HORIZONS, DATASETS,
    DL_MODEL_CONFIGS, BASELINE_MODEL_CONFIGS, ALL_MODEL_CONFIGS,
    ALL_TS_FEATURES, DEMO_COL, VITAL_COL, LAB_COL,
    RESULTS_DIR, FIGURES_DIR, PROCESSED_DIR,
    get_checkpoint_path, feature_display,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "feature_importance"
FIGURES_OUT = FIGURES_DIR / "feature_importance"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_REPEATS = 5
BATCH_SIZE_GPU = 512
MAX_SAMPLES = 2000  # Stratified random sampling per dataset


# %%
# =============================================================================
# Feature definitions
# =============================================================================

# CNN: 48 time-series features (indices 0-47) + 3 demo features (separate tensor)
# Transformer/LSTM: event triples [feature_id, time, value] where feature_id maps to 51 features
# Baseline: flat (51,) vector: 48 ts features + 3 demo features

FEATURE_NAMES_CNN_TS = VITAL_COL + LAB_COL  # 48 time-series
FEATURE_NAMES_ALL = VITAL_COL + LAB_COL + DEMO_COL  # 51 total

# Feature encoding for transformer/LSTM (demo_col first, then vital, then lab)
DEMO_COL_ENC = ["age", "sex", "bmi"]
FEATURE_ENCODING = {
    **{col: i for i, col in enumerate(DEMO_COL_ENC)},
    **{col: i + len(DEMO_COL_ENC) for i, col in enumerate(VITAL_COL)},
    **{col: i + len(DEMO_COL_ENC) + len(VITAL_COL) for i, col in enumerate(LAB_COL)},
}


# %%
# =============================================================================
# Model Loading Helpers
# =============================================================================

def _clear_and_setup_paths(model_type: str):
    """Clear conflicting modules and setup sys.path for the given model type."""
    # Remove potentially conflicting modules (including the 'src' package itself)
    prefixes = ("models.", "lightning_modules.", "optimizers.",
                "src.", "src")
    to_remove = [k for k in sys.modules
                 if any(k == p or k.startswith(p + ".") if not p.endswith(".") else k.startswith(p)
                        for p in prefixes)]
    # Also remove exact matches
    for exact in ["models", "lightning_modules", "optimizers", "src"]:
        if exact in sys.modules:
            to_remove.append(exact)
    for k in set(to_remove):
        sys.modules.pop(k, None)

    # Remove old model paths from sys.path
    model_dirs = ["train_cnn", "train_transformer", "train_transformer/src",
                  "train_lstm_attention", "train_lstm_attention/src"]
    for d in model_dirs:
        full = str(PROJECT_ROOT / d)
        while full in sys.path:
            sys.path.remove(full)

    # Add paths for the requested model type
    if model_type == "cnn":
        sys.path.insert(0, str(PROJECT_ROOT / "train_cnn"))
    elif model_type == "transformer":
        sys.path.insert(0, str(PROJECT_ROOT / "train_transformer" / "src"))
        sys.path.insert(0, str(PROJECT_ROOT / "train_transformer"))
    elif model_type == "lstm":
        sys.path.insert(0, str(PROJECT_ROOT / "train_lstm_attention" / "src"))
        sys.path.insert(0, str(PROJECT_ROOT / "train_lstm_attention"))


def load_cnn_model(pw: int):
    """Load CNN model from checkpoint."""
    _clear_and_setup_paths("cnn")
    from models.mask_rms_1d_cnn import Timeseries_CNN_Model
    from lightning_modules.classifier_module import AKI_Simple_TrainModule

    backbone = Timeseries_CNN_Model(
        num_demo_features=3,
        num_timeseries_features=48,
        num_classes=2,
        seq_len=256,
    )
    ckpt = get_checkpoint_path(DL_MODEL_CONFIGS["Masked CNN"], pw)
    module = AKI_Simple_TrainModule.load_from_checkpoint(
        str(ckpt), backbone_model=backbone, num_class=2, ordinal_class=False,
    )
    module.eval()
    module.to(DEVICE)
    return module


def load_transformer_model(pw: int):
    """Load ITE Transformer from checkpoint."""
    _clear_and_setup_paths("transformer")
    from src.models.ite_transformer import ITETransformer
    from src.lightning_modules.classifier_module import AKI_Simple_TrainModule

    config = {
        "d_model": 128, "num_layers": 2, "num_heads": 4,
        "dropout": 0.1, "num_classes": 2, "type_dict": FEATURE_ENCODING,
    }
    backbone = ITETransformer(config=config)
    ckpt = get_checkpoint_path(DL_MODEL_CONFIGS["ITE Transformer"], pw)
    module = AKI_Simple_TrainModule.load_from_checkpoint(
        str(ckpt), backbone_model=backbone, num_classes=2, ordinal_class=False,
    )
    module.eval()
    module.to(DEVICE)
    return module


def load_lstm_model(pw: int):
    """Load LSTM-Attention from checkpoint."""
    _clear_and_setup_paths("lstm")
    from src.models.lstm_attention import LSTMAttentionModel
    from src.lightning_modules.classifier_module import AKI_Simple_TrainModule

    config = {
        "d_model": 64, "num_layers": 2, "num_lstm_layers": 2, "num_heads": 4,
        "dropout": 0.1, "bidirectional": True, "num_classes": 2,
        "type_dict": FEATURE_ENCODING, "max_seq_len": 1024,
        "gradient_checkpointing": False, "ffn_expansion": 2,
    }
    backbone = LSTMAttentionModel(config=config)
    ckpt = get_checkpoint_path(DL_MODEL_CONFIGS["LSTM-Attention"], pw)
    module = AKI_Simple_TrainModule.load_from_checkpoint(
        str(ckpt), backbone_model=backbone, num_classes=2, ordinal_class=False,
    )
    module.eval()
    module.to(DEVICE)
    return module


# %%
# =============================================================================
# Data Loading for Feature Importance
# =============================================================================

def load_cnn_test_data(pw: int, hospital: str) -> tuple:
    """Load CNN-format test data (raw arrays before collation)."""
    data_dir = PROCESSED_DIR / hospital / "model" / f"prediction_window_{pw}"
    master_path = data_dir / "master.parquet"
    ts_dir = data_dir / "timeseries"

    master = pd.read_parquet(master_path)
    if "inclusion_yn" in master.columns:
        master = master.query("inclusion_yn == 1 & vital_yn == 1")
    if "error_yn" in master.columns:
        master = master.query("error_yn == 0")

    # For ilsan: use test split; for external: use all
    if hospital == "ilsan":
        rng = np.random.RandomState(1004)  # Same seed as training
        indices = np.arange(len(master))
        rng.shuffle(indices)
        n = len(master)
        test_start = int(n * 0.8)
        test_idx = indices[test_start:]
        master = master.iloc[test_idx].reset_index(drop=True)

    # Stratified sampling to MAX_SAMPLES
    if len(master) > MAX_SAMPLES:
        aki = master[master["label"] == 1]
        non_aki = master[master["label"] == 0]
        n_aki = max(1, int(MAX_SAMPLES * len(aki) / len(master)))
        n_non_aki = MAX_SAMPLES - n_aki
        sampled = pd.concat([
            aki.sample(min(n_aki, len(aki)), random_state=SEED),
            non_aki.sample(min(n_non_aki, len(non_aki)), random_state=SEED),
        ]).reset_index(drop=True)
        master = sampled
        logger.info(f"  Sampled {len(master)} patients (AKI={master['label'].sum()})")

    # Load scaling info
    scale_info = pd.read_parquet(PROCESSED_DIR / "scaling_info" / "ilsan" / "scaling_info.parquet", engine="fastparquet")

    # Load all patient data
    all_data = []
    all_meta = []
    all_labels = []
    valid_indices = []

    for idx, row in master.iterrows():
        fpath = ts_dir / f"{row['visit_id']}.gz"
        if not fpath.exists():
            continue
        d = joblib.load(fpath)
        ts_data = d["data"]  # (seq_len, 48)

        # Pad/truncate to 256
        if ts_data.shape[0] < 256:
            pad = np.full((256 - ts_data.shape[0], 48), np.nan)
            ts_data = np.concatenate([pad, ts_data], axis=0)
        elif ts_data.shape[0] > 256:
            ts_data = ts_data[-256:]

        sex_val = 0 if row.get("sex", "M") == "M" else 1
        meta = np.array([row["age"], sex_val, row["bmi"]], dtype=np.float64)

        all_data.append(ts_data)
        all_meta.append(meta)
        all_labels.append(int(row["label"]))
        valid_indices.append(idx)

    return np.array(all_data), np.array(all_meta), np.array(all_labels), scale_info


def load_transformer_test_data(pw: int, hospital: str) -> tuple:
    """Load transformer/LSTM-format test data (event triples)."""
    data_dir = PROCESSED_DIR / hospital / "transformer" / f"prediction_window_{pw}"
    master_path = data_dir / "master.parquet"
    ts_dir = data_dir / "timeseries"

    master = pd.read_parquet(master_path)
    if "inclusion_yn" in master.columns:
        master = master.query("inclusion_yn == 1 & vital_yn == 1")

    if hospital == "ilsan":
        rng = np.random.RandomState(1004)
        indices = np.arange(len(master))
        rng.shuffle(indices)
        n = len(master)
        test_start = int(n * 0.8)
        test_idx = indices[test_start:]
        master = master.iloc[test_idx].reset_index(drop=True)

    # Stratified sampling to MAX_SAMPLES
    if len(master) > MAX_SAMPLES:
        aki = master[master["label"] == 1]
        non_aki = master[master["label"] == 0]
        n_aki = max(1, int(MAX_SAMPLES * len(aki) / len(master)))
        n_non_aki = MAX_SAMPLES - n_aki
        sampled = pd.concat([
            aki.sample(min(n_aki, len(aki)), random_state=SEED),
            non_aki.sample(min(n_non_aki, len(non_aki)), random_state=SEED),
        ]).reset_index(drop=True)
        master = sampled
        logger.info(f"  Sampled {len(master)} patients (AKI={master['label'].sum()})")

    scale_info = pd.read_parquet(PROCESSED_DIR / "scaling_info" / "ilsan" / "scaling_info.parquet", engine="fastparquet")

    all_data = []
    all_labels = []

    for idx, row in master.iterrows():
        fpath = ts_dir / f"{row['visit_id']}.gz"
        if not fpath.exists():
            continue
        d = joblib.load(fpath)
        data = d["data"]  # (seq_len, 3) event triples
        all_data.append(data)
        all_labels.append(int(row["label"]))

    return all_data, np.array(all_labels), scale_info


# %%
# =============================================================================
# Inference Functions
# =============================================================================

def infer_cnn_batch(model, data_batch, meta_batch, scale_info) -> np.ndarray:
    """Run CNN inference on a batch. Returns probabilities."""
    # Normalize
    feat_col = VITAL_COL + LAB_COL
    demo_col = ["age", "bmi"]

    ts_scale = scale_info.query("feature in @feat_col").set_index("feature").loc[feat_col]
    demo_scale = scale_info.query("feature in @demo_col").set_index("feature").loc[demo_col]

    ts_medians = ts_scale["median"].values
    ts_iqrs = ts_scale["iqr"].values
    demo_medians = demo_scale["median"].values
    demo_iqrs = demo_scale["iqr"].values

    # Normalize data
    data_norm = data_batch.copy().astype(np.float32)
    data_norm = (data_norm - ts_medians) / ts_iqrs
    data_norm = np.nan_to_num(data_norm, nan=0.0)

    # Normalize meta (age, bmi only - sex stays as-is)
    meta_norm = meta_batch.copy().astype(np.float32)
    meta_norm[:, 0] = (meta_norm[:, 0] - demo_medians[0]) / demo_iqrs[0]  # age
    meta_norm[:, 2] = (meta_norm[:, 2] - demo_medians[1]) / demo_iqrs[1]  # bmi

    data_t = torch.from_numpy(data_norm).float().to(DEVICE)
    meta_t = torch.from_numpy(meta_norm).float().to(DEVICE)

    with torch.no_grad():
        logits = model(meta_t, data_t)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def collate_transformer_batch(data_list, scale_info):
    """Collate variable-length event triples into padded batch."""
    # Normalize values
    scale_dict = {}
    for _, row in scale_info.iterrows():
        scale_dict[int(row["feature_id"])] = (row["median"], row["iqr"])

    batch_data = []
    lengths = []
    for data in data_list:
        data = data.copy().astype(np.float32)
        # Normalize value column based on feature_id
        for i in range(len(data)):
            fid = int(data[i, 0])
            if fid in scale_dict:
                med, iqr = scale_dict[fid]
                if iqr > 0:
                    data[i, 2] = (data[i, 2] - med) / iqr
        batch_data.append(data)
        lengths.append(len(data))

    max_len = max(lengths)
    padded = np.zeros((len(data_list), max_len, 3), dtype=np.float32)
    mask = np.zeros((len(data_list), max_len), dtype=bool)

    for i, (data, l) in enumerate(zip(batch_data, lengths)):
        padded[i, :l] = data
        mask[i, :l] = True

    return torch.from_numpy(padded).to(DEVICE), torch.from_numpy(mask).to(DEVICE)


def infer_transformer_batch(model, data_list, scale_info) -> np.ndarray:
    """Run Transformer/LSTM inference on a batch."""
    data_t, mask_t = collate_transformer_batch(data_list, scale_info)
    with torch.no_grad():
        logits = model(data_t, mask_t)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


# %%
# =============================================================================
# Permutation Feature Importance - CNN
# =============================================================================

def permutation_fi_cnn(
    model, data: np.ndarray, meta: np.ndarray, labels: np.ndarray,
    scale_info: pd.DataFrame, n_repeats: int = N_REPEATS, seed: int = SEED,
) -> pd.DataFrame:
    """Compute permutation feature importance for CNN model."""
    rng = np.random.RandomState(seed)
    n_samples = len(data)

    # Baseline AUROC
    all_probs = []
    for start in range(0, n_samples, BATCH_SIZE_GPU):
        end = min(start + BATCH_SIZE_GPU, n_samples)
        probs = infer_cnn_batch(model, data[start:end], meta[start:end], scale_info)
        all_probs.append(probs)
    baseline_probs = np.concatenate(all_probs)
    baseline_auroc = roc_auc_score(labels, baseline_probs)
    logger.info(f"  Baseline AUROC: {baseline_auroc:.4f}")

    results = []

    # Permute time-series features (48)
    for feat_idx, feat_name in enumerate(tqdm(FEATURE_NAMES_CNN_TS, desc="  TS features")):
        importances = []
        for rep in range(n_repeats):
            perm_data = data.copy()
            perm_idx = rng.permutation(n_samples)
            perm_data[:, :, feat_idx] = data[perm_idx, :, feat_idx]

            perm_probs = []
            for start in range(0, n_samples, BATCH_SIZE_GPU):
                end = min(start + BATCH_SIZE_GPU, n_samples)
                probs = infer_cnn_batch(model, perm_data[start:end], meta[start:end], scale_info)
                perm_probs.append(probs)
            perm_probs = np.concatenate(perm_probs)
            perm_auroc = roc_auc_score(labels, perm_probs)
            importances.append(baseline_auroc - perm_auroc)

        results.append({
            "feature_name": feat_name,
            "mean_importance": np.mean(importances),
            "std_importance": np.std(importances),
            "baseline_auroc": baseline_auroc,
            "permuted_auroc_mean": baseline_auroc - np.mean(importances),
        })

    # Permute demographic features (3)
    for demo_idx, demo_name in enumerate(tqdm(DEMO_COL, desc="  Demo features")):
        importances = []
        for rep in range(n_repeats):
            perm_meta = meta.copy()
            perm_idx = rng.permutation(n_samples)
            perm_meta[:, demo_idx] = meta[perm_idx, demo_idx]

            perm_probs = []
            for start in range(0, n_samples, BATCH_SIZE_GPU):
                end = min(start + BATCH_SIZE_GPU, n_samples)
                probs = infer_cnn_batch(model, data[start:end], perm_meta[start:end], scale_info)
                perm_probs.append(probs)
            perm_probs = np.concatenate(perm_probs)
            perm_auroc = roc_auc_score(labels, perm_probs)
            importances.append(baseline_auroc - perm_auroc)

        results.append({
            "feature_name": demo_name,
            "mean_importance": np.mean(importances),
            "std_importance": np.std(importances),
            "baseline_auroc": baseline_auroc,
            "permuted_auroc_mean": baseline_auroc - np.mean(importances),
        })

    return pd.DataFrame(results)


# %%
# =============================================================================
# Permutation Feature Importance - Transformer/LSTM
# =============================================================================

def permutation_fi_transformer(
    model, data_list: list, labels: np.ndarray,
    scale_info: pd.DataFrame, n_repeats: int = N_REPEATS, seed: int = SEED,
) -> pd.DataFrame:
    """Compute permutation feature importance for Transformer/LSTM (event triple format)."""
    rng = np.random.RandomState(seed)
    n_samples = len(data_list)

    # Baseline AUROC
    all_probs = []
    for start in range(0, n_samples, BATCH_SIZE_GPU):
        end = min(start + BATCH_SIZE_GPU, n_samples)
        probs = infer_transformer_batch(model, data_list[start:end], scale_info)
        all_probs.append(probs)
    baseline_probs = np.concatenate(all_probs)
    baseline_auroc = roc_auc_score(labels, baseline_probs)
    logger.info(f"  Baseline AUROC: {baseline_auroc:.4f}")

    # Build feature_id to feature_name mapping
    id_to_name = {v: k for k, v in FEATURE_ENCODING.items()}

    results = []

    for feat_id in tqdm(range(51), desc="  Features"):
        feat_name = id_to_name[feat_id]
        importances = []

        for rep in range(n_repeats):
            perm_idx = rng.permutation(n_samples)

            # Create permuted data: for each patient, replace events with matching feature_id
            # with events from the permuted patient
            perm_data = []
            for i in range(n_samples):
                orig = data_list[i].copy()
                donor = data_list[perm_idx[i]]

                # Find events for this feature in both original and donor
                orig_mask = orig[:, 0].astype(int) == feat_id
                donor_events = donor[donor[:, 0].astype(int) == feat_id]

                if orig_mask.any() and len(donor_events) > 0:
                    # Remove original events for this feature
                    other_events = orig[~orig_mask]
                    # Add donor's events for this feature
                    perm_patient = np.concatenate([other_events, donor_events], axis=0)
                    # Sort by time
                    perm_patient = perm_patient[perm_patient[:, 1].argsort()]
                elif orig_mask.any() and len(donor_events) == 0:
                    # Donor has no events for this feature - just remove from original
                    perm_patient = orig[~orig_mask]
                else:
                    perm_patient = orig

                if len(perm_patient) == 0:
                    perm_patient = orig  # Fallback

                perm_data.append(perm_patient)

            # Compute permuted AUROC
            perm_probs = []
            for start in range(0, n_samples, BATCH_SIZE_GPU):
                end = min(start + BATCH_SIZE_GPU, n_samples)
                probs = infer_transformer_batch(model, perm_data[start:end], scale_info)
                perm_probs.append(probs)
            perm_probs = np.concatenate(perm_probs)
            perm_auroc = roc_auc_score(labels, perm_probs)
            importances.append(baseline_auroc - perm_auroc)

        results.append({
            "feature_name": feat_name,
            "mean_importance": np.mean(importances),
            "std_importance": np.std(importances),
            "baseline_auroc": baseline_auroc,
            "permuted_auroc_mean": baseline_auroc - np.mean(importances),
        })

    return pd.DataFrame(results)


# %%
# =============================================================================
# Permutation Feature Importance - Baseline (XGBoost, LR)
# =============================================================================

def permutation_fi_baseline(
    model_name: str, pw: int, dataset: str,
    n_repeats: int = N_REPEATS, seed: int = SEED,
) -> pd.DataFrame:
    """Compute permutation FI for baseline models using existing predictions."""
    from utils.data_loader import load_predictions

    rng = np.random.RandomState(seed)

    # Load model
    config = BASELINE_MODEL_CONFIGS[model_name]
    ckpt_path = get_checkpoint_path(config, pw)
    model = joblib.load(ckpt_path)

    # Load test data using the ML pipeline
    sys.path.insert(0, str(PROJECT_ROOT / "train_ml"))
    from data_utils import build_dataset, load_scaling_info

    hospital_map = {"ilsan_test": "ilsan", "cchlmc_external": "cchlmc", "mimic-iv_external": "mimic-iv"}
    hospital = hospital_map[dataset]

    data_dir = str(PROCESSED_DIR / hospital / "model" / f"prediction_window_{pw}")
    master = pd.read_parquet(f"{data_dir}/master.parquet")
    if "inclusion_yn" in master.columns:
        master = master.query("inclusion_yn == 1 & vital_yn == 1")
    if "error_yn" in master.columns:
        master = master.query("error_yn == 0")

    if hospital == "ilsan":
        rng_split = np.random.RandomState(1004)
        indices = np.arange(len(master))
        rng_split.shuffle(indices)
        test_start = int(len(master) * 0.8)
        test_idx = indices[test_start:]
        master = master.iloc[test_idx].reset_index(drop=True)

    # Stratified sampling to MAX_SAMPLES
    if len(master) > MAX_SAMPLES:
        aki = master[master["label"] == 1]
        non_aki = master[master["label"] == 0]
        n_aki = max(1, int(MAX_SAMPLES * len(aki) / len(master)))
        n_non_aki = MAX_SAMPLES - n_aki
        sampled = pd.concat([
            aki.sample(min(n_aki, len(aki)), random_state=SEED),
            non_aki.sample(min(n_non_aki, len(non_aki)), random_state=SEED),
        ]).reset_index(drop=True)
        master = sampled
        logger.info(f"  Sampled {len(master)} patients (AKI={master['label'].sum()})")

    # Monkey-patch pd.read_parquet to use fastparquet for scaling_info
    _orig_rp = pd.read_parquet
    def _patched_rp(path, *a, **kw):
        if "scaling_info" in str(path):
            kw["engine"] = "fastparquet"
        return _orig_rp(path, *a, **kw)
    pd.read_parquet = _patched_rp
    try:
        scaling_info = load_scaling_info(str(PROCESSED_DIR))
        X, y, _ = build_dataset(master, f"{data_dir}/timeseries", scaling_info, normalize=True, num_workers=8)
    finally:
        pd.read_parquet = _orig_rp

    # Baseline prediction
    if hasattr(model, "predict_proba"):
        baseline_probs = model.predict_proba(X)[:, 1]
    else:
        baseline_probs = model.predict(X)

    baseline_auroc = roc_auc_score(y, baseline_probs)
    logger.info(f"  Baseline AUROC: {baseline_auroc:.4f}")

    feature_names = ALL_TS_FEATURES + DEMO_COL  # 51 features
    n_samples = len(X)
    results = []

    for feat_idx, feat_name in enumerate(tqdm(feature_names, desc=f"  {model_name} features")):
        importances = []
        for rep in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(n_samples)
            X_perm[:, feat_idx] = X[perm_idx, feat_idx]

            if hasattr(model, "predict_proba"):
                perm_probs = model.predict_proba(X_perm)[:, 1]
            else:
                perm_probs = model.predict(X_perm)

            perm_auroc = roc_auc_score(y, perm_probs)
            importances.append(baseline_auroc - perm_auroc)

        results.append({
            "feature_name": feat_name,
            "mean_importance": np.mean(importances),
            "std_importance": np.std(importances),
            "baseline_auroc": baseline_auroc,
            "permuted_auroc_mean": baseline_auroc - np.mean(importances),
        })

    return pd.DataFrame(results)


# %%
# =============================================================================
# Visualization
# =============================================================================

def plot_feature_importance(fi_df: pd.DataFrame, model: str, horizon: int, dataset: str, top_n: int = 15):
    """Plot top-N feature importance bar chart."""
    df = fi_df.sort_values("mean_importance", ascending=True).tail(top_n).copy()
    df["display_name"] = df["feature_name"].map(feature_display)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#F44336" if v > 0 else "#2196F3" for v in df["mean_importance"]]
    ax.barh(df["display_name"], df["mean_importance"], xerr=df["std_importance"],
            color=colors, edgecolor="white", capsize=3, alpha=0.8)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("AUROC Drop (Importance)", fontsize=11)
    ax.text(0.98, 0.02, f"Baseline AUROC = {df['baseline_auroc'].iloc[0]:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="gray")
    plt.tight_layout()

    safe_model = model.lower().replace(" ", "_").replace("-", "_")
    safe_dataset = dataset.replace("-", "_")
    fig_path = FIGURES_OUT / f"fi_single_{safe_model}_pw{horizon}_{safe_dataset}.pdf"
    fig.savefig(fig_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# %%
# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("Step 3: Permutation Feature Importance (Single-Point)")
    logger.info("=" * 70)

    all_results = []
    hospital_map = {"ilsan_test": "ilsan", "cchlmc_external": "cchlmc", "mimic-iv_external": "mimic-iv"}

    # DL Models
    dl_model_loaders = {
        "Masked CNN": ("cnn", load_cnn_model),
        "ITE Transformer": ("transformer", load_transformer_model),
        "LSTM-Attention": ("lstm", load_lstm_model),
    }

    for model_name, (model_type, loader_fn) in dl_model_loaders.items():
        for pw in PREDICTION_HORIZONS:
            logger.info(f"\n{'='*50}")
            logger.info(f"Loading {model_name} PW={pw}h")
            try:
                model = loader_fn(pw)
            except Exception as e:
                logger.error(f"Failed to load {model_name} PW={pw}h: {e}")
                continue

            for dataset_name, ds_info in DATASETS.items():
                hospital = hospital_map[dataset_name]
                logger.info(f"  Computing FI for {dataset_name}...")

                try:
                    if model_type == "cnn":
                        data, meta, labels, scale_info = load_cnn_test_data(pw, hospital)
                        fi_df = permutation_fi_cnn(model, data, meta, labels, scale_info)
                    else:
                        data_list, labels, scale_info = load_transformer_test_data(pw, hospital)
                        fi_df = permutation_fi_transformer(model, data_list, labels, scale_info)

                    fi_df["model"] = model_name
                    fi_df["horizon"] = pw
                    fi_df["dataset"] = dataset_name
                    all_results.append(fi_df)

                    # Plot
                    fig_path = plot_feature_importance(fi_df, model_name, pw, dataset_name)
                    logger.info(f"  Saved figure: {fig_path}")

                    # Save intermediate
                    safe_model = model_name.lower().replace(" ", "_").replace("-", "_")
                    safe_ds = dataset_name.replace("-", "_")
                    fi_df.to_csv(RESULTS_OUT / f"fi_single_{safe_model}_pw{pw}_{safe_ds}.csv", index=False)

                except Exception as e:
                    logger.error(f"  Error: {e}", exc_info=True)

            # Free GPU memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Baseline Models
    for model_name in BASELINE_MODEL_CONFIGS:
        for pw in PREDICTION_HORIZONS:
            for dataset_name in DATASETS:
                logger.info(f"\n{model_name} PW={pw}h {dataset_name}...")
                try:
                    fi_df = permutation_fi_baseline(model_name, pw, dataset_name)
                    fi_df["model"] = model_name
                    fi_df["horizon"] = pw
                    fi_df["dataset"] = dataset_name
                    all_results.append(fi_df)

                    fig_path = plot_feature_importance(fi_df, model_name, pw, dataset_name)
                    logger.info(f"  Saved: {fig_path}")

                    safe_model = model_name.lower().replace(" ", "_")
                    safe_ds = dataset_name.replace("-", "_")
                    fi_df.to_csv(RESULTS_OUT / f"fi_single_{safe_model}_pw{pw}_{safe_ds}.csv", index=False)

                except Exception as e:
                    logger.error(f"  Error: {e}", exc_info=True)

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(RESULTS_OUT / "feature_importance_single_all.csv", index=False)
        logger.info(f"\nSaved combined results: {RESULTS_OUT / 'feature_importance_single_all.csv'}")

    logger.info("\nStep 3 Complete!")


# %%
if __name__ == "__main__":
    main()
