# %%
"""
Step 5: Feature Subset Performance Comparison

Evaluate model performance when using only top-ranked features based on
Step 3 permutation feature importance.

Feature subsets:
- Top 50% (~25 features)
- Top 33% (~17 features)
- Top 20% (~10 features)

Method:
- Baseline (XGBoost, LR): Zero-masking of excluded features (flat vector).
- CNN: Zero-masking of excluded feature channels in (batch, 256, 48) + demo.
- Transformer/LSTM: Remove events with excluded feature_ids from event triples.
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

from sklearn.metrics import roc_auc_score, average_precision_score

from utils.config import (
    SEED, PREDICTION_HORIZONS, DATASETS, PROCESSED_DIR,
    DL_MODEL_CONFIGS, BASELINE_MODEL_CONFIGS,
    ALL_TS_FEATURES, DEMO_COL, VITAL_COL, LAB_COL,
    RESULTS_DIR, FIGURES_DIR, get_checkpoint_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "sensitivity_analysis"
FIGURES_OUT = FIGURES_DIR / "sensitivity_analysis"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_GPU = 512

FEATURE_SUBSETS = {
    "top_50pct": 0.50,
    "top_33pct": 0.33,
    "top_20pct": 0.20,
}

# Feature encoding for transformer/LSTM (demo_col first, then vital, then lab)
DEMO_COL_ENC = ["age", "sex", "bmi"]
FEATURE_ENCODING = {
    **{col: i for i, col in enumerate(DEMO_COL_ENC)},
    **{col: i + len(DEMO_COL_ENC) for i, col in enumerate(VITAL_COL)},
    **{col: i + len(DEMO_COL_ENC) + len(VITAL_COL) for i, col in enumerate(LAB_COL)},
}
HOSPITAL_MAP = {"ilsan_test": "ilsan", "cchlmc_external": "cchlmc", "mimic-iv_external": "mimic-iv"}


# %%
def load_fi_rankings() -> pd.DataFrame:
    """Load feature importance rankings from Step 3."""
    fi_path = RESULTS_DIR / "feature_importance" / "feature_importance_single_all.csv"
    if fi_path.exists():
        return pd.read_csv(fi_path)
    # Try individual files
    fi_dir = RESULTS_DIR / "feature_importance"
    fi_files = list(fi_dir.glob("fi_single_*.csv"))
    if fi_files:
        return pd.concat([pd.read_csv(f) for f in fi_files], ignore_index=True)
    logger.warning("No feature importance results found from Step 3!")
    return pd.DataFrame()


def get_top_features(fi_df: pd.DataFrame, model: str, pw: int, dataset: str, fraction: float) -> list[str]:
    """Get top features by importance ranking."""
    subset = fi_df.query("model == @model and horizon == @pw and dataset == @dataset")
    if subset.empty:
        # Fallback: use any available ranking for this model
        subset = fi_df.query("model == @model")
    if subset.empty:
        return ALL_TS_FEATURES + DEMO_COL  # Return all if no ranking

    # Average across datasets if needed
    ranked = subset.groupby("feature_name")["mean_importance"].mean().sort_values(ascending=False)
    n_keep = max(1, int(len(ranked) * fraction))
    return ranked.head(n_keep).index.tolist()


# %%
# =============================================================================
# DL Model Loading (reused from Step 3)
# =============================================================================

def _clear_and_setup_paths(model_type: str):
    """Clear conflicting modules and setup sys.path for the given model type."""
    prefixes = ("models.", "lightning_modules.", "optimizers.",
                "src.", "src")
    to_remove = [k for k in sys.modules
                 if any(k == p or k.startswith(p + ".") if not p.endswith(".") else k.startswith(p)
                        for p in prefixes)]
    for exact in ["models", "lightning_modules", "optimizers", "src"]:
        if exact in sys.modules:
            to_remove.append(exact)
    for k in set(to_remove):
        sys.modules.pop(k, None)

    model_dirs = ["train_cnn", "train_transformer", "train_transformer/src",
                  "train_lstm_attention", "train_lstm_attention/src"]
    for d in model_dirs:
        full = str(PROJECT_ROOT / d)
        while full in sys.path:
            sys.path.remove(full)

    if model_type == "cnn":
        sys.path.insert(0, str(PROJECT_ROOT / "train_cnn"))
    elif model_type == "transformer":
        sys.path.insert(0, str(PROJECT_ROOT / "train_transformer" / "src"))
        sys.path.insert(0, str(PROJECT_ROOT / "train_transformer"))
    elif model_type == "lstm":
        sys.path.insert(0, str(PROJECT_ROOT / "train_lstm_attention" / "src"))
        sys.path.insert(0, str(PROJECT_ROOT / "train_lstm_attention"))


def load_cnn_model(pw: int):
    _clear_and_setup_paths("cnn")
    from models.mask_rms_1d_cnn import Timeseries_CNN_Model
    from lightning_modules.classifier_module import AKI_Simple_TrainModule

    backbone = Timeseries_CNN_Model(
        num_demo_features=3, num_timeseries_features=48,
        num_classes=2, seq_len=256,
    )
    ckpt = get_checkpoint_path(DL_MODEL_CONFIGS["Masked CNN"], pw)
    module = AKI_Simple_TrainModule.load_from_checkpoint(
        str(ckpt), backbone_model=backbone, num_class=2, ordinal_class=False,
    )
    module.eval()
    module.to(DEVICE)
    return module


def load_transformer_model(pw: int):
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
# DL Data Loading
# =============================================================================

def load_cnn_test_data(pw: int, hospital: str) -> tuple:
    """Load CNN-format test data."""
    data_dir = PROCESSED_DIR / hospital / "model" / f"prediction_window_{pw}"
    master = pd.read_parquet(data_dir / "master.parquet")
    if "inclusion_yn" in master.columns:
        master = master.query("inclusion_yn == 1 & vital_yn == 1")
    if "error_yn" in master.columns:
        master = master.query("error_yn == 0")

    if hospital == "ilsan":
        rng = np.random.RandomState(1004)
        indices = np.arange(len(master))
        rng.shuffle(indices)
        test_start = int(len(master) * 0.8)
        master = master.iloc[indices[test_start:]].reset_index(drop=True)

    scale_info = pd.read_parquet(
        PROCESSED_DIR / "scaling_info" / "ilsan" / "scaling_info.parquet",
        engine="fastparquet",
    )
    ts_dir = data_dir / "timeseries"

    all_data, all_meta, all_labels = [], [], []
    for _, row in master.iterrows():
        fpath = ts_dir / f"{row['visit_id']}.gz"
        if not fpath.exists():
            continue
        d = joblib.load(fpath)
        ts_data = d["data"]

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

    return np.array(all_data), np.array(all_meta), np.array(all_labels), scale_info


def load_transformer_test_data(pw: int, hospital: str) -> tuple:
    """Load transformer/LSTM-format test data (event triples)."""
    data_dir = PROCESSED_DIR / hospital / "transformer" / f"prediction_window_{pw}"
    master = pd.read_parquet(data_dir / "master.parquet")
    if "inclusion_yn" in master.columns:
        master = master.query("inclusion_yn == 1 & vital_yn == 1")

    if hospital == "ilsan":
        rng = np.random.RandomState(1004)
        indices = np.arange(len(master))
        rng.shuffle(indices)
        test_start = int(len(master) * 0.8)
        master = master.iloc[indices[test_start:]].reset_index(drop=True)

    scale_info = pd.read_parquet(
        PROCESSED_DIR / "scaling_info" / "ilsan" / "scaling_info.parquet",
        engine="fastparquet",
    )
    ts_dir = data_dir / "timeseries"

    all_data, all_labels = [], []
    for _, row in master.iterrows():
        fpath = ts_dir / f"{row['visit_id']}.gz"
        if not fpath.exists():
            continue
        d = joblib.load(fpath)
        all_data.append(d["data"])
        all_labels.append(int(row["label"]))

    return all_data, np.array(all_labels), scale_info


# %%
# =============================================================================
# DL Inference Functions
# =============================================================================

def infer_cnn_batch(model, data_batch, meta_batch, scale_info) -> np.ndarray:
    """Run CNN inference on a batch."""
    feat_col = VITAL_COL + LAB_COL
    demo_col = ["age", "bmi"]

    ts_scale = scale_info.query("feature in @feat_col").set_index("feature").loc[feat_col]
    demo_scale = scale_info.query("feature in @demo_col").set_index("feature").loc[demo_col]

    data_norm = data_batch.copy().astype(np.float32)
    data_norm = (data_norm - ts_scale["median"].values) / ts_scale["iqr"].values
    data_norm = np.nan_to_num(data_norm, nan=0.0)

    meta_norm = meta_batch.copy().astype(np.float32)
    meta_norm[:, 0] = (meta_norm[:, 0] - demo_scale["median"].values[0]) / demo_scale["iqr"].values[0]
    meta_norm[:, 2] = (meta_norm[:, 2] - demo_scale["median"].values[1]) / demo_scale["iqr"].values[1]

    data_t = torch.from_numpy(data_norm).float().to(DEVICE)
    meta_t = torch.from_numpy(meta_norm).float().to(DEVICE)

    with torch.no_grad():
        logits = model(meta_t, data_t)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def collate_transformer_batch(data_list, scale_info):
    """Collate variable-length event triples into padded batch."""
    scale_dict = {}
    for _, row in scale_info.iterrows():
        scale_dict[int(row["feature_id"])] = (row["median"], row["iqr"])

    batch_data = []
    lengths = []
    for data in data_list:
        data = data.copy().astype(np.float32)
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
# DL Feature Subset Evaluation
# =============================================================================

def evaluate_cnn_subset(
    model, data: np.ndarray, meta: np.ndarray, labels: np.ndarray,
    scale_info: pd.DataFrame, keep_features: list[str],
) -> dict:
    """Evaluate CNN with feature subset (zero-mask excluded channels)."""
    n = len(data)
    ts_features = VITAL_COL + LAB_COL  # 48 ts features
    demo_features = DEMO_COL  # age, sex, bmi

    # Full performance
    all_probs = []
    for s in range(0, n, BATCH_SIZE_GPU):
        e = min(s + BATCH_SIZE_GPU, n)
        all_probs.append(infer_cnn_batch(model, data[s:e], meta[s:e], scale_info))
    full_probs = np.concatenate(all_probs)
    full_auroc = roc_auc_score(labels, full_probs)
    full_auprc = average_precision_score(labels, full_probs)

    # Masked: zero out excluded ts feature channels
    ts_mask_indices = [i for i, f in enumerate(ts_features) if f not in keep_features]
    demo_mask_indices = [i for i, f in enumerate(demo_features) if f not in keep_features]

    data_masked = data.copy()
    meta_masked = meta.copy()
    if ts_mask_indices:
        data_masked[:, :, ts_mask_indices] = np.nan  # will become 0 after normalization
    if demo_mask_indices:
        for idx in demo_mask_indices:
            meta_masked[:, idx] = 0.0

    all_probs_m = []
    for s in range(0, n, BATCH_SIZE_GPU):
        e = min(s + BATCH_SIZE_GPU, n)
        all_probs_m.append(infer_cnn_batch(model, data_masked[s:e], meta_masked[s:e], scale_info))
    masked_probs = np.concatenate(all_probs_m)
    masked_auroc = roc_auc_score(labels, masked_probs)
    masked_auprc = average_precision_score(labels, masked_probs)

    return {
        "full_auroc": full_auroc,
        "full_auprc": full_auprc,
        "masked_auroc": masked_auroc,
        "masked_auprc": masked_auprc,
        "auroc_drop_pct": (full_auroc - masked_auroc) / full_auroc * 100,
        "n_features_kept": len(keep_features),
        "n_features_total": len(ts_features) + len(demo_features),
    }


def evaluate_transformer_subset(
    model, data_list: list, labels: np.ndarray,
    scale_info: pd.DataFrame, keep_features: list[str],
) -> dict:
    """Evaluate Transformer/LSTM with feature subset (remove excluded events)."""
    n = len(data_list)

    # Full performance
    all_probs = []
    for s in range(0, n, BATCH_SIZE_GPU):
        e = min(s + BATCH_SIZE_GPU, n)
        all_probs.append(infer_transformer_batch(model, data_list[s:e], scale_info))
    full_probs = np.concatenate(all_probs)
    full_auroc = roc_auc_score(labels, full_probs)
    full_auprc = average_precision_score(labels, full_probs)

    # Build set of feature_ids to keep
    keep_ids = {FEATURE_ENCODING[f] for f in keep_features if f in FEATURE_ENCODING}

    # Filter events: keep only events whose feature_id is in keep_ids
    masked_data = []
    for patient_data in data_list:
        mask = np.array([int(row[0]) in keep_ids for row in patient_data])
        filtered = patient_data[mask]
        if len(filtered) == 0:
            # Keep at least one dummy event to avoid empty sequences
            filtered = np.zeros((1, 3), dtype=patient_data.dtype)
        masked_data.append(filtered)

    all_probs_m = []
    for s in range(0, n, BATCH_SIZE_GPU):
        e = min(s + BATCH_SIZE_GPU, n)
        all_probs_m.append(infer_transformer_batch(model, masked_data[s:e], scale_info))
    masked_probs = np.concatenate(all_probs_m)
    masked_auroc = roc_auc_score(labels, masked_probs)
    masked_auprc = average_precision_score(labels, masked_probs)

    return {
        "full_auroc": full_auroc,
        "full_auprc": full_auprc,
        "masked_auroc": masked_auroc,
        "masked_auprc": masked_auprc,
        "auroc_drop_pct": (full_auroc - masked_auroc) / full_auroc * 100,
        "n_features_kept": len(keep_features),
        "n_features_total": 51,
    }


# %%
# =============================================================================
# Baseline Feature Subset Evaluation
# =============================================================================

def evaluate_baseline_subset(
    model_name: str, pw: int, dataset: str,
    keep_features: list[str],
) -> dict:
    """Evaluate baseline model with feature subset (zero-mask excluded features)."""
    sys.path.insert(0, str(PROJECT_ROOT / "train_ml"))
    from data_utils import build_dataset, load_scaling_info

    hospital = HOSPITAL_MAP[dataset]

    # Load model
    config = BASELINE_MODEL_CONFIGS[model_name]
    ckpt_path = get_checkpoint_path(config, pw)
    model = joblib.load(ckpt_path)

    # Load data
    data_dir = str(PROCESSED_DIR / hospital / "model" / f"prediction_window_{pw}")
    master = pd.read_parquet(f"{data_dir}/master.parquet")
    if "inclusion_yn" in master.columns:
        master = master.query("inclusion_yn == 1 & vital_yn == 1")
    if "error_yn" in master.columns:
        master = master.query("error_yn == 0")

    if hospital == "ilsan":
        rng = np.random.RandomState(1004)
        indices = np.arange(len(master))
        rng.shuffle(indices)
        test_start = int(len(master) * 0.8)
        master = master.iloc[indices[test_start:]].reset_index(drop=True)

    # Monkey-patch pd.read_parquet to use fastparquet for scaling_info
    _orig_read_parquet = pd.read_parquet
    def _patched_read_parquet(path, *args, **kwargs):
        if "scaling_info" in str(path):
            kwargs["engine"] = "fastparquet"
        return _orig_read_parquet(path, *args, **kwargs)
    pd.read_parquet = _patched_read_parquet
    try:
        scaling_info = load_scaling_info(str(PROCESSED_DIR))
        X, y, _ = build_dataset(master, f"{data_dir}/timeseries", scaling_info, normalize=True, num_workers=8)
    finally:
        pd.read_parquet = _orig_read_parquet

    # Full performance
    feature_names = ALL_TS_FEATURES + DEMO_COL
    if hasattr(model, "predict_proba"):
        full_probs = model.predict_proba(X)[:, 1]
    else:
        full_probs = model.predict(X)
    full_auroc = roc_auc_score(y, full_probs)
    full_auprc = average_precision_score(y, full_probs)

    # Masked performance
    mask_indices = [i for i, f in enumerate(feature_names) if f not in keep_features]
    X_masked = X.copy()
    X_masked[:, mask_indices] = 0.0

    if hasattr(model, "predict_proba"):
        masked_probs = model.predict_proba(X_masked)[:, 1]
    else:
        masked_probs = model.predict(X_masked)
    masked_auroc = roc_auc_score(y, masked_probs)
    masked_auprc = average_precision_score(y, masked_probs)

    return {
        "full_auroc": full_auroc,
        "full_auprc": full_auprc,
        "masked_auroc": masked_auroc,
        "masked_auprc": masked_auprc,
        "auroc_drop_pct": (full_auroc - masked_auroc) / full_auroc * 100,
        "n_features_kept": len(keep_features),
        "n_features_total": len(feature_names),
    }


# %%
def main():
    logger.info("=" * 70)
    logger.info("Step 5: Feature Subset Performance Comparison")
    logger.info("=" * 70)

    fi_df = load_fi_rankings()
    if fi_df.empty:
        logger.error("No feature importance data available. Run Step 3 first.")
        return

    all_rows = []

    # ── DL Models ──────────────────────────────────────────────────────
    dl_model_loaders = {
        "ITE Transformer": ("transformer", load_transformer_model),
        "LSTM-Attention": ("lstm", load_lstm_model),
        "Masked CNN": ("cnn", load_cnn_model),
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

            # Load data once per (model_type, pw, dataset)
            for dataset_name in DATASETS:
                hospital = HOSPITAL_MAP[dataset_name]
                logger.info(f"  {dataset_name}...")

                try:
                    if model_type == "cnn":
                        data, meta, labels, scale_info = load_cnn_test_data(pw, hospital)
                    else:
                        data_list, labels, scale_info = load_transformer_test_data(pw, hospital)

                    for subset_name, fraction in FEATURE_SUBSETS.items():
                        top_features = get_top_features(fi_df, model_name, pw, dataset_name, fraction)
                        logger.info(f"    {subset_name}: {len(top_features)} features")

                        if model_type == "cnn":
                            result = evaluate_cnn_subset(model, data, meta, labels, scale_info, top_features)
                        else:
                            result = evaluate_transformer_subset(model, data_list, labels, scale_info, top_features)

                        result["model_name"] = model_name
                        result["horizon"] = pw
                        result["dataset"] = dataset_name
                        result["feature_subset"] = subset_name
                        result["fraction"] = fraction
                        all_rows.append(result)

                        logger.info(f"      AUROC: {result['full_auroc']:.3f} -> {result['masked_auroc']:.3f} "
                                    f"(drop: {result['auroc_drop_pct']:.1f}%)")
                except Exception as e:
                    logger.error(f"    Error: {e}", exc_info=True)

            del model
            gc.collect()
            torch.cuda.empty_cache()

    # ── Baseline Models ────────────────────────────────────────────────
    for model_name in BASELINE_MODEL_CONFIGS:
        for pw in PREDICTION_HORIZONS:
            for dataset_name in DATASETS:
                logger.info(f"\n{model_name} PW={pw}h {dataset_name}...")

                for subset_name, fraction in FEATURE_SUBSETS.items():
                    top_features = get_top_features(fi_df, model_name, pw, dataset_name, fraction)
                    logger.info(f"  {subset_name}: {len(top_features)} features")

                    try:
                        result = evaluate_baseline_subset(model_name, pw, dataset_name, top_features)
                        result["model_name"] = model_name
                        result["horizon"] = pw
                        result["dataset"] = dataset_name
                        result["feature_subset"] = subset_name
                        result["fraction"] = fraction
                        all_rows.append(result)

                        logger.info(f"    AUROC: {result['full_auroc']:.3f} -> {result['masked_auroc']:.3f} "
                                    f"(drop: {result['auroc_drop_pct']:.1f}%)")
                    except Exception as e:
                        logger.error(f"    Error: {e}")

    if all_rows:
        result_df = pd.DataFrame(all_rows)
        out_path = RESULTS_OUT / "feature_subset_comparison.csv"
        result_df.to_csv(out_path, index=False)
        logger.info(f"\nSaved results: {out_path}")

        # Generate summary table
        summary = result_df.pivot_table(
            index=["model_name", "horizon", "dataset"],
            columns="feature_subset",
            values=["masked_auroc", "auroc_drop_pct"],
            aggfunc="first",
        )
        summary_path = RESULTS_OUT / "feature_subset_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"Saved summary: {summary_path}")

    logger.info("\nStep 5 Complete!")


# %%
if __name__ == "__main__":
    main()
