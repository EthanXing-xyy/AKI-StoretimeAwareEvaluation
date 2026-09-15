# %%
"""
Step 4: Permutation Feature Importance (Online Simulation)

TRUE per-feature permutation FI at each time window (-72h ~ 0h).
For each time window:
  1. Load raw patient data, truncate to that time point
  2. Permute each feature across patients
  3. Re-run model inference
  4. Measure AUROC drop

Uses Step 1 matched reference time for non-AKI patients.
2-phase approach for speed: Phase 1 (1 repeat all features) → Phase 2 (5 repeats top-15).
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gc
import logging
import pickle
import time
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils.config import (
    SEED, PROJECT_ROOT, PREDICTION_HORIZONS, DATASETS, DATASET_ORDER,
    DL_MODEL_CONFIGS, BASELINE_MODEL_CONFIGS, ALL_MODEL_CONFIGS,
    ALL_TS_FEATURES, DEMO_COL, VITAL_COL, LAB_COL,
    RESULTS_DIR, FIGURES_DIR, PROCESSED_DIR,
    MODEL_DISPLAY_NAMES, MODEL_COLORS, MODEL_ORDER,
    DATASET_DISPLAY, DATASET_PANEL_LABELS,
    FIGURE_DPI_EXPORT, model_display, feature_display,
    get_checkpoint_path,
)
from utils.data_loader import load_online_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_OUT = RESULTS_DIR / "feature_importance"
FIGURES_OUT = FIGURES_DIR / "feature_importance"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

TIME_WINDOWS = [0, 6, 12, 24, 48, 72]
WINDOW_HALF = 3.0
N_REPEATS = 5
TOP_N_PHASE2 = 15
MAX_SAMPLES = 2000
BATCH_SIZE = 512
HOSPITAL_MAP = {"ilsan_test": "ilsan", "cchlmc_external": "cchlmc", "mimic-iv_external": "mimic-iv"}

FEATURE_NAMES_ALL = VITAL_COL + LAB_COL + DEMO_COL  # 51
FEATURE_NAMES_TS = VITAL_COL + LAB_COL  # 48

DEMO_COL_ENC = ["age", "sex", "bmi"]
FEATURE_ENCODING = {
    **{col: i for i, col in enumerate(DEMO_COL_ENC)},
    **{col: i + len(DEMO_COL_ENC) for i, col in enumerate(VITAL_COL)},
    **{col: i + len(DEMO_COL_ENC) + len(VITAL_COL) for i, col in enumerate(LAB_COL)},
}


# ── Model loading (reuse from step3) ────────────────────────────────

from experiments.step3_feature_importance_single import (
    load_cnn_model, load_transformer_model, load_lstm_model,
    infer_cnn_batch, collate_transformer_batch, infer_transformer_batch,
)


# ── Data loading: raw patient data with time truncation ──────────────

def load_cnn_patients_for_online(pw, hospital, online_df, time_window):
    """Load raw CNN data for patients at a specific time window.

    For each patient, truncate time series to the cumulative data
    available at (event_time - time_window) hours from admission.
    """
    data_dir = PROCESSED_DIR / hospital / "model" / f"prediction_window_{pw}"
    ts_dir = data_dir / "timeseries"
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

    # Filter online predictions to this time window
    tw_mask = (online_df["time_to_event"] >= time_window - WINDOW_HALF) & \
              (online_df["time_to_event"] < time_window + WINDOW_HALF)
    tw_df = online_df[tw_mask].copy()

    if len(tw_df) == 0:
        return None, None, None, None, 0

    # Get unique patients in this window — pick one prediction per patient (closest to time_window)
    tw_df["tw_dist"] = (tw_df["time_to_event"] - time_window).abs()
    tw_df = tw_df.sort_values("tw_dist").drop_duplicates(subset="visit_id", keep="first")

    # Sample if too many
    if len(tw_df) > MAX_SAMPLES:
        aki = tw_df[tw_df["label"] == 1]
        non_aki = tw_df[tw_df["label"] == 0]
        n_aki = max(1, int(MAX_SAMPLES * len(aki) / len(tw_df)))
        n_non = MAX_SAMPLES - n_aki
        tw_df = pd.concat([
            aki.sample(min(n_aki, len(aki)), random_state=SEED),
            non_aki.sample(min(n_non, len(non_aki)), random_state=SEED),
        ])

    # Load raw data for each patient, truncated to their timestamp
    all_data, all_meta, all_labels = [], [], []
    for _, row in tw_df.iterrows():
        vid = row["visit_id"]
        fpath = ts_dir / f"{vid}.gz"
        if not fpath.exists():
            continue
        d = joblib.load(fpath)
        ts_data = d["data"]  # (seq_len, 48)

        # Truncate: use data up to timestamp hours
        trunc_len = max(1, int(row["timestamp"]) + 1)
        if trunc_len < ts_data.shape[0]:
            ts_data = ts_data[:trunc_len]

        # Pad/truncate to 256
        if ts_data.shape[0] < 256:
            pad = np.full((256 - ts_data.shape[0], 48), np.nan)
            ts_data = np.concatenate([pad, ts_data], axis=0)
        elif ts_data.shape[0] > 256:
            ts_data = ts_data[-256:]

        # Get demographics from master
        master_row = master[master["visit_id"] == vid]
        if len(master_row) == 0:
            continue
        mr = master_row.iloc[0]
        sex_val = 0 if mr.get("sex", "M") == "M" else 1
        meta = np.array([mr["age"], sex_val, mr["bmi"]], dtype=np.float64)

        all_data.append(ts_data)
        all_meta.append(meta)
        all_labels.append(int(row["label"]))

    if len(all_data) < 30:
        return None, None, None, None, len(all_data)

    return np.array(all_data), np.array(all_meta), np.array(all_labels), scale_info, len(all_data)


def load_transformer_patients_for_online(pw, hospital, online_df, time_window):
    """Load Transformer/LSTM data truncated to time window."""
    data_dir = PROCESSED_DIR / hospital / "transformer" / f"prediction_window_{pw}"
    ts_dir = data_dir / "timeseries"
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

    tw_mask = (online_df["time_to_event"] >= time_window - WINDOW_HALF) & \
              (online_df["time_to_event"] < time_window + WINDOW_HALF)
    tw_df = online_df[tw_mask].copy()
    if len(tw_df) == 0:
        return None, None, None, 0

    tw_df["tw_dist"] = (tw_df["time_to_event"] - time_window).abs()
    tw_df = tw_df.sort_values("tw_dist").drop_duplicates(subset="visit_id", keep="first")

    if len(tw_df) > MAX_SAMPLES:
        aki = tw_df[tw_df["label"] == 1]
        non_aki = tw_df[tw_df["label"] == 0]
        n_aki = max(1, int(MAX_SAMPLES * len(aki) / len(tw_df)))
        n_non = MAX_SAMPLES - n_aki
        tw_df = pd.concat([
            aki.sample(min(n_aki, len(aki)), random_state=SEED),
            non_aki.sample(min(n_non, len(non_aki)), random_state=SEED),
        ])

    all_data, all_labels = [], []
    for _, row in tw_df.iterrows():
        vid = row["visit_id"]
        fpath = ts_dir / f"{vid}.gz"
        if not fpath.exists():
            continue
        d = joblib.load(fpath)
        events = d["data"]  # (n_events, 3): [feature_id, time_hours, value]

        # Truncate events to timestamp
        trunc_time = row["timestamp"]
        events = events[events[:, 1] <= trunc_time]

        if len(events) == 0:
            continue

        all_data.append(events)
        all_labels.append(int(row["label"]))

    if len(all_data) < 30:
        return None, None, None, len(all_data)

    return all_data, np.array(all_labels), scale_info, len(all_data)


# ── Permutation FI at a single time window ───────────────────────────

def permutation_fi_cnn_window(model, data, meta, labels, scale_info, rng):
    """2-phase permutation FI for CNN at one time window."""
    n = len(data)

    # Baseline
    all_p = []
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        all_p.append(infer_cnn_batch(model, data[s:e], meta[s:e], scale_info))
    baseline_probs = np.concatenate(all_p)
    baseline_auroc = roc_auc_score(labels, baseline_probs)

    # Phase 1: 1 repeat per feature
    p1 = {}
    for fi, fn in enumerate(FEATURE_NAMES_TS):
        pd_ = data.copy()
        pd_[:, :, fi] = data[rng.permutation(n), :, fi]
        pp = []
        for s in range(0, n, BATCH_SIZE):
            e = min(s + BATCH_SIZE, n)
            pp.append(infer_cnn_batch(model, pd_[s:e], meta[s:e], scale_info))
        p1[fn] = baseline_auroc - roc_auc_score(labels, np.concatenate(pp))

    for di, dn in enumerate(DEMO_COL):
        pm = meta.copy()
        pm[:, di] = meta[rng.permutation(n), di]
        pp = []
        for s in range(0, n, BATCH_SIZE):
            e = min(s + BATCH_SIZE, n)
            pp.append(infer_cnn_batch(model, data[s:e], pm[s:e], scale_info))
        p1[dn] = baseline_auroc - roc_auc_score(labels, np.concatenate(pp))

    # Phase 2: top-k with N repeats
    top_feats = set(f for f, _ in sorted(p1.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N_PHASE2])
    results = []
    for fn in FEATURE_NAMES_ALL:
        if fn in top_feats:
            imps = [p1[fn]]
            for _ in range(N_REPEATS - 1):
                if fn in FEATURE_NAMES_TS:
                    fi = FEATURE_NAMES_TS.index(fn)
                    pd_ = data.copy()
                    pd_[:, :, fi] = data[rng.permutation(n), :, fi]
                    pp = []
                    for s in range(0, n, BATCH_SIZE):
                        e = min(s + BATCH_SIZE, n)
                        pp.append(infer_cnn_batch(model, pd_[s:e], meta[s:e], scale_info))
                else:
                    di = DEMO_COL.index(fn)
                    pm = meta.copy()
                    pm[:, di] = meta[rng.permutation(n), di]
                    pp = []
                    for s in range(0, n, BATCH_SIZE):
                        e = min(s + BATCH_SIZE, n)
                        pp.append(infer_cnn_batch(model, data[s:e], pm[s:e], scale_info))
                imps.append(baseline_auroc - roc_auc_score(labels, np.concatenate(pp)))
        else:
            imps = [p1[fn]]
        results.append({"feature_name": fn, "mean_importance": np.mean(imps),
                        "std_importance": np.std(imps) if len(imps) > 1 else 0.0,
                        "baseline_auroc": baseline_auroc})
    return results


def permutation_fi_transformer_window(model, data_list, labels, scale_info, rng):
    """2-phase permutation FI for Transformer/LSTM at one time window."""
    n = len(data_list)
    id_to_name = {v: k for k, v in FEATURE_ENCODING.items()}

    # Baseline
    all_p = []
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        all_p.append(infer_transformer_batch(model, data_list[s:e], scale_info))
    baseline_probs = np.concatenate(all_p)
    baseline_auroc = roc_auc_score(labels, baseline_probs)

    def _permute_feat(feat_id):
        pidx = rng.permutation(n)
        pdata = []
        for i in range(n):
            orig = data_list[i].copy()
            donor = data_list[pidx[i]]
            omask = orig[:, 0].astype(int) == feat_id
            devents = donor[donor[:, 0].astype(int) == feat_id]
            if omask.any() and len(devents) > 0:
                other = orig[~omask]
                combined = np.concatenate([other, devents], axis=0)
                combined = combined[combined[:, 1].argsort()]
                pdata.append(combined)
            elif omask.any():
                pdata.append(orig[~omask] if (~omask).any() else orig)
            else:
                pdata.append(orig)
            if len(pdata[-1]) == 0:
                pdata[-1] = orig
        pp = []
        for s in range(0, n, BATCH_SIZE):
            e = min(s + BATCH_SIZE, n)
            pp.append(infer_transformer_batch(model, pdata[s:e], scale_info))
        return roc_auc_score(labels, np.concatenate(pp))

    # Phase 1
    p1 = {}
    for fid in range(51):
        fn = id_to_name[fid]
        p1[fn] = baseline_auroc - _permute_feat(fid)

    # Phase 2
    top_feats = set(f for f, _ in sorted(p1.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N_PHASE2])
    results = []
    for fn, fid in sorted(FEATURE_ENCODING.items(), key=lambda x: x[1]):
        if fn in top_feats:
            imps = [p1[fn]]
            for _ in range(N_REPEATS - 1):
                imps.append(baseline_auroc - _permute_feat(fid))
        else:
            imps = [p1[fn]]
        results.append({"feature_name": fn, "mean_importance": np.mean(imps),
                        "std_importance": np.std(imps) if len(imps) > 1 else 0.0,
                        "baseline_auroc": baseline_auroc})
    return results


# ── Baseline online FI (XGBoost, LR) ────────────────────────────────

def _run_baseline_online_fi(model_name, pw, hospital, online_df, rng):
    """Per-feature permutation FI for baseline models at each time window.

    Loads raw CNN data, builds flat feature vector (mean aggregation of
    time series + demographics), then permutes and re-predicts with sklearn.
    """
    config = BASELINE_MODEL_CONFIGS[model_name]
    ckpt_path = get_checkpoint_path(config, pw)
    bl_model = joblib.load(ckpt_path)

    scale_info = pd.read_parquet(
        PROCESSED_DIR / "scaling_info" / "ilsan" / "scaling_info.parquet",
        engine="fastparquet",
    )

    all_tw_results = []
    feature_names = FEATURE_NAMES_ALL  # 51

    for tw in TIME_WINDOWS:
        logger.info(f"  Time window: {tw}h")

        # Reuse CNN data loader to get truncated patient data
        data, meta, labels, si, n = load_cnn_patients_for_online(
            pw, hospital, online_df, tw)
        if data is None:
            logger.warning(f"    Skipped (N={n} < 30)")
            continue
        logger.info(f"    N={n}, AKI={labels.sum()}")

        # Build flat features: mean of each TS feature + demographics
        ts_means = np.nanmean(data, axis=1)  # (N, 48) — mean over time
        ts_means = np.nan_to_num(ts_means, nan=0.0)
        # Normalize with scaling info
        feat_col = VITAL_COL + LAB_COL
        ts_scale = si.query("feature in @feat_col").set_index("feature").loc[feat_col]
        ts_means = (ts_means - ts_scale["median"].values) / ts_scale["iqr"].values
        ts_means = np.nan_to_num(ts_means, nan=0.0)

        # Normalize demographics
        demo_scale = si.query("feature in ['age','bmi']").set_index("feature").loc[["age", "bmi"]]
        meta_norm = meta.copy().astype(np.float32)
        meta_norm[:, 0] = (meta_norm[:, 0] - demo_scale.loc["age", "median"]) / demo_scale.loc["age", "iqr"]
        meta_norm[:, 2] = (meta_norm[:, 2] - demo_scale.loc["bmi", "median"]) / demo_scale.loc["bmi", "iqr"]

        X = np.concatenate([ts_means, meta_norm], axis=1)  # (N, 51)

        # Baseline prediction
        if hasattr(bl_model, "predict_proba"):
            baseline_probs = bl_model.predict_proba(X)[:, 1]
        else:
            baseline_probs = bl_model.predict(X)
        baseline_auroc = roc_auc_score(labels, baseline_probs)

        # Phase 1: 1 repeat per feature
        p1 = {}
        for fi, fn in enumerate(feature_names):
            Xp = X.copy()
            Xp[:, fi] = X[rng.permutation(n), fi]
            if hasattr(bl_model, "predict_proba"):
                pp = bl_model.predict_proba(Xp)[:, 1]
            else:
                pp = bl_model.predict(Xp)
            p1[fn] = baseline_auroc - roc_auc_score(labels, pp)

        # Phase 2: top-k with N repeats
        top_feats = set(f for f, _ in sorted(p1.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N_PHASE2])
        for fn in feature_names:
            fi = feature_names.index(fn)
            if fn in top_feats:
                imps = [p1[fn]]
                for _ in range(N_REPEATS - 1):
                    Xp = X.copy()
                    Xp[:, fi] = X[rng.permutation(n), fi]
                    if hasattr(bl_model, "predict_proba"):
                        pp = bl_model.predict_proba(Xp)[:, 1]
                    else:
                        pp = bl_model.predict(Xp)
                    imps.append(baseline_auroc - roc_auc_score(labels, pp))
            else:
                imps = [p1[fn]]

            all_tw_results.append({
                "feature_name": fn,
                "mean_importance": np.mean(imps),
                "std_importance": np.std(imps) if len(imps) > 1 else 0.0,
                "baseline_auroc": baseline_auroc,
                "time_window": tw,
                "n_samples": n,
            })

    return all_tw_results


# ── Load online predictions with matched reference ───────────────────

def load_online_with_matched(model_name, pw, dataset_name):
    """Load online predictions and apply matched reference time for non-AKI."""
    from utils.metrics import decode_pred_proba, sigmoid
    df = load_online_predictions(model_name, pw, dataset_name)
    if df["pred_proba"].dtype == object:
        df["pred_proba"] = decode_pred_proba(df["pred_proba"])
    if df["pred_proba"].max() > 1.0 or df["pred_proba"].min() < 0.0:
        df["pred_proba"] = df["pred_proba"].apply(sigmoid)

    hospital = HOSPITAL_MAP[dataset_name]
    match_path = RESULTS_DIR / "reference_time" / f"reference_time_matching_{hospital}.csv"
    if match_path.exists():
        mdf = pd.read_csv(match_path)
        pseudo_map = mdf.set_index("visit_id")["pseudo_onset_hours"]
        non_aki = df["label"] == 0
        matched = df.loc[non_aki, "visit_id"].map(pseudo_map)
        df.loc[non_aki, "time_to_event"] = matched.values - df.loc[non_aki, "timestamp"].values

    return df


# ── Visualization ────────────────────────────────────────────────────

def plot_temporal_fi_trajectory(results_df, model_short, pw, dataset_name, top_n=10):
    """Line plot: top features' importance over time windows."""
    mask = (results_df["model"] == model_short) & \
           (results_df["horizon"] == pw) & \
           (results_df["dataset"] == dataset_name) & \
           (results_df["patient_group"] == "all")
    df = results_df[mask]
    if len(df) == 0:
        return

    # Get top features by average importance
    avg_imp = df.groupby("feature_name")["mean_importance"].mean().sort_values(ascending=False)
    top_features = avg_imp.head(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = cm.get_cmap("tab10")
    for i, feat in enumerate(top_features):
        fd = df[df["feature_name"] == feat].sort_values("time_window")
        ax.plot(fd["time_window"], fd["mean_importance"], marker="o", markersize=4,
                color=cmap(i), label=feature_display(feat), linewidth=1.5)

    ax.set_xlim([74, -2])
    ax.set_xlabel("Hours before event", fontsize=11)
    ax.set_ylabel("Feature Importance (AUROC drop)", fontsize=11)
    # Title removed per request
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()

    safe_ds = dataset_name.replace("-", "_")
    fig.savefig(FIGURES_OUT / f"online_fi_trajectory_{model_short}_pw{pw}_{safe_ds}.pdf",
                dpi=FIGURE_DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)


def plot_fi_heatmap(results_df, model_short, pw, dataset_name, top_n=20):
    """Heatmap: features × time windows."""
    mask = (results_df["model"] == model_short) & \
           (results_df["horizon"] == pw) & \
           (results_df["dataset"] == dataset_name) & \
           (results_df["patient_group"] == "all")
    df = results_df[mask]
    if len(df) == 0:
        return

    pivot = df.pivot_table(index="feature_name", columns="time_window", values="mean_importance")
    avg_imp = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[avg_imp.head(top_n).index]
    display_labels = [feature_display(f) for f in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-pivot.values.max(), vmax=pivot.values.max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)}h" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(display_labels, fontsize=8)
    ax.set_xlabel("Hours before event", fontsize=11)
    # Title removed per request
    plt.colorbar(im, ax=ax, label="Importance (AUROC drop)")
    plt.tight_layout()

    safe_ds = dataset_name.replace("-", "_")
    fig.savefig(FIGURES_OUT / f"online_fi_heatmap_{model_short}_pw{pw}_{safe_ds}.pdf",
                dpi=FIGURE_DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Step 4: Online Permutation Feature Importance (per-feature)")
    logger.info(f"Device: {DEVICE}, MAX_SAMPLES={MAX_SAMPLES}")
    logger.info("=" * 70)

    rng = np.random.RandomState(SEED)
    all_results = []
    checkpoint_path = RESULTS_OUT / "step4_online_fi_checkpoint.pkl"

    # Resume from checkpoint if exists
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            saved = pickle.load(f)
            all_results = saved.get("results", [])
            completed = saved.get("completed", set())
            logger.info(f"Resumed from checkpoint: {len(completed)} combos done")

    dl_loaders = {
        "Masked CNN": ("cnn", load_cnn_model),
        "ITE Transformer": ("transformer", load_transformer_model),
        "LSTM-Attention": ("lstm", load_lstm_model),
    }

    for model_name, (model_type, loader_fn) in dl_loaders.items():
        model_short = model_display(model_name)
        for pw in PREDICTION_HORIZONS:
            for dataset_name in DATASET_ORDER:
                combo_key = f"{model_name}|{pw}|{dataset_name}"
                if combo_key in completed:
                    logger.info(f"SKIP {combo_key}")
                    continue

                hospital = HOSPITAL_MAP[dataset_name]
                logger.info(f"\n{'='*50}")
                logger.info(f"{model_short} PW={pw}h {dataset_name}")
                t0 = time.time()

                try:
                    model = loader_fn(pw)
                    online_df = load_online_with_matched(model_name, pw, dataset_name)
                    logger.info(f"  Online predictions: {len(online_df)} rows")

                    for tw in TIME_WINDOWS:
                        logger.info(f"  Time window: {tw}h before event")

                        if model_type == "cnn":
                            data, meta, labels, si, n = load_cnn_patients_for_online(
                                pw, hospital, online_df, tw)
                            if data is None:
                                logger.warning(f"    Skipped (N={n} < 30)")
                                continue
                            logger.info(f"    N={n}, AKI={labels.sum()}")
                            fi_results = permutation_fi_cnn_window(model, data, meta, labels, si, rng)
                        else:
                            data_list, labels, si, n = load_transformer_patients_for_online(
                                pw, hospital, online_df, tw)
                            if data_list is None:
                                logger.warning(f"    Skipped (N={n} < 30)")
                                continue
                            logger.info(f"    N={n}, AKI={labels.sum()}")
                            fi_results = permutation_fi_transformer_window(model, data_list, labels, si, rng)

                        for r in fi_results:
                            r.update({
                                "model": model_short,
                                "model_internal": model_name,
                                "horizon": pw,
                                "dataset": dataset_name,
                                "time_window": tw,
                                "patient_group": "all",
                                "n_samples": n,
                            })
                        all_results.extend(fi_results)

                    completed.add(combo_key)
                    logger.info(f"  Done in {time.time()-t0:.0f}s")

                    # Save checkpoint
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump({"results": all_results, "completed": completed}, f)

                except Exception as e:
                    logger.error(f"  ERROR: {e}", exc_info=True)

                finally:
                    if "model" in dir():
                        del model
                    gc.collect()
                    torch.cuda.empty_cache()

    # ── Baseline models (XGBoost, LR) ──────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Baseline models: XGBoost, Logistic Regression")
    logger.info("=" * 50)

    for model_name in BASELINE_MODEL_CONFIGS:
        model_short = model_display(model_name)
        for pw in PREDICTION_HORIZONS:
            for dataset_name in DATASET_ORDER:
                combo_key = f"{model_name}|{pw}|{dataset_name}"
                if combo_key in completed:
                    logger.info(f"SKIP {combo_key}")
                    continue

                hospital = HOSPITAL_MAP[dataset_name]
                logger.info(f"\n{model_short} PW={pw}h {dataset_name}")
                t0 = time.time()

                try:
                    online_df = load_online_with_matched(model_name, pw, dataset_name)
                    fi_results_all_tw = _run_baseline_online_fi(
                        model_name, pw, hospital, online_df, rng)

                    for r in fi_results_all_tw:
                        r.update({
                            "model": model_short,
                            "model_internal": model_name,
                            "horizon": pw,
                            "dataset": dataset_name,
                            "patient_group": "all",
                        })
                    all_results.extend(fi_results_all_tw)

                    completed.add(combo_key)
                    logger.info(f"  Done in {time.time()-t0:.0f}s")

                    with open(checkpoint_path, "wb") as f:
                        pickle.dump({"results": all_results, "completed": completed}, f)

                except Exception as e:
                    logger.error(f"  ERROR: {e}", exc_info=True)

    # Save final results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_OUT / "online_fi_per_feature.csv", index=False)
        logger.info(f"\nSaved: online_fi_per_feature.csv ({len(df)} rows)")

        # Generate figures
        logger.info("Generating figures...")
        for model_short in MODEL_ORDER:
            for pw in PREDICTION_HORIZONS:
                for ds in DATASET_ORDER:
                    plot_temporal_fi_trajectory(df, model_short, pw, ds)
                    plot_fi_heatmap(df, model_short, pw, ds)
        logger.info("Figures done!")

    logger.info("\nStep 4 Complete!")


if __name__ == "__main__":
    main()
