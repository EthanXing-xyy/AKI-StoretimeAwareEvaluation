# %%
"""
Step A: Time-Window-Level Metric Computation for Clinical Faithfulness Analysis

Computes AUROC, AUPRC, PPV, Sensitivity at each time window (0-72h before event)
for all model × horizon × dataset combinations using raw online simulation predictions.
Bootstrap 95% CI for AUROC and PPV.

Optimized with:
  - joblib Parallel across 45 model×horizon×dataset combinations
  - Single bootstrap loop computes ALL metrics simultaneously
  - Cached ilsan_test loading for Youden threshold
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.config import (
    SEED, PREDICTION_HORIZONS, DATASETS, TIME_POINTS_ONLINE,
    ALL_MODEL_CONFIGS, DL_MODEL_CONFIGS,
    RESULTS_DIR, HOSPITALS,
)
from utils.data_loader import load_online_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "clinical_faithfulness"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)

PPV_THRESHOLDS = [0.3, 0.4, 0.5]
N_BOOTSTRAP = 1000
WINDOW_SIZE = 6.0
N_JOBS = 8  # parallel workers


# %%
def _compute_single_bootstrap(y_true, y_prob, thresholds, rng_seed, n_bootstrap=N_BOOTSTRAP):
    """Single vectorized bootstrap: compute AUROC + PPV at all thresholds in one pass."""
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)

    # Pre-generate all bootstrap indices at once: (n_bootstrap, n)
    all_idx = rng.randint(0, n, size=(n_bootstrap, n))

    auroc_boots = []
    ppv_boots = {th: [] for th in thresholds}

    for i in range(n_bootstrap):
        idx = all_idx[i]
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        auroc_boots.append(roc_auc_score(yt, yp))
        for th in thresholds:
            pred = yp >= th
            tp = (pred & (yt == 1)).sum()
            fp = (pred & (yt == 0)).sum()
            ppv_boots[th].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)

    def _ci(arr):
        if arr:
            return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        return np.nan, np.nan

    auroc_ci = _ci(auroc_boots)
    ppv_cis = {th: _ci(ppv_boots[th]) for th in thresholds}
    return auroc_ci, ppv_cis


def compute_metrics_at_timepoint(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    time_point: float,
    youden_threshold: float,
    n_total: int,
    n_aki: int,
) -> dict:
    """Compute all metrics for one time window. Bootstrap all thresholds at once."""
    all_thresholds = PPV_THRESHOLDS + [youden_threshold]

    # Point estimates
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # Single bootstrap for AUROC + all PPV thresholds
    auroc_ci, ppv_cis = _compute_single_bootstrap(
        y_true, y_prob, all_thresholds, rng_seed=SEED
    )

    row = {
        "time_window": time_point,
        "hours_before_onset": time_point,
        "n_total": n_total,
        "n_aki": n_aki,
        "n_nonaki": n_total - n_aki,
        "auroc": auroc,
        "auroc_ci_lower": auroc_ci[0],
        "auroc_ci_upper": auroc_ci[1],
        "auprc": auprc,
    }

    # PPV + Sensitivity at fixed thresholds
    for th in PPV_THRESHOLDS:
        th_str = str(th).replace(".", "")
        pred = (y_prob >= th).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        row[f"ppv_{th_str}"] = ppv
        row[f"ppv_{th_str}_ci_lower"] = ppv_cis[th][0]
        row[f"ppv_{th_str}_ci_upper"] = ppv_cis[th][1]
        row[f"sensitivity_{th_str}"] = sens

    # PPV + Sensitivity at Youden
    pred_y = (y_prob >= youden_threshold).astype(int)
    tp_y = ((pred_y == 1) & (y_true == 1)).sum()
    fp_y = ((pred_y == 1) & (y_true == 0)).sum()
    fn_y = ((pred_y == 0) & (y_true == 1)).sum()
    row["youden_threshold"] = youden_threshold
    row["ppv_youden"] = tp_y / (tp_y + fp_y) if (tp_y + fp_y) > 0 else 0.0
    row["ppv_youden_ci_lower"] = ppv_cis[youden_threshold][0]
    row["ppv_youden_ci_upper"] = ppv_cis[youden_threshold][1]
    row["sensitivity_youden"] = tp_y / (tp_y + fn_y) if (tp_y + fn_y) > 0 else 0.0

    return row


# %%
def _apply_matched_reference(df: pd.DataFrame, matching_df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-AKI time_to_event with matched pseudo-onset reference."""
    if matching_df is None or matching_df.empty:
        return df
    pseudo_map = matching_df.set_index("visit_id")["pseudo_onset_hours"]
    non_aki_mask = df["label"] == 0
    matched_vals = df.loc[non_aki_mask, "visit_id"].map(pseudo_map)
    # time_to_event = pseudo_onset_hours - timestamp
    df.loc[non_aki_mask, "time_to_event"] = (
        matched_vals.values - df.loc[non_aki_mask, "timestamp"].values
    )
    return df


def _process_one_combination(model_name, pw, dataset_name, ds_info, model_type, youden_cache, matching_data):
    """Process a single model×horizon×dataset combination. Called in parallel."""
    rows = []
    try:
        df = load_online_predictions(model_name, pw, dataset_name)

        # Apply matched reference for non-AKI patients
        if dataset_name in matching_data:
            df = _apply_matched_reference(df, matching_data[dataset_name])

        # Get Youden threshold from single-point evaluation (ilsan)
        cache_key = (model_name, pw)
        youden_th = youden_cache.get(cache_key, 0.5)

        half = WINDOW_SIZE / 2
        for tp in TIME_POINTS_ONLINE:
            mask = (df["time_to_event"] >= tp - half) & (df["time_to_event"] < tp + half)
            subset = df[mask]
            n_total = len(subset)
            n_aki = int(subset["label"].sum()) if n_total > 0 else 0

            if n_total < 30 or len(np.unique(subset["label"])) < 2:
                continue

            result = compute_metrics_at_timepoint(
                subset["label"].values, subset["pred_proba"].values,
                tp, youden_th, n_total, n_aki,
            )
            result["model"] = model_name
            result["model_type"] = model_type
            result["horizon"] = pw
            result["dataset"] = dataset_name
            result["dataset_display"] = ds_info["display_name"]
            rows.append(result)

    except Exception as e:
        logger.error(f"Error {model_name}/{pw}h/{dataset_name}: {e}")

    return rows


# %%
def run_step_a() -> pd.DataFrame:
    """Compute time-window metrics for all combinations (parallelized)."""
    logger.info("=" * 60)
    logger.info("Step A: Time-Window Metric Computation (parallel, n_jobs=%d)", N_JOBS)
    logger.info("=" * 60)

    # Load Youden thresholds from ilsan single-point evaluation
    logger.info("Loading Youden thresholds from single-point evaluation (ilsan)...")
    sp_path = RESULTS_DIR / "baseline_comparison" / "single_point_all_models.csv"
    sp_df = pd.read_csv(sp_path)
    sp_ilsan = sp_df[sp_df["dataset"] == "ilsan_test"]
    youden_cache = {}
    for _, row in sp_ilsan.iterrows():
        key = (row["model_name"], int(row["horizon"]))
        youden_cache[key] = float(row["threshold"])
        logger.info(f"  {row['model_name']:25s} PW{int(row['horizon']):2d}h: threshold={row['threshold']:.6f}")
    logger.info(f"  Loaded {len(youden_cache)} thresholds")

    # Load matched reference time data
    logger.info("Loading matched reference time data...")
    hospital_map = {"ilsan_test": "ilsan", "cchlmc_external": "cchlmc", "mimic-iv_external": "mimic-iv"}
    matching_data = {}
    for dataset_name, hospital in hospital_map.items():
        path = RESULTS_DIR / "reference_time" / f"reference_time_matching_{hospital}.csv"
        if path.exists():
            matching_data[dataset_name] = pd.read_csv(path)
            logger.info(f"  {dataset_name}: {len(matching_data[dataset_name])} matched rows")
        else:
            logger.warning(f"  {dataset_name}: matching file not found at {path}")

    # Build job list
    jobs = []
    for model_name, config in ALL_MODEL_CONFIGS.items():
        model_type = "DL" if model_name in DL_MODEL_CONFIGS else "Baseline"
        for pw in PREDICTION_HORIZONS:
            for dataset_name, ds_info in DATASETS.items():
                jobs.append((model_name, pw, dataset_name, ds_info, model_type))

    logger.info(f"Running {len(jobs)} combinations with {N_JOBS} workers (matched reference)...")

    # Parallel execution
    results_nested = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_process_one_combination)(mn, pw, dn, di, mt, youden_cache, matching_data)
        for mn, pw, dn, di, mt in jobs
    )

    all_rows = [row for sublist in results_nested for row in sublist]
    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_OUT / "step_a_timewindow_metrics.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(result_df)} rows)")
    return result_df


# %%
if __name__ == "__main__":
    df = run_step_a()
    print(f"\nStep A complete: {len(df)} rows")
    print(df.groupby(["model", "horizon", "dataset"]).size().reset_index(name="n_timepoints"))
