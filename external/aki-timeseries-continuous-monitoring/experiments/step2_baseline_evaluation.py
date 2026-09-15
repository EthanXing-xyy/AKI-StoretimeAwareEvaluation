# %%
"""
Step 2: Baseline Model Evaluation (XGBoost, Logistic Regression)

Comprehensive evaluation of baseline models matching DL model evaluation:
- 2-A: Single-point evaluation (AUROC, AUPRC, Sensitivity, Specificity, F1 with 95% CI)
- 2-B: Online simulation (AUROC trajectory over time, with matched reference time)
- 2-C: Calibration (Brier score before/after isotonic recalibration)
- 2-D: Feature importance (XGBoost gain + permutation, LR coefficients + permutation)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import spearmanr

from utils.config import (
    SEED, PROJECT_ROOT, PREDICTION_HORIZONS, DATASETS, TIME_POINTS_ONLINE,
    DL_MODEL_CONFIGS, BASELINE_MODEL_CONFIGS, ALL_MODEL_CONFIGS,
    ALL_TS_FEATURES, DEMO_COL, ALL_FEATURES,
    RESULTS_DIR, FIGURES_DIR, get_checkpoint_path,
    DATASET_ORDER, DATASET_DISPLAY,
)
from utils.metrics import (
    calculate_youden_threshold, compute_binary_metrics,
    bootstrap_metrics_with_ci, isotonic_recalibration_cv, compute_brier_scores,
)
from utils.data_loader import (
    load_predictions, load_online_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "baseline_comparison"
FIGURES_OUT = FIGURES_DIR / "baseline_comparison"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)


# %%
# =============================================================================
# 2-A: Single-Point Evaluation
# =============================================================================

def run_single_point_evaluation() -> pd.DataFrame:
    """Evaluate all models (DL + baseline) on single-point predictions."""
    logger.info("=" * 60)
    logger.info("2-A: Single-Point Evaluation")
    logger.info("=" * 60)

    all_rows = []

    for model_name, config in ALL_MODEL_CONFIGS.items():
        model_type = "DL" if model_name in DL_MODEL_CONFIGS else "Baseline"

        for pw in PREDICTION_HORIZONS:
            for dataset_name, ds_info in DATASETS.items():
                try:
                    df = load_predictions(model_name, pw, dataset_name)
                    y_true = df["label"].values
                    y_prob = df["pred_proba"].values

                    if len(np.unique(y_true)) < 2:
                        logger.warning(f"Skipping {model_name}/{pw}h/{dataset_name}: single class")
                        continue

                    # Use Youden threshold from internal dev set
                    if dataset_name == "ilsan_test":
                        threshold = calculate_youden_threshold(y_true, y_prob)
                    else:
                        # Load internal test set to get threshold
                        df_dev = load_predictions(model_name, pw, "ilsan_test")
                        threshold = calculate_youden_threshold(
                            df_dev["label"].values, df_dev["pred_proba"].values
                        )

                    metrics = bootstrap_metrics_with_ci(y_true, y_prob, threshold, n_bootstrap=1000, seed=SEED)
                    metrics["model_type"] = model_type
                    metrics["model_name"] = model_name
                    metrics["horizon"] = pw
                    metrics["dataset"] = dataset_name
                    metrics["dataset_display"] = ds_info["display_name"]
                    all_rows.append(metrics)

                    logger.info(f"{model_name} | PW={pw}h | {dataset_name}: "
                                f"AUROC={metrics['auroc']:.3f} "
                                f"({metrics['auroc_ci_lower']:.3f}-{metrics['auroc_ci_upper']:.3f})")

                except Exception as e:
                    logger.warning(f"Error loading {model_name}/{pw}h/{dataset_name}: {e}")

    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_OUT / "single_point_all_models.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved single-point results: {out_path}")
    return result_df


# %%
# =============================================================================
# 2-B: Online Simulation
# =============================================================================

def compute_auroc_at_timepoint(df: pd.DataFrame, time_point: float, window_size: float = 6.0) -> dict:
    """Compute AUROC at a specific time-to-event window."""
    half_window = window_size / 2
    mask = (df["time_to_event"] >= time_point - half_window) & (df["time_to_event"] < time_point + half_window)
    subset = df[mask]

    if len(subset) < 30 or len(np.unique(subset["label"])) < 2:
        return {"time_point": time_point, "auroc": np.nan, "n": len(subset), "n_events": int(subset["label"].sum())}

    auroc = roc_auc_score(subset["label"], subset["pred_proba"])
    auprc = average_precision_score(subset["label"], subset["pred_proba"])
    return {
        "time_point": time_point,
        "auroc": auroc,
        "auprc": auprc,
        "n": len(subset),
        "n_events": int(subset["label"].sum()),
    }


def run_online_simulation_evaluation() -> pd.DataFrame:
    """Evaluate all models on online simulation with AUROC trajectory."""
    logger.info("=" * 60)
    logger.info("2-B: Online Simulation Evaluation")
    logger.info("=" * 60)

    all_rows = []

    for model_name, config in ALL_MODEL_CONFIGS.items():
        model_type = "DL" if model_name in DL_MODEL_CONFIGS else "Baseline"

        for pw in PREDICTION_HORIZONS:
            for dataset_name, ds_info in DATASETS.items():
                try:
                    df = load_online_predictions(model_name, pw, dataset_name)

                    for tp in TIME_POINTS_ONLINE:
                        result = compute_auroc_at_timepoint(df, tp)
                        result["model_type"] = model_type
                        result["model_name"] = model_name
                        result["horizon"] = pw
                        result["dataset"] = dataset_name
                        result["dataset_display"] = ds_info["display_name"]
                        result["reference_type"] = "discharge"
                        all_rows.append(result)

                except Exception as e:
                    logger.warning(f"Error: {model_name}/{pw}h/{dataset_name}: {e}")

    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_OUT / "online_simulation_all_models.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved online simulation results: {out_path}")
    return result_df


def run_online_simulation_matched() -> pd.DataFrame:
    """Run online simulation with matched reference times from Step 1."""
    logger.info("=" * 60)
    logger.info("2-B (matched): Online Simulation with Matched Reference Time")
    logger.info("=" * 60)

    # Load matching data
    hospital_map = {"ilsan_test": "ilsan", "cchlmc_external": "cchlmc", "mimic-iv_external": "mimic-iv"}
    matching_data = {}
    for dataset_name, hospital in hospital_map.items():
        path = RESULTS_DIR / "reference_time" / f"reference_time_matching_{hospital}.csv"
        if path.exists():
            matching_data[dataset_name] = pd.read_csv(path)
        else:
            logger.warning(f"Matching file not found: {path}")

    all_rows = []

    for model_name, config in ALL_MODEL_CONFIGS.items():
        model_type = "DL" if model_name in DL_MODEL_CONFIGS else "Baseline"

        for pw in PREDICTION_HORIZONS:
            for dataset_name, ds_info in DATASETS.items():
                try:
                    df = load_online_predictions(model_name, pw, dataset_name)

                    # Apply matched reference time for non-AKI
                    if dataset_name in matching_data:
                        match_df = matching_data[dataset_name]
                        pseudo_map = match_df.set_index("visit_id")["pseudo_onset_hours"]
                        non_aki_mask = df["label"] == 0
                        matched_vals = df.loc[non_aki_mask, "visit_id"].map(pseudo_map)

                        # time_to_event = pseudo_onset_hours - timestamp
                        df.loc[non_aki_mask, "time_to_event"] = (
                            matched_vals.values - df.loc[non_aki_mask, "timestamp"].values
                        )

                    for tp in TIME_POINTS_ONLINE:
                        result = compute_auroc_at_timepoint(df, tp)
                        result["model_type"] = model_type
                        result["model_name"] = model_name
                        result["horizon"] = pw
                        result["dataset"] = dataset_name
                        result["dataset_display"] = ds_info["display_name"]
                        result["reference_type"] = "matched"
                        all_rows.append(result)

                except Exception as e:
                    logger.warning(f"Error: {model_name}/{pw}h/{dataset_name}: {e}")

    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_OUT / "online_simulation_matched_all_models.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved matched online simulation: {out_path}")
    return result_df


# %%
# =============================================================================
# 2-C: Calibration
# =============================================================================

def run_calibration_analysis() -> pd.DataFrame:
    """Compute calibration metrics for all models."""
    logger.info("=" * 60)
    logger.info("2-C: Calibration Analysis")
    logger.info("=" * 60)

    all_rows = []

    for model_name, config in ALL_MODEL_CONFIGS.items():
        model_type = "DL" if model_name in DL_MODEL_CONFIGS else "Baseline"

        for pw in PREDICTION_HORIZONS:
            for dataset_name, ds_info in DATASETS.items():
                try:
                    df = load_predictions(model_name, pw, dataset_name)
                    y_true = df["label"].values
                    y_prob = df["pred_proba"].values

                    brier = compute_brier_scores(y_true, y_prob, seed=SEED)
                    brier["model_type"] = model_type
                    brier["model_name"] = model_name
                    brier["horizon"] = pw
                    brier["dataset"] = dataset_name
                    brier["dataset_display"] = ds_info["display_name"]
                    all_rows.append(brier)

                    logger.info(f"{model_name} | PW={pw}h | {dataset_name}: "
                                f"Brier={brier['brier_before']:.4f} -> {brier['brier_after']:.4f} "
                                f"({brier['improvement_pct']:.1f}% improvement)")

                except Exception as e:
                    logger.warning(f"Error: {model_name}/{pw}h/{dataset_name}: {e}")

    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_OUT / "calibration_all_models.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved calibration results: {out_path}")
    return result_df


def plot_calibration_curves():
    """Generate calibration curve comparison plots."""
    logger.info("Generating calibration curve figures...")

    for pw in PREDICTION_HORIZONS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = {"ITE Transformer": "#2196F3", "LSTM-Attention": "#4CAF50", "Masked CNN": "#FF9800",
                  "XGBoost": "#9C27B0", "Logistic Regression": "#F44336"}
        linestyles = {"ITE Transformer": "-", "LSTM-Attention": "--", "Masked CNN": "-.",
                      "XGBoost": ":", "Logistic Regression": (0, (3, 1, 1, 1))}

        for ax_idx, (title, do_recal) in enumerate([("Before Recalibration", False), ("After Isotonic Recalibration", True)]):
            ax = axes[ax_idx]

            for model_name in ALL_MODEL_CONFIGS:
                try:
                    df = load_predictions(model_name, pw, "ilsan_test")
                    y_true = df["label"].values
                    y_prob = df["pred_proba"].values

                    if do_recal:
                        y_prob = isotonic_recalibration_cv(y_true, y_prob, seed=SEED)

                    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
                    ax.plot(mean_pred, fraction_pos, label=model_name,
                            color=colors.get(model_name, "gray"),
                            linestyle=linestyles.get(model_name, "-"), linewidth=1.5)
                except Exception:
                    pass

            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
            ax.set_xlabel("Mean Predicted Probability", fontsize=11)
            ax.set_ylabel("Fraction of Positives", fontsize=11)
            ax.set_title(f"{title} (PH={pw}h)", fontsize=12)
            ax.legend(fontsize=8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        plt.tight_layout()
        fig_path = FIGURES_OUT / f"calibration_comparison_pw{pw}.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved calibration figure: {fig_path}")


# %%
# =============================================================================
# 2-D: Feature Importance (Baseline Models)
# =============================================================================

def compute_baseline_native_importance() -> pd.DataFrame:
    """Extract native feature importance from XGBoost (gain) and LR (coefficients)."""
    logger.info("=" * 60)
    logger.info("2-D: Baseline Native Feature Importance")
    logger.info("=" * 60)

    # Feature names: 48 time-series + 3 demographics (matching train_ml order)
    feature_names = ALL_TS_FEATURES + DEMO_COL

    all_rows = []

    for pw in PREDICTION_HORIZONS:
        # XGBoost: gain-based importance
        try:
            ckpt_path = get_checkpoint_path(BASELINE_MODEL_CONFIGS["XGBoost"], pw)
            model = joblib.load(ckpt_path)
            importances = model.feature_importances_  # gain-based by default
            for i, (fname, imp) in enumerate(zip(feature_names, importances)):
                all_rows.append({
                    "model_name": "XGBoost",
                    "horizon": pw,
                    "feature_name": fname,
                    "feature_idx": i,
                    "importance_type": "gain",
                    "importance_value": float(imp),
                })
            logger.info(f"XGBoost PW={pw}h: extracted {len(importances)} feature importances")
        except Exception as e:
            logger.warning(f"XGBoost PW={pw}h importance error: {e}")

        # Logistic Regression: coefficient magnitude
        try:
            ckpt_path = get_checkpoint_path(BASELINE_MODEL_CONFIGS["Logistic Regression"], pw)
            model = joblib.load(ckpt_path)
            coefs = np.abs(model.coef_[0])
            for i, (fname, coef) in enumerate(zip(feature_names, coefs)):
                all_rows.append({
                    "model_name": "Logistic Regression",
                    "horizon": pw,
                    "feature_name": fname,
                    "feature_idx": i,
                    "importance_type": "coefficient_abs",
                    "importance_value": float(coef),
                })
            logger.info(f"LR PW={pw}h: extracted {len(coefs)} coefficients")
        except Exception as e:
            logger.warning(f"LR PW={pw}h importance error: {e}")

    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_OUT / "baseline_native_importance.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved native importance: {out_path}")
    return result_df


def plot_baseline_importance(imp_df: pd.DataFrame):
    """Plot baseline feature importance bar charts."""
    for model_name in ["XGBoost", "Logistic Regression"]:
        for pw in PREDICTION_HORIZONS:
            subset = imp_df.query("model_name == @model_name and horizon == @pw").copy()
            if subset.empty:
                continue

            subset = subset.sort_values("importance_value", ascending=True).tail(15)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(subset["feature_name"], subset["importance_value"], color="#2196F3", edgecolor="white")
            imp_type = subset["importance_type"].iloc[0]
            ax.set_xlabel(f"Importance ({imp_type})", fontsize=11)
            ax.set_title(f"{model_name} - Top 15 Features (PH={pw}h)", fontsize=12)
            plt.tight_layout()

            fig_path = FIGURES_OUT / f"importance_{model_name.lower().replace(' ', '_')}_pw{pw}.pdf"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved importance figure: {fig_path}")


# %%
# =============================================================================
# Visualization: Online Simulation Overlay
# =============================================================================

def plot_online_overlay(online_df: pd.DataFrame, reference_type: str = "discharge"):
    """Plot AUROC trajectory overlay with all 5 models."""
    logger.info("Generating online simulation overlay figures...")

    colors = {"ITE Transformer": "#2196F3", "LSTM-Attention": "#4CAF50", "Masked CNN": "#FF9800",
              "XGBoost": "#9C27B0", "Logistic Regression": "#F44336"}
    linestyles = {"ITE Transformer": "-", "LSTM-Attention": "--", "Masked CNN": "-.",
                  "XGBoost": ":", "Logistic Regression": (0, (3, 1, 1, 1))}
    markers = {"ITE Transformer": "o", "LSTM-Attention": "s", "Masked CNN": "^",
               "XGBoost": "D", "Logistic Regression": "v"}

    for pw in PREDICTION_HORIZONS:
        pw_data = online_df.query("horizon == @pw and reference_type == @reference_type")
        if pw_data.empty:
            continue

        datasets_in_data = [d for d in DATASET_ORDER if d in pw_data["dataset"].unique()]
        fig, axes = plt.subplots(1, len(datasets_in_data), figsize=(5 * len(datasets_in_data), 5))
        if len(datasets_in_data) == 1:
            axes = [axes]

        for ax_idx, dataset_name in enumerate(datasets_in_data):
            ax = axes[ax_idx]
            ds_data = pw_data.query("dataset == @dataset_name")
            ds_display = DATASET_DISPLAY.get(dataset_name, dataset_name)

            for model_name in ALL_MODEL_CONFIGS:
                model_data = ds_data.query("model_name == @model_name").sort_values("time_point", ascending=False)
                if model_data.empty or model_data["auroc"].isna().all():
                    continue

                ax.plot(model_data["time_point"], model_data["auroc"],
                        label=model_name, color=colors.get(model_name, "gray"),
                        linestyle=linestyles.get(model_name, "-"),
                        marker=markers.get(model_name, "o"), markersize=4, linewidth=1.5)

            ax.set_xlabel("Hours to Event", fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel("AUROC", fontsize=11)
            ax.set_title(f"{ds_display}", fontsize=12, fontweight="bold")
            ax.set_ylim([0.5, 1.0])
            ax.invert_xaxis()
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.3)

        plt.suptitle(f"Online Simulation - PH={pw}h ({reference_type} reference)", fontsize=13)
        plt.tight_layout()
        suffix = "_matched" if reference_type == "matched" else ""
        fig_path = FIGURES_OUT / f"online_overlay_pw{pw}{suffix}.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {fig_path}")


# %%
# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("Step 2: Baseline Model Evaluation")
    logger.info("=" * 70)

    # 2-A: Single-point
    sp_df = run_single_point_evaluation()

    # 2-B: Online simulation (discharge reference)
    online_df = run_online_simulation_evaluation()
    plot_online_overlay(online_df, "discharge")

    # 2-B: Online simulation (matched reference)
    online_matched_df = run_online_simulation_matched()
    plot_online_overlay(online_matched_df, "matched")

    # Combine online results
    online_all = pd.concat([online_df, online_matched_df], ignore_index=True)
    online_all.to_csv(RESULTS_OUT / "online_simulation_combined.csv", index=False)

    # 2-C: Calibration
    cal_df = run_calibration_analysis()
    plot_calibration_curves()

    # 2-D: Native feature importance
    imp_df = compute_baseline_native_importance()
    plot_baseline_importance(imp_df)

    logger.info("\n" + "=" * 70)
    logger.info("Step 2 Complete")
    logger.info("=" * 70)

    return sp_df, online_all, cal_df, imp_df


# %%
if __name__ == "__main__":
    sp_df, online_all, cal_df, imp_df = main()
