# %%
"""
Step 7: Alert Burden / False-Positive Rate Analysis
Reviewer 2 #5 — threshold sweep, NNE, per-patient alert frequency, lead time.
Analysis range: onset -72h ~ 0h.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from utils.config import (
    SEED, PREDICTION_HORIZONS, DATASETS, DATASET_ORDER,
    ALL_MODEL_CONFIGS, DL_MODEL_CONFIGS, BASELINE_MODEL_CONFIGS,
    RESULTS_DIR, FIGURES_DIR,
    MODEL_DISPLAY_NAMES, MODEL_COLORS, MODEL_ORDER,
    DATASET_DISPLAY, DATASET_PANEL_LABELS, DATASET_COLORS,
    PH_COLORS, FIGURE_DPI_DISPLAY, FIGURE_DPI_EXPORT,
    model_display,
)
from utils.data_loader import (
    load_online_predictions, load_online_with_matched_reference,
)
from utils.metrics import calculate_youden_threshold, decode_pred_proba, sigmoid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "alert_burden"
FIGURES_OUT = FIGURES_DIR / "alert_burden"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TIME_WINDOWS = [0, 12, 24, 36, 48, 60, 72]  # hours before event


# ── Helper: load and prepare online data ─────────────────────────────

def load_and_prepare_online(model_name, pw, dataset_name, use_matched=True):
    """Load online predictions for -72h to 0h window."""
    if use_matched:
        match_path = RESULTS_DIR / "reference_time" / f"reference_time_matching_{DATASETS[dataset_name]['hospital']}.csv"
        if match_path.exists():
            matching_df = pd.read_csv(match_path)
            df = load_online_with_matched_reference(model_name, pw, dataset_name, matching_df)
        else:
            df = load_online_predictions(model_name, pw, dataset_name)
    else:
        df = load_online_predictions(model_name, pw, dataset_name)

    # Decode pred_proba
    if df["pred_proba"].dtype == object:
        df["pred_proba"] = decode_pred_proba(df["pred_proba"])
    if df["pred_proba"].max() > 1.0 or df["pred_proba"].min() < 0.0:
        df["pred_proba"] = df["pred_proba"].apply(sigmoid)

    # Filter to -72h ~ 0h window (time_to_event = hours remaining until event)
    df = df[(df["time_to_event"] >= 0) & (df["time_to_event"] <= 72)].copy()
    return df


# ── 7-A: Threshold sweep per time window ─────────────────────────────

def run_threshold_sweep():
    """Compute sensitivity, specificity, PPV, NPV, FPR, FNR at each threshold × time window."""
    logger.info("=" * 60)
    logger.info("7-A: Threshold Sweep")
    logger.info("=" * 60)

    all_rows = []

    for model_name in ALL_MODEL_CONFIGS:
        for pw in PREDICTION_HORIZONS:
            for dataset_name in DATASET_ORDER:
                try:
                    df = load_and_prepare_online(model_name, pw, dataset_name)
                    if len(df) == 0:
                        continue

                    # Get Youden threshold from internal set for reference
                    from utils.data_loader import load_predictions
                    try:
                        df_sp = load_predictions(model_name, pw, "ilsan_test")
                        youden_thr = calculate_youden_threshold(df_sp["label"].values, df_sp["pred_proba"].values)
                    except Exception:
                        youden_thr = 0.5

                    for tw in TIME_WINDOWS:
                        # Get predictions near this time window (±3h bin)
                        half_bin = 3.0
                        mask = (df["time_to_event"] >= tw - half_bin) & (df["time_to_event"] < tw + half_bin)
                        df_tw = df[mask]
                        if len(df_tw) < 10:
                            continue

                        y_true = df_tw["label"].values
                        y_prob = df_tw["pred_proba"].values
                        n_events = int(y_true.sum())

                        for thr in THRESHOLDS:
                            y_pred = (y_prob >= thr).astype(int)
                            tp = int(((y_pred == 1) & (y_true == 1)).sum())
                            fp = int(((y_pred == 1) & (y_true == 0)).sum())
                            fn = int(((y_pred == 0) & (y_true == 1)).sum())
                            tn = int(((y_pred == 0) & (y_true == 0)).sum())

                            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                            nne = 1.0 / ppv if ppv > 0 else float("inf")

                            all_rows.append({
                                "model": model_display(model_name),
                                "model_internal": model_name,
                                "horizon": pw,
                                "dataset": dataset_name,
                                "dataset_display": DATASET_DISPLAY[dataset_name],
                                "time_window": tw,
                                "threshold": thr,
                                "sensitivity": sens,
                                "specificity": spec,
                                "ppv": ppv,
                                "npv": npv,
                                "fpr": fpr,
                                "fnr": fnr,
                                "nne": nne,
                                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                "n": len(df_tw),
                                "n_events": n_events,
                                "youden_threshold": youden_thr,
                            })

                    logger.info(f"  {model_display(model_name)} PW={pw} {dataset_name}: OK ({len(df)} rows)")

                except Exception as e:
                    logger.error(f"  {model_name} PW={pw} {dataset_name}: {e}")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(RESULTS_OUT / "threshold_sweep.csv", index=False)
    logger.info(f"Saved threshold sweep: {len(df_out)} rows")
    return df_out


# ── 7-B: Per-patient alert frequency ─────────────────────────────────

def run_alert_frequency():
    """Compute per-patient alert counts across the monitoring window."""
    logger.info("=" * 60)
    logger.info("7-B: Per-Patient Alert Frequency")
    logger.info("=" * 60)

    all_rows = []

    for model_name in ALL_MODEL_CONFIGS:
        for pw in PREDICTION_HORIZONS:
            for dataset_name in DATASET_ORDER:
                try:
                    df = load_and_prepare_online(model_name, pw, dataset_name)
                    if len(df) == 0:
                        continue

                    for thr in THRESHOLDS:
                        df["alert"] = (df["pred_proba"] >= thr).astype(int)

                        # Per-patient alert counts
                        patient_alerts = df.groupby(["visit_id", "label"]).agg(
                            n_alerts=("alert", "sum"),
                            n_predictions=("alert", "count"),
                        ).reset_index()

                        aki_patients = patient_alerts[patient_alerts["label"] == 1]
                        nonaki_patients = patient_alerts[patient_alerts["label"] == 0]

                        all_rows.append({
                            "model": model_display(model_name),
                            "horizon": pw,
                            "dataset": dataset_name,
                            "dataset_display": DATASET_DISPLAY[dataset_name],
                            "threshold": thr,
                            "mean_alerts_aki": aki_patients["n_alerts"].mean() if len(aki_patients) > 0 else 0,
                            "median_alerts_aki": aki_patients["n_alerts"].median() if len(aki_patients) > 0 else 0,
                            "mean_alerts_nonaki": nonaki_patients["n_alerts"].mean() if len(nonaki_patients) > 0 else 0,
                            "median_alerts_nonaki": nonaki_patients["n_alerts"].median() if len(nonaki_patients) > 0 else 0,
                            "pct_patients_alerted_aki": (aki_patients["n_alerts"] > 0).mean() * 100 if len(aki_patients) > 0 else 0,
                            "pct_patients_alerted_nonaki": (nonaki_patients["n_alerts"] > 0).mean() * 100 if len(nonaki_patients) > 0 else 0,
                            "n_aki_patients": len(aki_patients),
                            "n_nonaki_patients": len(nonaki_patients),
                        })

                    logger.info(f"  {model_display(model_name)} PW={pw} {dataset_name}: OK")

                except Exception as e:
                    logger.error(f"  {model_name} PW={pw} {dataset_name}: {e}")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(RESULTS_OUT / "alert_frequency.csv", index=False)
    logger.info(f"Saved alert frequency: {len(df_out)} rows")
    return df_out


# ── 7-C: NNE matrix ──────────────────────────────────────────────────

def compute_nne_matrix(threshold_df):
    """Extract NNE = 1/PPV as a matrix (threshold × time_window)."""
    # Already computed in threshold sweep
    nne_df = threshold_df[["model", "horizon", "dataset", "threshold", "time_window", "nne"]].copy()
    nne_df.to_csv(RESULTS_OUT / "nne_matrix.csv", index=False)
    return nne_df


# ── 7-D: Alert-to-onset lead time (TP cases) ────────────────────────

def run_lead_time_analysis():
    """For TP alerts, compute time from first alert to actual AKI onset."""
    logger.info("=" * 60)
    logger.info("7-D: Alert-to-Onset Lead Time (TP cases)")
    logger.info("=" * 60)

    all_rows = []

    for model_name in ALL_MODEL_CONFIGS:
        for pw in PREDICTION_HORIZONS:
            for dataset_name in DATASET_ORDER:
                try:
                    df = load_and_prepare_online(model_name, pw, dataset_name)
                    if len(df) == 0:
                        continue

                    aki_df = df[df["label"] == 1]
                    if len(aki_df) == 0:
                        continue

                    for thr in THRESHOLDS:
                        tp_alerts = aki_df[aki_df["pred_proba"] >= thr]
                        if len(tp_alerts) == 0:
                            all_rows.append({
                                "model": model_display(model_name),
                                "horizon": pw,
                                "dataset": dataset_name,
                                "dataset_display": DATASET_DISPLAY[dataset_name],
                                "threshold": thr,
                                "median_lead_time_hours": np.nan,
                                "iqr_lower": np.nan,
                                "iqr_upper": np.nan,
                                "mean_lead_time_hours": np.nan,
                                "n_tp_alerts": 0,
                                "n_tp_patients": 0,
                            })
                            continue

                        # First TP alert per patient (earliest detection)
                        first_tp = tp_alerts.groupby("visit_id")["time_to_event"].max()  # max = earliest (most hours remaining)
                        lead_times = first_tp.values

                        all_rows.append({
                            "model": model_display(model_name),
                            "horizon": pw,
                            "dataset": dataset_name,
                            "dataset_display": DATASET_DISPLAY[dataset_name],
                            "threshold": thr,
                            "median_lead_time_hours": float(np.median(lead_times)),
                            "iqr_lower": float(np.percentile(lead_times, 25)),
                            "iqr_upper": float(np.percentile(lead_times, 75)),
                            "mean_lead_time_hours": float(np.mean(lead_times)),
                            "n_tp_alerts": len(tp_alerts),
                            "n_tp_patients": len(first_tp),
                        })

                    logger.info(f"  {model_display(model_name)} PW={pw} {dataset_name}: OK")

                except Exception as e:
                    logger.error(f"  {model_name} PW={pw} {dataset_name}: {e}")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(RESULTS_OUT / "lead_time_analysis.csv", index=False)
    logger.info(f"Saved lead time analysis: {len(df_out)} rows")
    return df_out


# ── Figures ───────────────────────────────────────────────────────────

def plot_threshold_sweep(sweep_df):
    """Plot threshold vs sensitivity/FPR/NNE for each model × horizon × dataset."""
    for pw in PREDICTION_HORIZONS:
        for dataset_name in DATASET_ORDER:
            ds_display = DATASET_DISPLAY[dataset_name]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for model_short in MODEL_ORDER:
                mask = (sweep_df["model"] == model_short) & \
                       (sweep_df["horizon"] == pw) & \
                       (sweep_df["dataset"] == dataset_name) & \
                       (sweep_df["time_window"] == 0)  # at onset
                df_m = sweep_df[mask].sort_values("threshold")
                if len(df_m) == 0:
                    continue

                color = MODEL_COLORS[model_short]
                # Sensitivity
                axes[0].plot(df_m["threshold"], df_m["sensitivity"], color=color,
                            marker="o", markersize=4, label=model_short, linewidth=1.5)
                # FPR
                axes[1].plot(df_m["threshold"], df_m["fpr"], color=color,
                            marker="o", markersize=4, label=model_short, linewidth=1.5)
                # NNE (clip at 50 for visualization)
                nne_clipped = df_m["nne"].clip(upper=50)
                axes[2].plot(df_m["threshold"], nne_clipped, color=color,
                            marker="o", markersize=4, label=model_short, linewidth=1.5)

            axes[0].set_ylabel("Sensitivity", fontsize=11)
            axes[0].set_ylim(0, 1.05)
            axes[1].set_ylabel("False Positive Rate", fontsize=11)
            axes[1].set_ylim(0, 1.05)
            axes[2].set_ylabel("NNE (1/PPV)", fontsize=11)
            axes[2].axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="NNE=10")

            for ax in axes:
                ax.set_xlabel("Decision Threshold", fontsize=11)
                ax.legend(fontsize=8, framealpha=0.9)
                ax.grid(True, alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            fig.suptitle(f"Threshold Sweep — {ds_display} (PH={pw}h, at onset)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            safe_ds = dataset_name.replace("-", "_")
            fig.savefig(FIGURES_OUT / f"threshold_sweep_pw{pw}_{safe_ds}.pdf", dpi=FIGURE_DPI_EXPORT, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved: threshold_sweep_pw{pw}_{safe_ds}.pdf")


def plot_alert_burden_heatmap(sweep_df):
    """Heatmap: X=time_window, Y=threshold, color=mean alerts per patient (for best model)."""
    for pw in PREDICTION_HORIZONS:
        for dataset_name in DATASET_ORDER:
            for model_short in MODEL_ORDER[:3]:  # DL models
                mask = (sweep_df["model"] == model_short) & \
                       (sweep_df["horizon"] == pw) & \
                       (sweep_df["dataset"] == dataset_name)
                df_m = sweep_df[mask]
                if len(df_m) == 0:
                    continue

                pivot = df_m.pivot_table(index="threshold", columns="time_window",
                                         values="fpr", aggfunc="mean")
                if pivot.empty:
                    continue

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                              extent=[pivot.columns.min()-3, pivot.columns.max()+3,
                                      pivot.index.max()+0.05, pivot.index.min()-0.05])
                ax.set_xlabel("Hours before event", fontsize=11)
                ax.set_ylabel("Decision Threshold", fontsize=11)
                ax.set_title(f"FPR — {model_short} {DATASET_DISPLAY[dataset_name]} (PH={pw}h)", fontsize=12)
                plt.colorbar(im, ax=ax, label="False Positive Rate")
                plt.tight_layout()

                safe_ds = dataset_name.replace("-", "_")
                fig.savefig(FIGURES_OUT / f"fpr_heatmap_{model_short}_pw{pw}_{safe_ds}.pdf",
                           dpi=FIGURE_DPI_DISPLAY, bbox_inches="tight")
                plt.close(fig)

    logger.info("  Saved alert burden heatmaps")


def suggest_optimal_thresholds(sweep_df, lead_df):
    """Suggest optimal threshold ranges based on Youden and NNE criteria."""
    lines = ["# Optimal Threshold Recommendations\n"]

    for pw in PREDICTION_HORIZONS:
        lines.append(f"\n## Prediction Horizon = {pw}h\n")
        for dataset_name in DATASET_ORDER:
            lines.append(f"\n### {DATASET_DISPLAY[dataset_name]}\n")
            for model_short in MODEL_ORDER:
                mask = (sweep_df["model"] == model_short) & \
                       (sweep_df["horizon"] == pw) & \
                       (sweep_df["dataset"] == dataset_name) & \
                       (sweep_df["time_window"] == 0)
                df_m = sweep_df[mask].sort_values("threshold")
                if len(df_m) == 0:
                    continue

                # Youden index (max sens + spec - 1)
                df_m = df_m.copy()
                df_m["youden"] = df_m["sensitivity"] + df_m["specificity"] - 1
                best_youden = df_m.loc[df_m["youden"].idxmax()]

                # NNE ≤ 10 range
                nne_ok = df_m[df_m["nne"] <= 10]
                nne_range = f"[{nne_ok['threshold'].min():.1f}, {nne_ok['threshold'].max():.1f}]" if len(nne_ok) > 0 else "N/A"

                lines.append(f"- **{model_short}**: Youden optimal thr={best_youden['threshold']:.1f} "
                            f"(sens={best_youden['sensitivity']:.3f}, spec={best_youden['specificity']:.3f}); "
                            f"NNE≤10 range: {nne_range}")

    text = "\n".join(lines)
    (RESULTS_OUT / "optimal_threshold_recommendations.md").write_text(text)
    logger.info("Saved optimal threshold recommendations")
    return text


# ── Main ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Step 7: Alert Burden / FPR Analysis")
    logger.info("=" * 70)

    sweep_df = run_threshold_sweep()
    alert_df = run_alert_frequency()
    nne_df = compute_nne_matrix(sweep_df)
    lead_df = run_lead_time_analysis()

    # Figures
    logger.info("\nGenerating figures...")
    plot_threshold_sweep(sweep_df)
    plot_alert_burden_heatmap(sweep_df)
    suggest_optimal_thresholds(sweep_df, lead_df)

    # Combined CSV
    combined = sweep_df.merge(
        lead_df[["model", "horizon", "dataset", "threshold",
                 "median_lead_time_hours", "mean_lead_time_hours"]],
        on=["model", "horizon", "dataset", "threshold"],
        how="left",
    )
    combined.to_csv(RESULTS_OUT / "alert_burden_combined.csv", index=False)
    logger.info(f"\nSaved combined: {len(combined)} rows")

    logger.info("\nStep 7 Complete!")


if __name__ == "__main__":
    main()
