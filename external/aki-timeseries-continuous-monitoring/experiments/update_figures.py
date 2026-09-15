# %%
"""
Figure Update: Regenerate all revision figures with enhanced formatting.
- Model names: TF, LSTM, CNN, XGB, LR
- Dataset order: NHIS, CSHH, MIMIC-IV
- Consistent colors/markers/line styles across all figures
- Enhanced draft style (publication quality)
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
from matplotlib.lines import Line2D

from utils.config import (
    SEED, PREDICTION_HORIZONS, DATASETS, DATASET_ORDER,
    ALL_MODEL_CONFIGS, DL_MODEL_CONFIGS, BASELINE_MODEL_CONFIGS,
    RESULTS_DIR, FIGURES_DIR,
    MODEL_DISPLAY_NAMES, MODEL_COLORS, MODEL_LINESTYLES, MODEL_MARKERS, MODEL_ORDER,
    DATASET_DISPLAY, DATASET_PANEL_LABELS, DATASET_COLORS,
    PH_COLORS, PH_ALPHAS, PH_LW_MULT,
    FIGURE_DPI_DISPLAY, FIGURE_DPI_EXPORT, FIGURE_FONT_FAMILY,
    model_display,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Global rcParams ──────────────────────────────────────────────────

def set_publication_style():
    plt.rcParams.update({
        "font.family": FIGURE_FONT_FAMILY,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": FIGURE_DPI_DISPLAY,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ── Figure 1: Online Simulation Overlay (DL + Baselines) ────────────

def figure_online_overlay():
    """
    Online simulation AUROC: 5 models × 3 horizons per dataset panel.
    1 row × 3 columns (NHIS, CSHH, MIMIC-IV).
    Matched reference time. Enhanced format from draft.
    """
    logger.info("=" * 60)
    logger.info("Figure: Online Simulation Overlay (5 models)")
    logger.info("=" * 60)

    set_publication_style()

    # Load matched online simulation results
    csv_path = RESULTS_DIR / "baseline_comparison" / "online_simulation_matched_all_models.csv"
    if not csv_path.exists():
        logger.warning(f"  Missing: {csv_path}")
        return
    df = pd.read_csv(csv_path)

    for pw in PREDICTION_HORIZONS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

        for col_idx, dataset_name in enumerate(DATASET_ORDER):
            ax = axes[col_idx]
            ds_display = DATASET_DISPLAY[dataset_name]
            panel_label = DATASET_PANEL_LABELS[dataset_name]

            for model_internal in ALL_MODEL_CONFIGS:
                model_short = model_display(model_internal)
                mask = (df["model_name"] == model_internal) & \
                       (df["horizon"] == pw) & \
                       (df["dataset"] == dataset_name)
                df_m = df[mask].sort_values("time_point")
                if len(df_m) == 0:
                    continue

                color = MODEL_COLORS[model_short]
                ls = MODEL_LINESTYLES[model_short]
                marker = MODEL_MARKERS[model_short]
                is_dl = model_internal in DL_MODEL_CONFIGS
                lw = 2.2 if is_dl else 1.5
                alpha = 0.95 if is_dl else 0.65

                ax.plot(
                    df_m["time_point"].values,
                    df_m["auroc"].values,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    marker=marker,
                    markersize=4 if is_dl else 3,
                    markevery=2,
                    label=model_short,
                    zorder=10 if is_dl else 5,
                )

            ax.set_xlim([74, -2])
            ax.set_ylim([0.48, 1.02])
            ax.set_xlabel("Hours before AKI onset", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("AUROC", fontsize=11)
            ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
            ax.set_title(ds_display)

            # Panel label
            ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top")

        # Legend
        legend_elements = []
        for ms in MODEL_ORDER:
            is_dl = ms in ["TF", "LSTM", "CNN"]
            legend_elements.append(
                Line2D([0], [0], color=MODEL_COLORS[ms], linestyle=MODEL_LINESTYLES[ms],
                       linewidth=2.2 if is_dl else 1.5, marker=MODEL_MARKERS[ms],
                       markersize=4, label=ms, alpha=0.95 if is_dl else 0.65)
            )

        fig.legend(handles=legend_elements, loc="lower center", ncol=5,
                   fontsize=9, framealpha=0.95, edgecolor="none",
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.16, top=0.92, wspace=0.25)

        out_dir = FIGURES_DIR / "baseline_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"online_overlay_pw{pw}_matched.pdf",
                   dpi=FIGURE_DPI_EXPORT, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  Saved: online_overlay_pw{pw}_matched.pdf")


# ── Figure 2: Calibration Comparison ─────────────────────────────────

def figure_calibration():
    """Calibration curves for all 5 models."""
    logger.info("=" * 60)
    logger.info("Figure: Calibration Comparison")
    logger.info("=" * 60)

    set_publication_style()
    cal_path = RESULTS_DIR / "baseline_comparison" / "calibration_all_models.csv"
    if not cal_path.exists():
        logger.warning(f"  Missing: {cal_path}")
        return

    df = pd.read_csv(cal_path)

    for pw in PREDICTION_HORIZONS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for col_idx, dataset_name in enumerate(DATASET_ORDER):
            ax = axes[col_idx]
            ds_display = DATASET_DISPLAY[dataset_name]

            df_sub = df[(df["horizon"] == pw) & (df["dataset"] == dataset_name)]
            if len(df_sub) == 0:
                continue

            # Bar chart of Brier scores (before/after)
            models_present = []
            brier_before = []
            brier_after = []
            colors_bars = []

            for model_internal in list(DL_MODEL_CONFIGS) + list(BASELINE_MODEL_CONFIGS):
                row = df_sub[df_sub["model_name"] == model_internal]
                if len(row) == 0:
                    continue
                ms = model_display(model_internal)
                models_present.append(ms)
                brier_before.append(row["brier_before"].values[0])
                brier_after.append(row["brier_after"].values[0])
                colors_bars.append(MODEL_COLORS[ms])

            x = np.arange(len(models_present))
            width = 0.35
            ax.bar(x - width/2, brier_before, width, label="Before", color=colors_bars, alpha=0.4, edgecolor="gray")
            ax.bar(x + width/2, brier_after, width, label="After", color=colors_bars, alpha=0.9, edgecolor="gray")
            ax.set_xticks(x)
            ax.set_xticklabels(models_present, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Brier Score" if col_idx == 0 else "")
            ax.set_title(ds_display)
            ax.legend(fontsize=7, framealpha=0.9)
            ax.grid(True, axis="y", alpha=0.25)

            ax.text(-0.12, 1.05, DATASET_PANEL_LABELS[dataset_name],
                    transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

        plt.tight_layout()
        out_dir = FIGURES_DIR / "baseline_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"calibration_comparison_pw{pw}.pdf",
                   dpi=FIGURE_DPI_EXPORT, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  Saved: calibration_comparison_pw{pw}.pdf")


# ── Figure 3: Feature Importance Bar Charts ──────────────────────────

def figure_feature_importance():
    """Top-15 feature importance bar charts for all models."""
    logger.info("=" * 60)
    logger.info("Figure: Feature Importance (Single-Point)")
    logger.info("=" * 60)

    set_publication_style()

    fi_path = RESULTS_DIR / "feature_importance" / "feature_importance_single_all.csv"
    if not fi_path.exists():
        logger.warning(f"  Missing: {fi_path}")
        return

    df = pd.read_csv(fi_path)

    out_dir = FIGURES_DIR / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pw in PREDICTION_HORIZONS:
        for dataset_name in DATASET_ORDER:
            ds_display = DATASET_DISPLAY[dataset_name]
            # Get models available for this combo
            df_sub = df[(df["horizon"] == pw) & (df["dataset"] == dataset_name)]
            models_in = df_sub["model"].unique()

            for model_internal in models_in:
                ms = model_display(model_internal)
                df_m = df_sub[df_sub["model"] == model_internal].copy()
                if len(df_m) == 0:
                    continue

                df_m = df_m.sort_values("mean_importance", ascending=True).tail(15)

                fig, ax = plt.subplots(figsize=(8, 6))
                color = MODEL_COLORS.get(ms, "#666666")
                ax.barh(df_m["feature_name"], df_m["mean_importance"],
                       xerr=df_m["std_importance"], color=color, alpha=0.8,
                       edgecolor="white", capsize=3)
                ax.axvline(x=0, color="black", linewidth=0.5)
                ax.set_xlabel("AUROC Drop (Importance)", fontsize=11)
                baseline_auroc = df_m["baseline_auroc"].iloc[0]
                ax.set_title(f"{ms} — {ds_display} (PH={pw}h)\n"
                            f"Baseline AUROC={baseline_auroc:.3f}", fontsize=12)
                plt.tight_layout()

                safe_model = ms.lower()
                safe_ds = dataset_name.replace("-", "_")
                fig.savefig(out_dir / f"fi_single_{safe_model}_pw{pw}_{safe_ds}.pdf",
                           dpi=FIGURE_DPI_EXPORT, bbox_inches="tight", facecolor="white")
                plt.close(fig)

    logger.info("  Saved feature importance bar charts")


# ── Figure 4: Sensitivity Analysis (Reference Time Comparison) ──────

def figure_sensitivity_analysis():
    """AUROC trajectory overlay: matched (solid) vs discharge (dashed)."""
    logger.info("=" * 60)
    logger.info("Figure: Sensitivity Analysis (Ref Time Comparison)")
    logger.info("=" * 60)

    set_publication_style()

    csv_path = RESULTS_DIR / "sensitivity_analysis" / "reference_time_comparison.csv"
    if not csv_path.exists():
        logger.warning(f"  Missing: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    out_dir = FIGURES_DIR / "sensitivity_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pw in PREDICTION_HORIZONS:
        for dataset_name in DATASET_ORDER:
            ds_display = DATASET_DISPLAY[dataset_name]
            fig, ax = plt.subplots(figsize=(8, 5.5))

            for model_internal in ALL_MODEL_CONFIGS:
                ms = model_display(model_internal)
                color = MODEL_COLORS[ms]

                for ref_type, ls_mod, alpha in [("matched", "-", 0.95), ("discharge", "--", 0.5)]:
                    mask = (df["model_name"] == model_internal) & \
                           (df["horizon"] == pw) & \
                           (df["dataset"] == dataset_name) & \
                           (df["reference_type"] == ref_type)
                    df_m = df[mask].sort_values("time_window")
                    if len(df_m) == 0:
                        continue

                    label = f"{ms} ({ref_type})" if ref_type == "matched" else None
                    ax.plot(df_m["time_window"].values, df_m["auroc"].values,
                           color=color, linestyle=ls_mod, linewidth=1.8, alpha=alpha,
                           label=label)

            ax.set_xlim([74, -2])
            ax.set_ylim([0.48, 1.02])
            ax.set_xlabel("Hours before event", fontsize=11)
            ax.set_ylabel("AUROC", fontsize=11)
            ax.set_title(f"Reference Time Comparison — {ds_display} (PH={pw}h)", fontsize=12)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
            plt.tight_layout()

            safe_ds = dataset_name.replace("-", "_")
            fig.savefig(out_dir / f"reference_comparison_pw{pw}_{safe_ds}.pdf",
                       dpi=FIGURE_DPI_EXPORT, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    logger.info("  Saved sensitivity analysis figures")


# ── Figure 5: Baseline Native FI ────────────────────────────────────

def figure_baseline_native_importance():
    """XGBoost gain + LR coefficients bar charts."""
    logger.info("=" * 60)
    logger.info("Figure: Baseline Native Feature Importance")
    logger.info("=" * 60)

    set_publication_style()

    csv_path = RESULTS_DIR / "baseline_comparison" / "baseline_native_importance.csv"
    if not csv_path.exists():
        logger.warning(f"  Missing: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    out_dir = FIGURES_DIR / "baseline_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pw in PREDICTION_HORIZONS:
        for model_internal in ["XGBoost", "Logistic Regression"]:
            ms = model_display(model_internal)
            df_m = df[(df["model_name"] == model_internal) & (df["horizon"] == pw)]
            if len(df_m) == 0:
                continue

            df_m = df_m.sort_values("importance_value", ascending=True).tail(15)

            fig, ax = plt.subplots(figsize=(8, 6))
            color = MODEL_COLORS[ms]
            imp_type = df_m["importance_type"].iloc[0]
            ax.barh(df_m["feature_name"], df_m["importance_value"], color=color, alpha=0.8,
                   edgecolor="white")
            ax.set_xlabel(f"Importance ({imp_type})", fontsize=11)
            ax.set_title(f"{ms} Native Feature Importance (PH={pw}h)", fontsize=12)
            plt.tight_layout()

            fig.savefig(out_dir / f"importance_{ms.lower()}_pw{pw}.pdf",
                       dpi=FIGURE_DPI_EXPORT, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    logger.info("  Saved baseline native importance figures")


# ── Figure 6: Feature Subset Performance ─────────────────────────────

def figure_feature_subset():
    """Feature subset performance comparison."""
    logger.info("=" * 60)
    logger.info("Figure: Feature Subset Performance")
    logger.info("=" * 60)

    set_publication_style()

    csv_path = RESULTS_DIR / "sensitivity_analysis" / "feature_subset_comparison.csv"
    if not csv_path.exists():
        logger.warning(f"  Missing: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    out_dir = FIGURES_DIR / "sensitivity_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pw in PREDICTION_HORIZONS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for col_idx, dataset_name in enumerate(DATASET_ORDER):
            ax = axes[col_idx]
            ds_display = DATASET_DISPLAY[dataset_name]

            df_sub = df[(df["horizon"] == pw) & (df["dataset"] == dataset_name)]
            if len(df_sub) == 0:
                continue

            for model_internal in df_sub["model_name"].unique():
                ms = model_display(model_internal)
                df_m = df_sub[df_sub["model_name"] == model_internal].sort_values("fraction")
                if len(df_m) == 0:
                    continue

                color = MODEL_COLORS.get(ms, "#666666")
                fractions = df_m["fraction"].values
                auroc_drops = df_m["auroc_drop_pct"].values

                ax.plot(fractions * 100, auroc_drops, color=color, marker="o",
                       markersize=5, linewidth=1.5, label=ms)

            ax.set_xlabel("Features kept (%)", fontsize=11)
            ax.set_ylabel("AUROC drop (%)" if col_idx == 0 else "", fontsize=11)
            ax.set_title(ds_display)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.legend(fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.25)

            ax.text(-0.12, 1.05, DATASET_PANEL_LABELS[dataset_name],
                    transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

        plt.tight_layout()
        fig.savefig(out_dir / f"feature_subset_pw{pw}.pdf",
                   dpi=FIGURE_DPI_EXPORT, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    logger.info("  Saved feature subset figures")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Figure Update: Regenerating all revision figures")
    logger.info("=" * 70)

    figure_online_overlay()
    figure_calibration()
    figure_feature_importance()
    figure_sensitivity_analysis()
    figure_baseline_native_importance()
    figure_feature_subset()

    logger.info("\nAll figures updated!")


if __name__ == "__main__":
    main()
