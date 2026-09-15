# %%
"""
Step D: Figure Generation for Clinical Faithfulness Analysis

D-1: AUROC + PPV Trajectory plots (per model, with CI and trend lines)
D-2: Clinical Faithfulness Summary Heatmap (tau values, significance)
D-3: DL vs Baseline comparison overlay
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from utils.config import (
    RESULTS_DIR, FIGURES_DIR,
    MODEL_DISPLAY_NAMES, MODEL_COLORS, MODEL_LINESTYLES, MODEL_MARKERS, MODEL_ORDER,
    DATASET_ORDER, DATASET_DISPLAY, DATASET_PANEL_LABELS,
    FIGURE_DPI_EXPORT, FIGURE_FONT_FAMILY,
    ALL_MODEL_CONFIGS, DL_MODEL_CONFIGS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_IN = RESULTS_DIR / "clinical_faithfulness"
FIGURES_OUT = FIGURES_DIR / "clinical_faithfulness"
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": FIGURE_FONT_FAMILY,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# %%
def _get_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _plot_trajectory_with_trend(
    ax, x, y, y_ci_lo, y_ci_hi, sens_slope, sens_intercept,
    color, linestyle, marker, label, show_ci=True,
):
    """Plot metric trajectory with CI shading and Sen's slope trend line."""
    ax.plot(x, y, color=color, linestyle=linestyle, marker=marker,
            markersize=5, linewidth=1.8, label=label, zorder=3)
    if show_ci and y_ci_lo is not None and y_ci_hi is not None:
        ax.fill_between(x, y_ci_lo, y_ci_hi, color=color, alpha=0.15, zorder=1)
    # Sen's slope trend line
    if sens_slope is not None and not np.isnan(sens_slope):
        x_line = np.array([max(x), min(x)])
        y_line = sens_intercept + sens_slope * x_line
        ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=1.0, alpha=0.6, zorder=2)


# %%
# D-1: AUROC + PPV Trajectory per model
def plot_d1_trajectory(
    metrics_df: pd.DataFrame,
    mk_df: pd.DataFrame,
    trend_df: pd.DataFrame,
):
    """Generate AUROC + PPV trajectory figures for each model."""
    logger.info("D-1: Generating AUROC + PPV trajectory figures...")

    for model_name in ALL_MODEL_CONFIGS:
        disp = _get_display_name(model_name)
        color = MODEL_COLORS[disp]

        for pw in metrics_df["horizon"].unique():
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f"{disp} — Prediction Window {pw}h", fontsize=14, fontweight="bold", y=0.98)

            for col_idx, dataset in enumerate(DATASET_ORDER):
                ds_display = DATASET_DISPLAY[dataset]
                panel_label = DATASET_PANEL_LABELS[dataset]

                sub = metrics_df[
                    (metrics_df["model"] == model_name) &
                    (metrics_df["horizon"] == pw) &
                    (metrics_df["dataset"] == dataset)
                ].sort_values("time_window", ascending=False)

                if sub.empty:
                    continue

                x = sub["hours_before_onset"].values

                # --- Top row: AUROC ---
                ax_auroc = axes[0, col_idx]
                y_auroc = sub["auroc"].values
                ci_lo = sub["auroc_ci_lower"].values
                ci_hi = sub["auroc_ci_upper"].values

                # Get trend info
                mk_row = mk_df[
                    (mk_df["model"] == model_name) &
                    (mk_df["horizon"] == pw) &
                    (mk_df["dataset"] == dataset) &
                    (mk_df["metric_column"] == "auroc")
                ]
                trend_row = trend_df[
                    (trend_df["model"] == model_name) &
                    (trend_df["horizon"] == pw) &
                    (trend_df["dataset"] == dataset) &
                    (trend_df["metric"] == "auroc")
                ]

                sens_slope = trend_row.iloc[0]["sens_slope"] if not trend_row.empty else np.nan
                sens_intercept = mk_row.iloc[0].get("sens_intercept", np.nan) if not mk_row.empty else np.nan
                # Compute intercept from trend_df linear
                if not trend_row.empty:
                    lin_intercept = trend_row.iloc[0].get("linear_intercept", np.nan)
                    lin_beta = trend_row.iloc[0].get("linear_beta", np.nan)
                else:
                    lin_intercept, lin_beta = np.nan, np.nan

                _plot_trajectory_with_trend(
                    ax_auroc, x, y_auroc, ci_lo, ci_hi,
                    sens_slope=lin_beta, sens_intercept=lin_intercept,
                    color=color, linestyle="-", marker="o", label="AUROC",
                )

                # Annotate MK test
                if not mk_row.empty:
                    tau = mk_row.iloc[0]["tau"]
                    p = mk_row.iloc[0]["p_value"]
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                    ax_auroc.text(
                        0.02, 0.05,
                        f"τ = {tau:.3f}, p = {p:.4f} {sig}",
                        transform=ax_auroc.transAxes, fontsize=8,
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    )

                ax_auroc.set_title(f"{panel_label}. {ds_display}", fontweight="bold")
                ax_auroc.set_ylim([0.5, 1.0])
                ax_auroc.set_xlim(75, -3)
                ax_auroc.set_ylabel("AUROC" if col_idx == 0 else "")
                ax_auroc.grid(True, alpha=0.3)

                # --- Bottom row: PPV (Youden threshold) ---
                ax_ppv = axes[1, col_idx]
                ppv_col = "ppv_youden"
                ppv_ci_lo_col = "ppv_youden_ci_lower"
                ppv_ci_hi_col = "ppv_youden_ci_upper"

                if ppv_col in sub.columns:
                    y_ppv = sub[ppv_col].values
                    ppv_lo = sub[ppv_ci_lo_col].values
                    ppv_hi = sub[ppv_ci_hi_col].values

                    mk_ppv = mk_df[
                        (mk_df["model"] == model_name) &
                        (mk_df["horizon"] == pw) &
                        (mk_df["dataset"] == dataset) &
                        (mk_df["metric_column"] == ppv_col)
                    ]
                    trend_ppv = trend_df[
                        (trend_df["model"] == model_name) &
                        (trend_df["horizon"] == pw) &
                        (trend_df["dataset"] == dataset) &
                        (trend_df["metric"] == ppv_col)
                    ]

                    lin_intercept_ppv = trend_ppv.iloc[0].get("linear_intercept", np.nan) if not trend_ppv.empty else np.nan
                    lin_beta_ppv = trend_ppv.iloc[0].get("linear_beta", np.nan) if not trend_ppv.empty else np.nan

                    _plot_trajectory_with_trend(
                        ax_ppv, x, y_ppv, ppv_lo, ppv_hi,
                        sens_slope=lin_beta_ppv, sens_intercept=lin_intercept_ppv,
                        color=color, linestyle="-", marker="s", label="PPV",
                    )

                    if not mk_ppv.empty:
                        tau = mk_ppv.iloc[0]["tau"]
                        p = mk_ppv.iloc[0]["p_value"]
                        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                        ax_ppv.text(
                            0.02, 0.05,
                            f"τ = {tau:.3f}, p = {p:.4f} {sig}",
                            transform=ax_ppv.transAxes, fontsize=8,
                            verticalalignment="bottom",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        )

                ax_ppv.set_xlabel("Hours Before Event")
                ax_ppv.set_ylabel("PPV (Youden)" if col_idx == 0 else "")
                ax_ppv.set_ylim([0.0, 1.0])
                ax_ppv.set_xlim(75, -3)
                ax_ppv.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig_path = FIGURES_OUT / f"d1_trajectory_{disp}_pw{pw}.pdf"
            fig.savefig(fig_path, dpi=FIGURE_DPI_EXPORT, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved: {fig_path}")


# %%
# D-2: Summary Heatmap
def plot_d2_heatmap(mk_df: pd.DataFrame):
    """Generate Clinical Faithfulness summary heatmap."""
    logger.info("D-2: Generating summary heatmap...")

    metric_cols = ["auroc", "auprc", "ppv_03", "ppv_05"]
    metric_labels = ["AUROC", "AUPRC", "PPV@0.3", "PPV@0.5"]

    for pw in mk_df["horizon"].unique():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Clinical Faithfulness — Mann-Kendall τ (Prediction Window {pw}h)",
                     fontsize=14, fontweight="bold")

        for ax_idx, dataset in enumerate(DATASET_ORDER):
            ax = axes[ax_idx]
            ds_display = DATASET_DISPLAY[dataset]

            sub = mk_df[
                (mk_df["horizon"] == pw) &
                (mk_df["dataset"] == dataset) &
                (mk_df["metric_column"].isin(metric_cols))
            ]

            if sub.empty:
                ax.set_visible(False)
                continue

            # Build matrix: rows = metrics, cols = models
            models_present = [m for m in ALL_MODEL_CONFIGS if m in sub["model"].values]
            model_disps = [_get_display_name(m) for m in models_present]

            tau_matrix = np.full((len(metric_cols), len(models_present)), np.nan)
            sig_matrix = np.full((len(metric_cols), len(models_present)), "", dtype=object)

            for i, mc in enumerate(metric_cols):
                for j, model in enumerate(models_present):
                    row = sub[(sub["model"] == model) & (sub["metric_column"] == mc)]
                    if not row.empty:
                        tau_val = row.iloc[0]["tau"]
                        p_val = row.iloc[0]["p_value"]
                        tau_matrix[i, j] = tau_val
                        if p_val < 0.001:
                            sig_matrix[i, j] = "***"
                        elif p_val < 0.01:
                            sig_matrix[i, j] = "**"
                        elif p_val < 0.05:
                            sig_matrix[i, j] = "*"

            # Custom colormap: gray for non-significant, green gradient for significant
            cmap = plt.cm.RdYlGn
            norm = mcolors.Normalize(vmin=-1, vmax=1)

            im = ax.imshow(tau_matrix, cmap=cmap, norm=norm, aspect="auto")

            # Annotate cells
            for i in range(len(metric_cols)):
                for j in range(len(models_present)):
                    val = tau_matrix[i, j]
                    sig = sig_matrix[i, j]
                    if np.isnan(val):
                        continue
                    text_color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}\n{sig}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=text_color)

            ax.set_xticks(range(len(model_disps)))
            ax.set_xticklabels(model_disps, fontsize=10)
            ax.set_yticks(range(len(metric_labels)))
            ax.set_yticklabels(metric_labels if ax_idx == 0 else [], fontsize=10)
            ax.set_title(f"{DATASET_PANEL_LABELS[dataset]}. {ds_display}", fontweight="bold")

        # Colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Kendall's τ")

        plt.tight_layout(rect=[0, 0, 0.95, 0.93])
        fig_path = FIGURES_OUT / f"d2_heatmap_pw{pw}.pdf"
        fig.savefig(fig_path, dpi=FIGURE_DPI_EXPORT, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")


# %%
# D-3: DL vs Baseline Comparison Overlay
def plot_d3_comparison(
    metrics_df: pd.DataFrame,
    mk_df: pd.DataFrame,
):
    """Plot DL vs Baseline AUROC trajectory comparison."""
    logger.info("D-3: Generating DL vs Baseline comparison...")

    for pw in metrics_df["horizon"].unique():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"AUROC Trajectory — All Models (Prediction Window {pw}h)",
                     fontsize=14, fontweight="bold")

        for ax_idx, dataset in enumerate(DATASET_ORDER):
            ax = axes[ax_idx]
            ds_display = DATASET_DISPLAY[dataset]

            for model_name in ALL_MODEL_CONFIGS:
                disp = _get_display_name(model_name)
                color = MODEL_COLORS[disp]
                ls = MODEL_LINESTYLES[disp]
                marker = MODEL_MARKERS[disp]

                sub = metrics_df[
                    (metrics_df["model"] == model_name) &
                    (metrics_df["horizon"] == pw) &
                    (metrics_df["dataset"] == dataset)
                ].sort_values("time_window", ascending=False)

                if sub.empty:
                    continue

                x = sub["hours_before_onset"].values
                y = sub["auroc"].values
                ci_lo = sub["auroc_ci_lower"].values
                ci_hi = sub["auroc_ci_upper"].values

                ax.plot(x, y, color=color, linestyle=ls, marker=marker,
                        markersize=5, linewidth=1.8, label=disp, zorder=3)
                ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.1, zorder=1)

            ax.set_xlabel("Hours Before Event")
            ax.set_ylabel("AUROC" if ax_idx == 0 else "")
            ax.set_title(f"{DATASET_PANEL_LABELS[dataset]}. {ds_display}", fontweight="bold")
            ax.set_ylim([0.5, 1.0])
            ax.set_xlim(75, -3)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = FIGURES_OUT / f"d3_comparison_pw{pw}.pdf"
        fig.savefig(fig_path, dpi=FIGURE_DPI_EXPORT, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")


# %%
def run_step_d():
    """Generate all figures."""
    logger.info("=" * 60)
    logger.info("Step D: Figure Generation")
    logger.info("=" * 60)

    metrics_df = pd.read_csv(RESULTS_IN / "step_a_timewindow_metrics.csv")
    mk_df = pd.read_csv(RESULTS_IN / "step_b_mann_kendall_results.csv")
    trend_df = pd.read_csv(RESULTS_IN / "step_c_trend_estimation.csv")

    plot_d1_trajectory(metrics_df, mk_df, trend_df)
    plot_d2_heatmap(mk_df)
    plot_d3_comparison(metrics_df, mk_df)

    logger.info("Step D complete.")


if __name__ == "__main__":
    run_step_d()
