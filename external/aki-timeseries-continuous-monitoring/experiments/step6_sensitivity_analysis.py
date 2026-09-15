# %%
"""
Step 6: Sensitivity Analysis - Reference Time Method Comparison

Compare ratio-matched reference time (Step 1) vs original discharge-based
reference time for online simulation results.

Analysis: onset -72h to 0h range.
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

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from utils.config import (
    SEED, PREDICTION_HORIZONS, DATASETS, TIME_POINTS_ONLINE,
    ALL_MODEL_CONFIGS, DL_MODEL_CONFIGS, BASELINE_MODEL_CONFIGS,
    RESULTS_DIR, FIGURES_DIR,
)
from utils.data_loader import load_online_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "sensitivity_analysis"
FIGURES_OUT = FIGURES_DIR / "sensitivity_analysis"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

WINDOW_HALF_SIZE = 3.0


# %%
def compute_auroc_at_timepoint(df: pd.DataFrame, time_point: float, half_window: float = WINDOW_HALF_SIZE) -> dict:
    """Compute AUROC at a specific time-to-event window."""
    mask = (df["time_to_event"] >= time_point - half_window) & (df["time_to_event"] < time_point + half_window)
    subset = df[mask]

    if len(subset) < 30 or len(np.unique(subset["label"])) < 2:
        return {"auroc": np.nan, "n": len(subset), "n_events": int(subset["label"].sum())}

    return {
        "auroc": roc_auc_score(subset["label"], subset["pred_proba"]),
        "n": len(subset),
        "n_events": int(subset["label"].sum()),
    }


def compute_online_metrics_both_references(
    model_name: str, pw: int, dataset: str,
) -> list[dict]:
    """Compute AUROC at all time points for both reference time methods."""
    hospital_map = {"ilsan_test": "ilsan", "cchlmc_external": "cchlmc", "mimic-iv_external": "mimic-iv"}
    hospital = hospital_map[dataset]

    df = load_online_predictions(model_name, pw, dataset)

    results = []

    # 1. Discharge reference (original)
    for tp in TIME_POINTS_ONLINE:
        metrics = compute_auroc_at_timepoint(df, tp)
        results.append({
            "model_name": model_name, "horizon": pw, "dataset": dataset,
            "time_window": tp, "reference_type": "discharge",
            **metrics,
        })

    # 2. Matched reference
    match_path = RESULTS_DIR / "reference_time" / f"reference_time_matching_{hospital}.csv"
    if match_path.exists():
        match_df = pd.read_csv(match_path)
        df_matched = df.copy()
        pseudo_map = match_df.set_index("visit_id")["pseudo_onset_hours"]
        non_aki_mask = df_matched["label"] == 0
        matched_vals = df_matched.loc[non_aki_mask, "visit_id"].map(pseudo_map)
        df_matched.loc[non_aki_mask, "time_to_event"] = matched_vals.values - df_matched.loc[non_aki_mask, "timestamp"].values

        for tp in TIME_POINTS_ONLINE:
            metrics = compute_auroc_at_timepoint(df_matched, tp)
            results.append({
                "model_name": model_name, "horizon": pw, "dataset": dataset,
                "time_window": tp, "reference_type": "matched",
                **metrics,
            })

    return results


# %%
def plot_auroc_overlay(results_df: pd.DataFrame):
    """Plot AUROC trajectory overlay: solid=matched, dashed=discharge."""
    colors = {"ITE Transformer": "#2196F3", "LSTM-Attention": "#4CAF50", "Masked CNN": "#FF9800",
              "XGBoost": "#9C27B0", "Logistic Regression": "#F44336"}

    for pw in PREDICTION_HORIZONS:
        for dataset_name, ds_info in DATASETS.items():
            fig, ax = plt.subplots(figsize=(8, 5))

            for model_name in ALL_MODEL_CONFIGS:
                for ref_type, ls in [("matched", "-"), ("discharge", "--")]:
                    subset = results_df.query(
                        "model_name == @model_name and horizon == @pw and "
                        "dataset == @dataset_name and reference_type == @ref_type"
                    ).sort_values("time_window", ascending=False)

                    if subset.empty or subset["auroc"].isna().all():
                        continue

                    label = f"{model_name} ({ref_type})" if ref_type == "matched" else None
                    ax.plot(subset["time_window"], subset["auroc"],
                            color=colors.get(model_name, "gray"),
                            linestyle=ls, linewidth=1.5 if ref_type == "matched" else 1.0,
                            alpha=1.0 if ref_type == "matched" else 0.5,
                            label=label, marker="o" if ref_type == "matched" else None,
                            markersize=3)

            ax.set_xlabel("Hours to Event", fontsize=11)
            ax.set_ylabel("AUROC", fontsize=11)
            # Title removed per request
            ax.set_ylim([0.5, 1.0])
            ax.invert_xaxis()
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            safe_ds = dataset_name.replace("-", "_")
            fig_path = FIGURES_OUT / f"reference_comparison_pw{pw}_{safe_ds}.pdf"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved: {fig_path}")


def compute_fi_ranking_correlation(fi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation of FI rankings between reference methods."""
    # This would use Step 4 results with both reference types
    # For now, compute correlation between different models/horizons
    results = []

    if fi_df.empty:
        return pd.DataFrame()

    for pw in fi_df["horizon"].unique():
        for dataset in fi_df["dataset"].unique():
            models = fi_df.query("horizon == @pw and dataset == @dataset")["model"].unique()
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    m1_data = fi_df.query("model == @models[@i] and horizon == @pw and dataset == @dataset")
                    m2_data = fi_df.query("model == @models[@j] and horizon == @pw and dataset == @dataset")

                    if m1_data.empty or m2_data.empty:
                        continue

                    merged = m1_data.merge(m2_data, on="feature_name", suffixes=("_1", "_2"))
                    if len(merged) < 5:
                        continue

                    rho, pval = spearmanr(merged["mean_importance_1"], merged["mean_importance_2"])
                    results.append({
                        "horizon": pw, "dataset": dataset,
                        "model_1": models[i], "model_2": models[j],
                        "spearman_rho": rho, "p_value": pval,
                        "n_features": len(merged),
                    })

    return pd.DataFrame(results)


# %%
def create_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table of AUROC values between methods."""
    pivot_discharge = results_df.query("reference_type == 'discharge'").pivot_table(
        index=["model_name", "horizon", "dataset"],
        columns="time_window",
        values="auroc",
    ).add_prefix("auroc_discharge_")

    pivot_matched = results_df.query("reference_type == 'matched'").pivot_table(
        index=["model_name", "horizon", "dataset"],
        columns="time_window",
        values="auroc",
    ).add_prefix("auroc_matched_")

    comparison = pivot_discharge.join(pivot_matched)

    # Compute differences
    for tp in TIME_POINTS_ONLINE:
        dc = f"auroc_discharge_{tp}"
        mc = f"auroc_matched_{tp}"
        if dc in comparison.columns and mc in comparison.columns:
            comparison[f"auroc_diff_{tp}"] = comparison[mc] - comparison[dc]

    return comparison


# %%
def generate_interpretation_summary(results_df: pd.DataFrame) -> str:
    """Generate interpretation text for Discussion section."""
    lines = ["## Reference Time Sensitivity Analysis Summary\n"]

    for ref_type in ["matched", "discharge"]:
        subset = results_df.query("reference_type == @ref_type")
        if subset.empty:
            continue

        lines.append(f"\n### {ref_type.title()} Reference Time")

        for pw in PREDICTION_HORIZONS:
            pw_data = subset.query("horizon == @pw and dataset == 'ilsan_test'")
            if pw_data.empty:
                continue

            for model_name in pw_data["model_name"].unique():
                model_data = pw_data.query("model_name == @model_name").sort_values("time_window")
                aurocs = model_data.set_index("time_window")["auroc"]

                # Check monotonic improvement
                valid = aurocs.dropna()
                if len(valid) > 1:
                    is_monotonic = all(valid.iloc[i] <= valid.iloc[i + 1] for i in range(len(valid) - 1))
                    trend = "monotonically increasing" if is_monotonic else "non-monotonic"
                    lines.append(f"- {model_name} (PH={pw}h): AUROC trend is {trend} "
                                 f"(72h: {valid.iloc[0]:.3f} -> 0h: {valid.iloc[-1]:.3f})")

    return "\n".join(lines)


# %%
def main():
    logger.info("=" * 70)
    logger.info("Step 6: Sensitivity Analysis - Reference Time Comparison")
    logger.info("=" * 70)

    all_results = []

    for model_name in ALL_MODEL_CONFIGS:
        for pw in PREDICTION_HORIZONS:
            for dataset_name in DATASETS:
                logger.info(f"{model_name} PW={pw}h {dataset_name}...")
                try:
                    results = compute_online_metrics_both_references(model_name, pw, dataset_name)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"  Error: {e}")

    if not all_results:
        logger.error("No results generated!")
        return

    results_df = pd.DataFrame(all_results)

    # Save raw results
    out_path = RESULTS_OUT / "reference_time_comparison.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")

    # Comparison table
    comparison = create_comparison_table(results_df)
    comparison.to_csv(RESULTS_OUT / "reference_time_comparison_table.csv")

    # Plots
    plot_auroc_overlay(results_df)

    # FI ranking correlation (using Step 3 results if available)
    fi_path = RESULTS_DIR / "feature_importance" / "feature_importance_single_all.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        corr_df = compute_fi_ranking_correlation(fi_df)
        if not corr_df.empty:
            corr_df.to_csv(RESULTS_OUT / "fi_ranking_correlation.csv", index=False)
            logger.info(f"Saved FI ranking correlations")

    # Interpretation summary
    summary_text = generate_interpretation_summary(results_df)
    summary_path = RESULTS_OUT / "interpretation_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.info(f"Saved interpretation: {summary_path}")

    logger.info("\nStep 6 Complete!")
    return results_df


# %%
if __name__ == "__main__":
    results_df = main()
