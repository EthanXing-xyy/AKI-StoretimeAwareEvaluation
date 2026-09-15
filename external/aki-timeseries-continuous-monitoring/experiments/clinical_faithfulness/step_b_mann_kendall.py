# %%
"""
Step B: Mann-Kendall Trend Test for Clinical Faithfulness

Tests whether metrics (AUROC, AUPRC, PPV) show statistically significant
monotonic increasing trends as time approaches AKI onset.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import pymannkendall as mk

from utils.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "clinical_faithfulness"


# %%
def run_mk_test(values: np.ndarray) -> dict:
    """
    Run Mann-Kendall test on a time-ordered series.
    Values should be ordered from furthest (72h) to closest (0h) to onset.
    Positive tau = increasing as onset approaches.
    """
    if len(values) < 4 or np.all(np.isnan(values)):
        return {"tau": np.nan, "p_value": np.nan, "trend_direction": "insufficient_data", "significant": False}

    try:
        result = mk.original_test(values)
        return {
            "tau": result.Tau,
            "p_value": result.p,
            "trend_direction": result.trend,  # 'increasing', 'decreasing', 'no trend'
            "significant": result.p < 0.05,
            "z_score": result.z,
            "s_statistic": result.s,
            "var_s": result.var_s,
            "sens_slope": result.slope,
            "sens_intercept": result.intercept,
        }
    except Exception as e:
        logger.warning(f"MK test failed: {e}")
        return {"tau": np.nan, "p_value": np.nan, "trend_direction": "error", "significant": False}


# %%
def run_step_b(metrics_df: pd.DataFrame = None) -> pd.DataFrame:
    """Run Mann-Kendall test on all metric trajectories."""
    logger.info("=" * 60)
    logger.info("Step B: Mann-Kendall Trend Test")
    logger.info("=" * 60)

    if metrics_df is None:
        path = RESULTS_OUT / "step_a_timewindow_metrics.csv"
        metrics_df = pd.read_csv(path)
        logger.info(f"Loaded metrics from {path}")

    # Define metrics to test
    test_metrics = [
        ("auroc", "AUROC", None),
        ("auprc", "AUPRC", None),
        ("ppv_03", "PPV", 0.3),
        ("ppv_04", "PPV", 0.4),
        ("ppv_05", "PPV", 0.5),
        ("ppv_youden", "PPV", "youden"),
        ("sensitivity_03", "Sensitivity", 0.3),
        ("sensitivity_04", "Sensitivity", 0.4),
        ("sensitivity_05", "Sensitivity", 0.5),
        ("sensitivity_youden", "Sensitivity", "youden"),
    ]

    all_rows = []
    groups = metrics_df.groupby(["model", "horizon", "dataset"])

    for (model, horizon, dataset), group_df in groups:
        # Sort by time_window ascending (72 → 60 → ... → 0)
        # This way index 0 = furthest, last = closest to onset
        group_sorted = group_df.sort_values("time_window", ascending=False).reset_index(drop=True)

        for col_name, metric_label, threshold in test_metrics:
            if col_name not in group_sorted.columns:
                continue
            values = group_sorted[col_name].values
            if np.all(np.isnan(values)):
                continue

            result = run_mk_test(values)
            result["model"] = model
            result["model_type"] = group_sorted["model_type"].iloc[0]
            result["horizon"] = horizon
            result["dataset"] = dataset
            result["dataset_display"] = group_sorted["dataset_display"].iloc[0]
            result["metric"] = metric_label
            result["metric_column"] = col_name
            result["threshold"] = threshold
            result["n_timepoints"] = len(values)
            all_rows.append(result)

    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_OUT / "step_b_mann_kendall_results.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(result_df)} rows)")

    # Summary
    for metric_col in ["auroc", "auprc", "ppv_03", "ppv_05"]:
        subset = result_df[result_df["metric_column"] == metric_col]
        n_total = len(subset)
        n_sig_inc = ((subset["significant"]) & (subset["trend_direction"] == "increasing")).sum()
        tau_range = f"{subset['tau'].min():.3f}–{subset['tau'].max():.3f}"
        logger.info(f"  {metric_col}: {n_sig_inc}/{n_total} significant increasing (tau range: {tau_range})")

    return result_df


# %%
if __name__ == "__main__":
    df = run_step_b()
    print(f"\nStep B complete: {len(df)} rows")
    # Print summary table for AUROC
    auroc_df = df[df["metric_column"] == "auroc"][["model", "horizon", "dataset", "tau", "p_value", "trend_direction", "significant"]]
    print("\nAUROC Mann-Kendall Summary:")
    print(auroc_df.to_string(index=False))
