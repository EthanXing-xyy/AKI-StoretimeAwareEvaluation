# %%
"""
Step C: MLE Trend Estimation for Clinical Faithfulness

C-1: Sen's Slope (non-parametric, paired with Mann-Kendall)
C-2: OLS Linear Regression with 95% CI
C-3: Optional logistic growth model comparison via AIC/BIC
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

from utils.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "clinical_faithfulness"


# %%
def logistic_growth(t, L, k, t0):
    """Logistic growth function: AUROC ~ L / (1 + exp(-k*(t - t0)))"""
    return L / (1.0 + np.exp(-k * (t - t0)))


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit OLS linear model: y = beta0 + beta1 * x."""
    X = sm.add_constant(x)
    try:
        model = sm.OLS(y, X).fit()
        beta1 = model.params[1]
        ci = model.conf_int(alpha=0.05)
        return {
            "linear_beta": beta1,
            "linear_beta_ci_lower": ci[1, 0],
            "linear_beta_ci_upper": ci[1, 1],
            "linear_r_squared": model.rsquared,
            "linear_p_value": model.pvalues[1],
            "linear_intercept": model.params[0],
            "linear_aic": model.aic,
            "linear_bic": model.bic,
        }
    except Exception as e:
        logger.warning(f"Linear fit failed: {e}")
        return {
            "linear_beta": np.nan, "linear_beta_ci_lower": np.nan,
            "linear_beta_ci_upper": np.nan, "linear_r_squared": np.nan,
            "linear_p_value": np.nan, "linear_intercept": np.nan,
            "linear_aic": np.nan, "linear_bic": np.nan,
        }


def fit_logistic(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit logistic growth model."""
    try:
        # Initial guesses: L=max(y), k=0.1, t0=median(x)
        p0 = [max(y) * 1.05, 0.05, np.median(x)]
        bounds = ([0.5, 0.001, -100], [1.0, 1.0, 200])
        popt, pcov = curve_fit(logistic_growth, x, y, p0=p0, bounds=bounds, maxfev=5000)
        y_pred = logistic_growth(x, *popt)
        residuals = y - y_pred
        n = len(y)
        k_params = 3
        rss = np.sum(residuals ** 2)
        # AIC/BIC for comparison
        log_lik = -n / 2 * np.log(2 * np.pi * rss / n) - n / 2
        aic = 2 * k_params - 2 * log_lik
        bic = k_params * np.log(n) - 2 * log_lik
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rss / ss_tot if ss_tot > 0 else 0

        return {
            "logistic_L": popt[0],
            "logistic_k": popt[1],
            "logistic_t0": popt[2],
            "logistic_r_squared": r2,
            "logistic_aic": aic,
            "logistic_bic": bic,
        }
    except Exception:
        return {
            "logistic_L": np.nan, "logistic_k": np.nan, "logistic_t0": np.nan,
            "logistic_r_squared": np.nan, "logistic_aic": np.nan, "logistic_bic": np.nan,
        }


# %%
def run_step_c(
    metrics_df: pd.DataFrame = None,
    mk_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Estimate trend slopes for all metric trajectories."""
    logger.info("=" * 60)
    logger.info("Step C: Trend Estimation (Sen's Slope + Linear MLE)")
    logger.info("=" * 60)

    if metrics_df is None:
        metrics_df = pd.read_csv(RESULTS_OUT / "step_a_timewindow_metrics.csv")
    if mk_df is None:
        mk_df = pd.read_csv(RESULTS_OUT / "step_b_mann_kendall_results.csv")

    # Metrics to analyze
    target_metrics = ["auroc", "auprc", "ppv_03", "ppv_04", "ppv_05", "ppv_youden"]

    all_rows = []
    groups = metrics_df.groupby(["model", "horizon", "dataset"])

    for (model, horizon, dataset), group_df in groups:
        # X = hours_before_onset (72, 60, ..., 0)
        # For linear model: negative beta1 means onset approach → metric increase
        group_sorted = group_df.sort_values("time_window", ascending=False).reset_index(drop=True)
        x = group_sorted["hours_before_onset"].values.astype(float)

        for metric_col in target_metrics:
            if metric_col not in group_sorted.columns:
                continue
            y = group_sorted[metric_col].values.astype(float)
            if np.all(np.isnan(y)):
                continue

            row = {
                "model": model,
                "model_type": group_sorted["model_type"].iloc[0],
                "horizon": horizon,
                "dataset": dataset,
                "dataset_display": group_sorted["dataset_display"].iloc[0],
                "metric": metric_col,
            }

            # Get Sen's slope from Mann-Kendall results
            mk_match = mk_df[
                (mk_df["model"] == model) &
                (mk_df["horizon"] == horizon) &
                (mk_df["dataset"] == dataset) &
                (mk_df["metric_column"] == metric_col)
            ]
            if not mk_match.empty:
                row["sens_slope"] = mk_match.iloc[0]["sens_slope"]
                row["mk_tau"] = mk_match.iloc[0]["tau"]
                row["mk_p_value"] = mk_match.iloc[0]["p_value"]
            else:
                row["sens_slope"] = np.nan
                row["mk_tau"] = np.nan
                row["mk_p_value"] = np.nan

            # Sen's slope CI (bootstrap-based approximation)
            # Use all pairwise slopes
            pairwise_slopes = []
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    if x[j] != x[i]:
                        pairwise_slopes.append((y[j] - y[i]) / (x[j] - x[i]))
            if pairwise_slopes:
                ps = np.array(pairwise_slopes)
                row["sens_slope_ci_lower"] = float(np.percentile(ps, 2.5))
                row["sens_slope_ci_upper"] = float(np.percentile(ps, 97.5))
            else:
                row["sens_slope_ci_lower"] = np.nan
                row["sens_slope_ci_upper"] = np.nan

            # C-2: Linear regression
            linear_result = fit_linear(x, y)
            row.update(linear_result)

            # C-3: Logistic growth (only for AUROC/AUPRC)
            if metric_col in ["auroc", "auprc"] and len(x) >= 4:
                logistic_result = fit_logistic(x, y)
                row.update(logistic_result)

                # Model selection
                if not np.isnan(logistic_result["logistic_bic"]) and not np.isnan(linear_result["linear_bic"]):
                    row["model_selected"] = "logistic" if logistic_result["logistic_bic"] < linear_result["linear_bic"] else "linear"
                else:
                    row["model_selected"] = "linear"
            else:
                row["model_selected"] = "linear"

            all_rows.append(row)

    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_OUT / "step_c_trend_estimation.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(result_df)} rows)")

    # Print interpretation for key combinations
    # Sen's slope > 0 = metric increases in ordered sequence (72h→0h) = increases toward onset
    # Linear β₁ < 0 = metric increases as hours_before_onset decreases = increases toward onset
    for _, r in result_df[result_df["metric"] == "auroc"].iterrows():
        if not np.isnan(r.get("linear_beta", np.nan)):
            beta = r["linear_beta"]
            direction = "increases" if beta < 0 else "decreases"
            logger.info(
                f"  {r['model']} (PW{r['horizon']}h, {r['dataset']}): "
                f"AUROC {direction} by {abs(beta):.6f}/hour toward onset "
                f"(β₁={beta:.6f}, tau={r['mk_tau']:.3f}, p={r['mk_p_value']:.4f})"
            )

    return result_df


# %%
if __name__ == "__main__":
    df = run_step_c()
    print(f"\nStep C complete: {len(df)} rows")
