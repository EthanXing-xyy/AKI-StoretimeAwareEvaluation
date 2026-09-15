"""Paper-like clinical faithfulness and alert burden, reusing the pinned upstream analysis code.

From the repository root after full monitoring inference:
    python Step2_Continuous_Monitoring/scripts/run_upstream_analyses.py

Reused unchanged (functions are called; only data loaders and output folders are redirected):
  experiments/clinical_faithfulness/step_a_compute_metrics.py  (_process_one_combination, bootstrap CIs)
  experiments/clinical_faithfulness/step_b_mann_kendall.py     (run_step_b, pymannkendall original_test)
  experiments/clinical_faithfulness/step_c_trend_estimation.py (run_step_c, Sen's slope CI, OLS, logistic)
  experiments/clinical_faithfulness/step_d_figures.py          (run_step_d)
  experiments/step7_alert_burden.py                            (threshold sweep, alert frequency, NNE, lead time, figures)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import cm_adapter as cm  # noqa: E402

LOGGER = logging.getLogger("continuous_monitoring.upstream")
STEP_MODULES = [
    "utils.data_loader",
    "experiments.clinical_faithfulness.step_a_compute_metrics",
    "experiments.clinical_faithfulness.step_b_mann_kendall",
    "experiments.clinical_faithfulness.step_c_trend_estimation",
    "experiments.clinical_faithfulness.step_d_figures",
    "experiments.step7_alert_burden",
]


def load_online_frames(predictions_root: Path, mode: str) -> tuple[dict, dict]:
    """Eligible monitoring predictions in the upstream online-prediction column layout."""
    frames, thresholds = {}, {}
    for horizon in cm.HORIZONS:
        for model in cm.MODELS:
            pred = pd.read_parquet(predictions_root / mode / f"horizon_{horizon}" / f"{model}.parquet")
            pred = pred[pred.eligible]
            name = cm.UPSTREAM_MODEL_NAMES[model]
            frames[(name, horizon)] = pd.DataFrame({
                "visit_id": pred.stay_id.to_numpy(),
                "label": pred.y_true.to_numpy(int),
                "pred_proba": pred.y_prob.to_numpy(float),
                "timestamp": pred.cutoff_hours_since_intime.to_numpy(float),
                "time_to_event": pred.time_to_event.to_numpy(float),
            })
            thresholds[(name, horizon)] = float(pred.threshold.iloc[0])
    return frames, thresholds


def faithfulness_summary(metrics_df: pd.DataFrame, mk_df: pd.DataFrame, trend_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in metrics_df.groupby(["model", "horizon"]):
        ordered = group.sort_values("time_window", ascending=False)  # -72 h ... 0 h
        row = {
            "reference_mode": mode, "model": model, "prediction_horizon": int(horizon),
            "n_timepoints": len(ordered),
            "auroc_t_minus_72": ordered.auroc.iloc[0], "auroc_t0": ordered.auroc.iloc[-1],
            "delta_auroc_72_to_0": ordered.auroc.iloc[-1] - ordered.auroc.iloc[0],
            "auroc_min": ordered.auroc.min(), "auroc_max": ordered.auroc.max(),
        }
        for column in ["auroc", "auprc", "ppv_youden", "sensitivity_youden"]:
            mk = mk_df[(mk_df.model == model) & (mk_df.horizon == horizon) & (mk_df.metric_column == column)]
            prefix = column.replace("_youden", "_frozen_threshold")
            if mk.empty:
                row.update({f"{prefix}_tau": np.nan, f"{prefix}_p": np.nan, f"{prefix}_trend": "not_computed", f"{prefix}_sens_slope_per_12h": np.nan})
                continue
            mk = mk.iloc[0]
            row.update({f"{prefix}_tau": mk.tau, f"{prefix}_p": mk.p_value, f"{prefix}_trend": mk.trend_direction, f"{prefix}_sens_slope_per_12h": mk.get("sens_slope", np.nan)})
        trend = trend_df[(trend_df.model == model) & (trend_df.horizon == horizon) & (trend_df.metric == "auroc")]
        row["auroc_linear_beta_per_hour"] = trend.linear_beta.iloc[0] if not trend.empty else np.nan
        row["auroc_linear_p"] = trend.linear_p_value.iloc[0] if not trend.empty else np.nan
        # Paper definition: statistically significant monotonic increasing AUROC trend toward onset.
        row["clinically_faithful"] = bool(row["auroc_trend"] == "increasing" and row["auroc_p"] < 0.05)
        rows.append(row)
    return pd.DataFrame(rows)


def alert_burden_summary(sweep: pd.DataFrame, lead: pd.DataFrame, frequency: pd.DataFrame, cm_metrics: pd.DataFrame, predictions_root: Path, mode: str) -> pd.DataFrame:
    rows = []
    display_to_model = {cm.UPSTREAM_MODEL_NAMES[m]: m for m in cm.MODELS}
    short_to_display = {"TF": "ITE Transformer", "LSTM": "LSTM-Attention", "CNN": "Masked CNN", "XGB": "XGBoost", "LR": "Logistic Regression"}
    for horizon in cm.HORIZONS:
        for model in cm.MODELS:
            short = next(k for k, v in short_to_display.items() if display_to_model[v] == model)
            pred = pd.read_parquet(predictions_root / mode / f"horizon_{horizon}" / f"{model}.parquet")
            patients = pred[pred.n_eligible_timepoints > 0].drop_duplicates("stay_id")
            aki, non_aki = patients[patients.y_true.eq(1)], patients[patients.y_true.eq(0)]
            frozen = cm_metrics[(cm_metrics.reference_mode == mode) & (cm_metrics.cohort == "all_eligible") & (cm_metrics.model == model) & (cm_metrics.prediction_horizon == horizon)]
            at_05 = sweep[(sweep.model == short) & (sweep.horizon == horizon) & (sweep.threshold == 0.5)]
            onset_05 = at_05[at_05.time_window == 0]
            lead_05 = lead[(lead.model == short) & (lead.horizon == horizon) & (lead.threshold == 0.5)]
            freq_05 = frequency[(frequency.model == short) & (frequency.horizon == horizon) & (frequency.threshold == 0.5)]
            rows.append({
                "reference_mode": mode, "model": model, "prediction_horizon": horizon,
                "frozen_threshold": float(pred.threshold.iloc[0]),
                "frozen_nne_t0": frozen.loc[frozen.monitoring_timepoint == 0, "nne"].iloc[0],
                "frozen_ppv_t0": frozen.loc[frozen.monitoring_timepoint == 0, "ppv"].iloc[0],
                "frozen_fpr_t0": frozen.loc[frozen.monitoring_timepoint == 0, "false_positive_rate"].iloc[0],
                "frozen_nne_min_over_timepoints": frozen.nne.min(), "frozen_nne_max_over_timepoints": frozen.nne.max(),
                "frozen_timepoints_nne_undefined": int(frozen.nne.isna().sum()),
                "frozen_pct_aki_alerted_any_timepoint": 100 * aki.ever_alerted.mean() if len(aki) else np.nan,
                "frozen_pct_non_aki_alerted_any_timepoint": 100 * non_aki.ever_alerted.mean() if len(non_aki) else np.nan,
                "frozen_mean_alerts_per_non_aki_patient": non_aki.n_alerts_patient.mean() if len(non_aki) else np.nan,
                "frozen_median_lead_time_h": aki.lead_time_hours.median(),
                "frozen_n_aki_with_lead_time": int(aki.lead_time_hours.notna().sum()),
                "upstream_thr05_nne_at_onset": onset_05.nne.iloc[0] if len(onset_05) else np.nan,
                "upstream_thr05_sensitivity_at_onset": onset_05.sensitivity.iloc[0] if len(onset_05) else np.nan,
                "upstream_thr05_fpr_at_onset": onset_05.fpr.iloc[0] if len(onset_05) else np.nan,
                "upstream_thr05_nne_min_over_windows": at_05.nne.replace(np.inf, np.nan).min() if len(at_05) else np.nan,
                "upstream_thr05_nne_max_over_windows": at_05.nne.replace(np.inf, np.nan).max() if len(at_05) else np.nan,
                "upstream_thr05_windows_nne_infinite": int(np.isinf(at_05.nne).sum()) if len(at_05) else 0,
                "upstream_thr05_pct_aki_alerted": freq_05.pct_patients_alerted_aki.iloc[0] if len(freq_05) else np.nan,
                "upstream_thr05_pct_non_aki_alerted": freq_05.pct_patients_alerted_nonaki.iloc[0] if len(freq_05) else np.nan,
                "upstream_thr05_median_lead_time_h": lead_05.median_lead_time_hours.iloc[0] if len(lead_05) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    paths, monitoring = cm.load_config()
    cm.enforce_frozen_policy(monitoring)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm.setup_logging(paths["logs_root"] / f"run_upstream_analyses_{stamp}.log")
    base_results = paths["metrics_root"] / "upstream_analysis"
    base_figures = paths["figures_root"] / "upstream_analysis"
    up = cm.import_upstream(paths["paper_repo_root"], base_results / "_import", base_figures / "_import", STEP_MODULES)
    step_a, step_b, step_c = up.step_a_compute_metrics, up.step_b_mann_kendall, up.step_c_trend_estimation
    step_d, step7, data_loader = up.step_d_figures, up.step7_alert_burden, up.data_loader
    cm_metrics = pd.read_csv(paths["metrics_root"] / "continuous_monitoring_metrics.csv")
    dataset_info = {"display_name": cm.DATASET_DISPLAY, "hospital": "mimic-iv", "type": "adapted_test"}

    faithfulness, burden = [], []
    for mode in monitoring["reference_time"]["modes"]:
        results_dir, figures_dir = base_results / mode, base_figures / mode
        for sub in ("clinical_faithfulness", "alert_burden"):
            (results_dir / sub).mkdir(parents=True, exist_ok=True)
            (figures_dir / sub).mkdir(parents=True, exist_ok=True)
        frames, thresholds = load_online_frames(paths["predictions_root"], mode)

        # ---- Clinical faithfulness (steps A-D) ----
        step_a.RESULTS_OUT = step_b.RESULTS_OUT = step_c.RESULTS_OUT = results_dir / "clinical_faithfulness"
        step_d.RESULTS_IN, step_d.FIGURES_OUT = results_dir / "clinical_faithfulness", figures_dir / "clinical_faithfulness"
        step_a.load_online_predictions = lambda model_name, pw, dataset: frames[(model_name, pw)].copy()
        rows = []
        for model_name in step_a.ALL_MODEL_CONFIGS:
            model_type = "DL" if model_name in step_a.DL_MODEL_CONFIGS else "Baseline"
            for horizon in cm.HORIZONS:
                produced = step_a._process_one_combination(model_name, horizon, cm.DATASET_KEY, dataset_info, model_type, thresholds, {})
                if len(produced) != len(monitoring["time_points_hours"]):
                    LOGGER.warning("%s %s h=%d: upstream step A produced %d/7 time windows (n<30 or single class skipped)", mode, model_name, horizon, len(produced))
                rows.extend(produced)
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(step_a.RESULTS_OUT / "step_a_timewindow_metrics.csv", index=False)
        mk_df = step_b.run_step_b(metrics_df)
        trend_df = step_c.run_step_c(metrics_df, mk_df)
        step_d.DATASET_ORDER = [cm.DATASET_KEY]
        step_d.DATASET_DISPLAY = {cm.DATASET_KEY: cm.DATASET_DISPLAY}
        step_d.DATASET_PANEL_LABELS = {cm.DATASET_KEY: "A"}
        step_d.run_step_d()
        faithfulness.append(faithfulness_summary(metrics_df, mk_df, trend_df, mode))

        # ---- Alert burden (step 7) ----
        step7.RESULTS_OUT, step7.FIGURES_OUT = results_dir / "alert_burden", figures_dir / "alert_burden"
        step7.DATASET_ORDER = [cm.DATASET_KEY]
        step7.DATASET_DISPLAY = {cm.DATASET_KEY: cm.DATASET_DISPLAY}
        step7.load_and_prepare_online = lambda model_name, pw, dataset_name: (
            lambda df: df[(df.time_to_event >= 0) & (df.time_to_event <= 72)].copy()
        )(frames[(model_name, pw)])
        # Record the frozen validation threshold in step 7's "youden_threshold" column; never refit it on test data.
        data_loader.load_predictions = lambda model_name, pw, dataset: pd.DataFrame({"label": [model_name], "pred_proba": [pw]})
        step7.calculate_youden_threshold = lambda labels, probs: thresholds[(labels[0], int(probs[0]))]
        sweep = step7.run_threshold_sweep()
        frequency = step7.run_alert_frequency()
        step7.compute_nne_matrix(sweep)
        lead = step7.run_lead_time_analysis()
        step7.plot_threshold_sweep(sweep)
        step7.plot_alert_burden_heatmap(sweep)
        step7.suggest_optimal_thresholds(sweep, lead)
        sweep.merge(lead[["model", "horizon", "dataset", "threshold", "median_lead_time_hours", "mean_lead_time_hours"]], on=["model", "horizon", "dataset", "threshold"], how="left").to_csv(step7.RESULTS_OUT / "alert_burden_combined.csv", index=False)
        burden.append(alert_burden_summary(sweep, lead, frequency, cm_metrics, paths["predictions_root"], mode))

    paths["summary_root"].mkdir(parents=True, exist_ok=True)
    pd.concat(faithfulness, ignore_index=True).to_csv(paths["summary_root"] / "clinical_faithfulness_summary.csv", index=False)
    pd.concat(burden, ignore_index=True).to_csv(paths["summary_root"] / "alert_burden_summary.csv", index=False)
    LOGGER.info("wrote clinical_faithfulness_summary.csv and alert_burden_summary.csv")


if __name__ == "__main__":
    main()
