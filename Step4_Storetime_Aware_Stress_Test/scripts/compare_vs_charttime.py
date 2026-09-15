"""Paired comparison of each availability variant against the original charttime-only results.

Reads Step 2 outputs (charttime reference) and Step 4 variant outputs; writes only under the
Step 4 comparison workspace. Interpretation rules are fixed in
configs/prespecified_rules.json (written before any stress-test inference).

  paired_discrimination_bootstrap.csv  AUROC/AUPRC per variant x mode x cohort x checkpoint x timepoint,
                                       paired 95% CI from 2000 stay resamples stratified by AKI/non-AKI
  threshold_metric_deltas.csv          frozen-threshold metrics (PPV, NNE, alert rate, ...) side by side
  patient_level_changes.csv            per-stay probability shifts and alert flips; identity check for
                                       stays whose inputs were untouched
  onset_subgroup_auroc.csv             creatinine-onset AKI or urine-onset AKI vs all non-AKI (t = -12, 0)
  faithfulness_flips.csv, ranking_changes.csv, alert_burden_deltas.csv
  prespecified_rule_evaluation.csv     primary endpoint: t = 0, matched, all eligible, per checkpoint

From the repository root after downstream processing for both variants:
    python Step4_Storetime_Aware_Stress_Test/scripts/compare_vs_charttime.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_patch as st  # noqa: E402

cm = st.cm
CM_ROOT = st.WORK_DIR / "Step2_Continuous_Monitoring"
TABLES = st.STRESS_ROOT / "comparison" / "tables"
VARIANTS = ["storetime", "onset_creatinine_only"]
MODES = ["matched", "discharge"]
TIMEPOINTS = [-72, -60, -48, -36, -24, -12, 0]
RULES = json.loads((st.CONFIG_ROOT / "prespecified_rules.json").read_text(encoding="utf-8"))
N_BOOT, SEED, MATERIAL_DELTA = 2000, 42, 0.02
# Eligibility may change when a stay's only pre-cutoff observations are not-yet-stored labs. Per the user's
# decision (configs/protocol_deviations.json), paired metrics use stay-timepoints eligible under both
# conditions and every eligibility change is reported in eligibility_changes.csv.
PAIR_COLUMNS = ["stay_id", "monitoring_timepoint", "y_true", "cutoff_dt", "threshold"]
PRED_COLUMNS = PAIR_COLUMNS + ["eligible", "ineligible_reason", "n_events_used", "y_prob", "alert"]


def root_of(condition: str) -> Path:
    return CM_ROOT if condition == "charttime" else st.STRESS_ROOT / condition


def load_predictions(condition: str, mode: str, horizon: int, model: str) -> pd.DataFrame:
    frame = pd.read_parquet(root_of(condition) / "predictions" / mode / f"horizon_{horizon}" / f"{model}.parquet", columns=PRED_COLUMNS)
    return frame.sort_values(["stay_id", "monitoring_timepoint"]).reset_index(drop=True)


def fast_auroc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUROC with average ranks for ties; works on (n,) or (B, n)."""
    ranks = rankdata(s, axis=-1)
    n1 = y.sum(-1)
    n0 = y.shape[-1] - n1
    return ((ranks * y).sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """sklearn.metrics.average_precision_score on one sample (distinct-threshold step sum)."""
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    cut = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[cut]
    precision = tps / (cut + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def self_check() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 500)
        s = np.round(rng.random(500), rng.integers(1, 4))  # include ties
        assert abs(fast_auroc(y.astype(bool), s) - roc_auc_score(y, s)) < 1e-12
        assert abs(average_precision(y, s) - average_precision_score(y, s)) < 1e-12


def paired_bootstrap(y: np.ndarray, s_ref: np.ndarray, s_new: np.ndarray) -> dict:
    """Same stratified resample for both conditions; percentile 95% CIs of each metric and of the difference."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = {
        "auroc_charttime": float(fast_auroc(y.astype(bool), s_ref)), "auroc_variant": float(fast_auroc(y.astype(bool), s_new)),
        "auprc_charttime": average_precision(y, s_ref), "auprc_variant": average_precision(y, s_new),
    }
    point["delta_auroc"] = point["auroc_variant"] - point["auroc_charttime"]
    point["delta_auprc"] = point["auprc_variant"] - point["auprc_charttime"]
    if len(pos) == 0 or len(neg) == 0:
        return point
    rng = np.random.default_rng(SEED)
    idx = np.hstack([rng.choice(pos, (N_BOOT, len(pos))), rng.choice(neg, (N_BOOT, len(neg)))])
    yb = y[idx]
    auc_ref, auc_new = fast_auroc(yb.astype(bool), s_ref[idx]), fast_auroc(yb.astype(bool), s_new[idx])
    ap_ref = np.array([average_precision(yb[b], s_ref[idx[b]]) for b in range(N_BOOT)])
    ap_new = np.array([average_precision(yb[b], s_new[idx[b]]) for b in range(N_BOOT)])
    for name, values in {"auroc_charttime": auc_ref, "auroc_variant": auc_new, "delta_auroc": auc_new - auc_ref,
                         "auprc_charttime": ap_ref, "auprc_variant": ap_new, "delta_auprc": ap_new - ap_ref}.items():
        point[f"{name}_ci_low"], point[f"{name}_ci_high"] = (float(v) for v in np.percentile(values, [2.5, 97.5]))
    return point


def removal_sets(variant: str) -> tuple[dict, pd.DataFrame]:
    """Stays whose slice lost >= 1 row, per (mode, timepoint); proven identical across all 15 runs."""
    runs = st.STRESS_ROOT / variant / "metrics" / "runs"
    reference = None
    for tag in sorted(p.name for p in runs.iterdir() if p.is_dir()):
        current = pd.read_csv(runs / tag / "availability" / "removed_rows_by_slice.csv").sort_values(["reference_mode", "monitoring_timepoint"]).reset_index(drop=True)
        if reference is not None and not current.equals(reference):
            raise SystemExit(f"{variant}: removed rows differ between runs ({tag})")
        reference = current
    removed = pd.read_parquet(runs / "h0_logistic_regression" / "availability" / "removed_rows_by_stay_feature.parquet")
    sets = {key: set(group.stay_id) for key, group in removed.groupby(["reference_mode", "monitoring_timepoint"])}
    return sets, reference.assign(availability=variant)


def compare_summary(name: str, keys: list[str], columns: list[str], variant: str, flag: str | None) -> pd.DataFrame:
    before = pd.read_csv(CM_ROOT / "summary" / name)
    after = pd.read_csv(st.STRESS_ROOT / variant / "summary" / name)
    merged = before[keys + columns].merge(after[keys + columns], on=keys, suffixes=("_charttime", "_variant"), validate="one_to_one")
    for column in columns:
        if pd.api.types.is_numeric_dtype(merged[f"{column}_charttime"]) and not pd.api.types.is_bool_dtype(merged[f"{column}_charttime"]):
            merged[f"delta_{column}"] = merged[f"{column}_variant"] - merged[f"{column}_charttime"]
    if flag:
        merged[f"{flag}_changed"] = merged[f"{flag}_charttime"] != merged[f"{flag}_variant"]
    return merged.assign(availability=variant)


def main() -> None:
    started = time.time()
    self_check()
    TABLES.mkdir(parents=True, exist_ok=True)
    _, monitoring = st._ORIGINAL["load_config"]()
    onset = pd.read_csv(st.ONSET_TABLE, usecols=["stay_id", "onset_source", "creatinine_defines_onset"])
    subgroups = {
        "creatinine_onset_aki": set(onset.loc[onset.creatinine_defines_onset.astype(str).str.lower().eq("true"), "stay_id"]),
        "urine_only_onset_aki": set(onset.loc[onset.onset_source.eq("urine"), "stay_id"]),
    }
    boot_rows, patient_rows, subgroup_rows, gate_rows, removal_tables, eligibility_changes = [], [], [], [], [], []
    for variant in VARIANTS:
        removed_sets, removal_table = removal_sets(variant)
        removal_tables.append(removal_table)
        for mode in MODES:
            for horizon in cm.HORIZONS:
                for model in cm.MODELS:
                    ref, new = load_predictions("charttime", mode, horizon, model), load_predictions(variant, mode, horizon, model)
                    paired = ref[PAIR_COLUMNS].equals(new[PAIR_COLUMNS])
                    gate_rows.append({"availability": variant, "check": f"pairing_{mode}_h{horizon}_{model}", "passed": paired})
                    if not paired:
                        raise SystemExit(f"{variant} {mode} h{horizon} {model}: stays/cutoffs/labels/thresholds differ from charttime; pairing broken")
                    changed = ref.eligible != new.eligible
                    if changed.any():
                        eligibility_changes.append(ref.loc[changed, ["stay_id", "monitoring_timepoint", "y_true", "cutoff_dt", "n_events_used"]].assign(
                            availability=variant, reference_mode=mode, prediction_horizon=horizon, model=model,
                            eligible_charttime=ref.eligible[changed], eligible_variant=new.eligible[changed],
                            ineligible_reason_variant=new.ineligible_reason[changed], n_events_used_variant=new.n_events_used[changed]))
                    tolerance = monitoring["static_reproduction_tolerance"]["ml" if model in cm.ML_MODELS else "deep"]
                    for timepoint in TIMEPOINTS:
                        mask = ref.monitoring_timepoint.eq(timepoint) & ref.eligible & new.eligible
                        r, n = ref[mask], new[mask]
                        y, s_ref, s_new = r.y_true.to_numpy(int), r.y_prob.to_numpy(float), n.y_prob.to_numpy(float)
                        ids = r.stay_id.to_numpy()
                        key = {"availability": variant, "reference_mode": mode, "prediction_horizon": horizon, "model": model, "monitoring_timepoint": timepoint}
                        sel = np.ones(len(ids), bool)
                        boot_rows.append({**key, "cohort": "all_eligible", "N": int(sel.sum()), "N_positive": int(y[sel].sum()), **paired_bootstrap(y[sel], s_ref[sel], s_new[sel])})

                        touched = np.isin(ids, list(removed_sets.get((mode, timepoint), set())))
                        delta = s_new - s_ref
                        alert_ref, alert_new = r.alert.to_numpy(int), n.alert.to_numpy(int)
                        for group, gsel in {"AKI": y == 1, "non-AKI": y == 0}.items():
                            d, t = delta[gsel], touched[gsel]
                            patient_rows.append({
                                **key, "group": group, "n_stays": int(gsel.sum()), "n_inputs_changed": int(t.sum()),
                                "n_prob_changed": int((np.abs(d) > tolerance).sum()),
                                "delta_prob_mean": float(d.mean()), "delta_prob_median": float(np.median(d)),
                                "delta_prob_mean_inputs_changed": float(d[t].mean()) if t.any() else np.nan,
                                "delta_prob_median_inputs_changed": float(np.median(d[t])) if t.any() else np.nan,
                                "delta_prob_q1_inputs_changed": float(np.percentile(d[t], 25)) if t.any() else np.nan,
                                "delta_prob_q3_inputs_changed": float(np.percentile(d[t], 75)) if t.any() else np.nan,
                                "n_alerts_charttime": int(alert_ref[gsel].sum()), "n_alerts_variant": int(alert_new[gsel].sum()),
                                "n_alerts_lost": int(((alert_ref == 1) & (alert_new == 0))[gsel].sum()),
                                "n_alerts_gained": int(((alert_ref == 0) & (alert_new == 1))[gsel].sum()),
                                "identity_max_abs_delta_untouched": float(np.abs(d[~t]).max()) if (~t).any() else np.nan,
                                "identity_tolerance": tolerance,
                                "identity_passed": bool((np.abs(d[~t]) <= tolerance).all()),
                            })
                        if timepoint in (-12, 0):
                            for subgroup, stay_set in subgroups.items():
                                sel = (y == 0) | ((y == 1) & np.isin(ids, list(stay_set)))
                                subgroup_rows.append({**key, "cohort": "all_eligible", "aki_subgroup": subgroup, "N_positive": int(y[sel].sum()), "N_negative": int((y[sel] == 0).sum()),
                                                      "n_aki_inputs_changed": int((touched & sel & (y == 1)).sum()), **paired_bootstrap(y[sel], s_ref[sel], s_new[sel])})
                    print(f"{variant} {mode} h{horizon} {model} done ({time.time() - started:.0f}s)", flush=True)

    boot = pd.DataFrame(boot_rows)
    boot["ci_excludes_zero"] = (boot.delta_auroc_ci_low > 0) | (boot.delta_auroc_ci_high < 0)
    boot["meets_material_rule"] = boot.ci_excludes_zero & boot.delta_auroc.abs().ge(MATERIAL_DELTA)
    boot.to_csv(TABLES / "paired_discrimination_bootstrap.csv", index=False)
    patients = pd.DataFrame(patient_rows)
    patients.to_csv(TABLES / "patient_level_changes.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(TABLES / "onset_subgroup_auroc.csv", index=False)
    pd.concat(removal_tables, ignore_index=True).to_csv(TABLES / "removed_rows_by_slice.csv", index=False)
    changes = pd.concat(eligibility_changes, ignore_index=True) if eligibility_changes else pd.DataFrame()
    changes.to_csv(TABLES / "eligibility_changes.csv", index=False)
    if len(changes):
        per_stay = changes.drop_duplicates(["availability", "reference_mode", "stay_id", "monitoring_timepoint"])
        stay_sets = changes.groupby(["availability", "reference_mode", "prediction_horizon", "model"]).apply(
            lambda g: tuple(sorted(zip(g.stay_id, g.monitoring_timepoint))), include_groups=False)
        same_across_checkpoints = stay_sets.groupby(level=["availability", "reference_mode"]).agg(lambda s: s.nunique() == 1 and len(s) == 15)
        gate_rows.append({"availability": "both", "check": "eligibility_changes_model_independent",
                          "passed": bool(same_across_checkpoints.all()),
                          "detail": f"{len(per_stay)} stay-timepoints: " + "; ".join(f"{r.availability}/{r.reference_mode} t={r.monitoring_timepoint} stay {r.stay_id} (y={r.y_true})" for r in per_stay.itertuples())})

    metric_keys = ["reference_mode", "cohort", "model", "prediction_horizon", "monitoring_timepoint"]
    metric_cols = ["N", "N_positive", "auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "nne", "alert_rate", "false_positive_rate", "f1", "tp", "fp", "tn", "fn", "brier_score"]
    base_metrics = pd.read_csv(CM_ROOT / "metrics" / "continuous_monitoring_metrics.csv")
    deltas = []
    for variant in VARIANTS:
        merged = base_metrics[metric_keys + metric_cols].merge(pd.read_csv(st.STRESS_ROOT / variant / "metrics" / "continuous_monitoring_metrics.csv")[metric_keys + metric_cols], on=metric_keys, suffixes=("_charttime", "_variant"), validate="one_to_one")
        # Pipeline metrics use each condition's own eligible set; N differences come only from eligibility_changes.csv.
        merged["N_differs"] = merged.N_charttime.ne(merged.N_variant) | merged.N_positive_charttime.ne(merged.N_positive_variant)
        for column in metric_cols[2:]:
            merged[f"delta_{column}"] = merged[f"{column}_variant"] - merged[f"{column}_charttime"]
        deltas.append(merged.assign(availability=variant))
    deltas = pd.concat(deltas, ignore_index=True)
    deltas.to_csv(TABLES / "threshold_metric_deltas.csv", index=False)
    check = boot.merge(deltas.loc[~deltas.N_differs, ["availability"] + metric_keys + ["auroc_charttime", "auroc_variant"]], on=["availability"] + metric_keys, suffixes=("", "_metrics"))
    auroc_agreement = float(max((check.auroc_charttime - check.auroc_charttime_metrics).abs().max(), (check.auroc_variant - check.auroc_variant_metrics).abs().max()))
    gate_rows.append({"availability": "both", "check": "bootstrap_point_auroc_equals_pipeline_metrics", "passed": auroc_agreement < 1e-9, "detail": auroc_agreement})

    flips, ranking, burden = [], [], []
    for variant in VARIANTS:
        flips.append(compare_summary("clinical_faithfulness_summary.csv", ["reference_mode", "prediction_horizon", "model"],
                                     ["auroc_t_minus_72", "auroc_t0", "delta_auroc_72_to_0", "auroc_tau", "auroc_p", "auroc_trend", "auroc_sens_slope_per_12h", "auprc_tau", "auprc_p", "clinically_faithful"], variant, "clinically_faithful"))
        ranking.append(compare_summary("static_vs_continuous_ranking.csv", ["ranking_scope", "model", "prediction_horizon"],
                                       ["static_auroc", "rank_static_auroc", "mean_continuous_auroc", "rank_mean_continuous_auroc", "auroc_t0", "auroc_tau", "clinically_faithful"], variant, "rank_mean_continuous_auroc"))
        burden.append(compare_summary("alert_burden_summary.csv", ["reference_mode", "prediction_horizon", "model"],
                                      ["frozen_ppv_t0", "frozen_nne_t0", "frozen_fpr_t0", "frozen_nne_min_over_timepoints", "frozen_nne_max_over_timepoints", "frozen_pct_aki_alerted_any_timepoint",
                                       "frozen_pct_non_aki_alerted_any_timepoint", "frozen_mean_alerts_per_non_aki_patient", "frozen_median_lead_time_h", "frozen_n_aki_with_lead_time",
                                       "upstream_thr05_nne_at_onset", "upstream_thr05_sensitivity_at_onset", "upstream_thr05_fpr_at_onset"], variant, None))
    pd.concat(flips, ignore_index=True).to_csv(TABLES / "faithfulness_flips.csv", index=False)
    ranking = pd.concat(ranking, ignore_index=True)
    ranking.to_csv(TABLES / "ranking_changes.csv", index=False)
    pd.concat(burden, ignore_index=True).to_csv(TABLES / "alert_burden_deltas.csv", index=False)

    primary = boot[boot.reference_mode.eq("matched") & boot.cohort.eq("all_eligible") & boot.monitoring_timepoint.eq(0)]
    evaluation = primary[["availability", "prediction_horizon", "model", "N", "N_positive", "auroc_charttime", "auroc_variant", "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high", "ci_excludes_zero", "meets_material_rule"]]
    evaluation.to_csv(TABLES / "prespecified_rule_evaluation.csv", index=False)
    identity = patients.groupby(["availability", "prediction_horizon", "model"]).agg(identity_passed=("identity_passed", "all"), max_abs_delta_untouched=("identity_max_abs_delta_untouched", "max")).reset_index()
    identity.to_csv(TABLES / "identity_check.csv", index=False)
    gate_rows.append({"availability": "both", "check": "identity_untouched_stays_within_tolerance", "passed": bool(identity.identity_passed.all()), "detail": float(identity.max_abs_delta_untouched.max())})
    pd.DataFrame(gate_rows).to_csv(TABLES / "pairing_and_identity_gates.csv", index=False)

    pd.set_option("display.width", 250)
    print(evaluation.round(4).to_string(index=False))
    print(identity.to_string(index=False))
    print(pd.concat(flips).query("clinically_faithful_changed")[["availability", "reference_mode", "prediction_horizon", "model", "auroc_tau_charttime", "auroc_tau_variant", "auroc_p_variant"]].to_string(index=False))
    print(f"rank changes: {int(ranking.rank_mean_continuous_auroc_changed.sum())}; runtime {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
