"""Retrospective continuous-monitoring (simulated online) inference on the frozen baselines.

From the repository root:
    python Step2_Continuous_Monitoring/scripts/run_monitoring.py --stage dry-run
    python Step2_Continuous_Monitoring/scripts/run_monitoring.py --stage full

For every selected (model, prediction horizon) checkpoint and every monitoring time point
t in [-72, -60, ..., 0] h relative to the reference time, each held-out test stay is
re-encoded from events with charttime <= reference + t and scored once.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import cm_adapter as cm  # noqa: E402

LOGGER = logging.getLogger("continuous_monitoring.run")

PREDICTION_COLUMNS = [
    "model", "prediction_horizon", "reference_mode", "monitoring_timepoint", "time_to_event",
    "subject_id", "hadm_id", "stay_id", "y_true", "eligible", "ineligible_reason",
    "reference_dt", "cutoff_dt", "cutoff_hours_since_intime", "n_events_used", "last_event_dt",
    "y_prob", "threshold", "y_pred", "alert",
    "n_eligible_timepoints", "n_alerts_patient", "ever_alerted", "first_alert_timepoint", "lead_time_hours",
]


def stage_layout(paths: dict[str, Path], stage: str, run_tag: str | None = None) -> dict[str, Path]:
    sub = "dry_run" if stage == "dry-run" else ""
    # Per-(horizon, model) full runs keep their tables apart; merge_monitoring_runs.py combines them.
    table_sub = Path(sub) / "runs" / run_tag if run_tag else Path(sub)
    layout = {
        "predictions": paths["predictions_root"] / sub,
        "metrics": paths["metrics_root"] / table_sub,
        "summary": paths["summary_root"] / table_sub,
        "figures": paths["figures_root"] / sub,
        "logs": paths["logs_root"],
    }
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def assemble_predictions(model: cm.FrozenModel, reference: pd.DataFrame, slices: list[tuple[int, pd.DataFrame, dict]]) -> pd.DataFrame:
    frames = []
    for timepoint, audit, probabilities in slices:
        frame = reference[["subject_id", "hadm_id", "stay_id", "label", "reference_dt", "reference_mode"]].merge(
            audit[["stay_id", "eligible", "ineligible_reason", "cutoff_dt", "cutoff_hours_since_intime", "n_events_in_slice", "last_event_dt"]],
            on="stay_id", how="left", validate="one_to_one",
        )
        frame["monitoring_timepoint"] = timepoint
        frame["y_prob"] = frame.stay_id.map(probabilities).astype(float)
        frames.append(frame)
    pred = pd.concat(frames, ignore_index=True).rename(columns={"label": "y_true", "n_events_in_slice": "n_events_used"})
    pred["model"] = model.model_name
    pred["prediction_horizon"] = model.horizon
    pred["time_to_event"] = -pred.monitoring_timepoint  # upstream naming: hours remaining to reference time
    pred["threshold"] = model.threshold
    pred["y_pred"] = pd.array(np.where(pred.eligible, (pred.y_prob >= model.threshold).astype(int), 0), dtype="Int64")
    pred.loc[~pred.eligible, "y_pred"] = pd.NA
    pred["alert"] = pred.y_pred
    if (pred.eligible != pred.y_prob.notna()).any():
        raise cm.MonitoringAbort(f"{model.artifact['artifact_id']}: eligibility and predictions disagree")

    eligible = pred[pred.eligible]
    alerts = eligible[eligible.alert.eq(1)]
    first_alert = alerts.groupby("stay_id").monitoring_timepoint.min()
    pred["n_eligible_timepoints"] = pred.stay_id.map(eligible.groupby("stay_id").size()).fillna(0).astype(int)
    pred["n_alerts_patient"] = pred.stay_id.map(alerts.groupby("stay_id").size()).fillna(0).astype(int)
    pred["first_alert_timepoint"] = pred.stay_id.map(first_alert)
    pred["ever_alerted"] = pred.first_alert_timepoint.notna()
    # upstream timeseries/clinical_utility.py::calculate_lead_times: first alert strictly before onset
    pred["lead_time_hours"] = np.where(pred.y_true.eq(1) & (pred.first_alert_timepoint < 0), -pred.first_alert_timepoint, np.nan)
    return pred[PREDICTION_COLUMNS]


def metric_rows(pred: pd.DataFrame, timepoints: list[int]) -> list[dict]:
    rows = []
    subset = pred[pred.eligible]
    first = pred.iloc[0]
    for timepoint in timepoints:
        group = subset[subset.monitoring_timepoint.eq(timepoint)]
        rows.append({
            "reference_mode": first.reference_mode, "cohort": "all_eligible", "model": first.model,
            "prediction_horizon": int(first.prediction_horizon), "monitoring_timepoint": timepoint,
            **cm.threshold_metrics(group.y_true.to_numpy(int), group.y_prob.to_numpy(float), float(first.threshold)),
        })
    return rows


def slice_summary(audit: pd.DataFrame, reference: pd.DataFrame, mode: str, horizon: int, timepoint: int) -> dict:
    audit = audit.merge(reference[["stay_id", "label"]], on="stay_id")
    eligible = audit[audit.eligible]
    reasons = audit.ineligible_reason.value_counts()
    offsets = eligible.get("last_event_offset_from_reference_h", pd.Series(dtype=float))
    after_next = int(eligible["events_after_next_timepoint"].sum()) if "events_after_next_timepoint" in eligible else 0
    minus_cutoff = ((eligible.last_event_dt - eligible.cutoff_dt).dt.total_seconds() / 3600).max()
    row = {
        "reference_mode": mode, "prediction_horizon": horizon, "monitoring_timepoint": timepoint,
        "n_patients": len(audit), "n_patients_eligible": len(eligible),
        "n_aki_eligible": int(eligible.label.sum()), "n_non_aki_eligible": int((eligible.label == 0).sum()),
        "n_ineligible_cutoff_before_icu_admission": int(reasons.get("cutoff_before_icu_admission", 0)),
        "n_ineligible_no_observation": int(reasons.get("no_timeseries_observation_before_cutoff", 0)),
        "n_events_before_cutoff": int(eligible.n_events_in_slice.sum()),
        "max_event_time_used_h_rel_reference": float(offsets.max()) if len(offsets) else np.nan,
        "max_event_minus_cutoff_h": float(minus_cutoff) if pd.notna(minus_cutoff) else np.nan,
        "events_after_next_timepoint": after_next,
        "source_events_after_cutoff_excluded": int(eligible.source_events_after_cutoff.sum()),
        "irregular_tokens_after_cutoff": int(eligible.irregular_tokens_after_cutoff.sum()),
        "cnn_bins_after_cutoff": int(eligible.cnn_bins_after_cutoff.sum()),
        "irregular_last_token_inconsistent": int((~eligible.irregular_last_token_consistent.astype(bool)).sum()),
        "cnn_cell_inconsistent": int((~eligible.cnn_cells_consistent.astype(bool)).sum()),
        "future_events_detected": int(audit.future_events_detected.sum()),
    }
    row["passed"] = bool(
        row["future_events_detected"] == 0 and row["events_after_next_timepoint"] == 0
        and (np.isnan(row["max_event_time_used_h_rel_reference"]) or row["max_event_time_used_h_rel_reference"] <= timepoint)
        and row["irregular_last_token_inconsistent"] == 0 and row["cnn_cell_inconsistent"] == 0
    )
    return row


def markdown_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "NaN" if np.isnan(value) else float_format.format(value)
        return str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["dry-run", "full"], required=True)
    parser.add_argument("--models", nargs="+", choices=cm.MODELS, help="full stage only: subset of models for this process")
    parser.add_argument("--horizons", nargs="+", type=int, choices=cm.HORIZONS, help="full stage only: subset of horizons for this process")
    args = parser.parse_args()

    paths, monitoring = cm.load_config()
    cm.enforce_frozen_policy(monitoring)
    run_tag = None
    if args.stage == "dry-run":
        if args.models or args.horizons:
            parser.error("--models/--horizons are fixed by monitoring.yaml dry_run for the dry-run stage")
        models, horizons = [monitoring["dry_run"]["model"]], [int(monitoring["dry_run"]["horizon"])]
    else:
        models = [m for m in monitoring["models"] if not args.models or m in args.models]
        horizons = [h for h in monitoring["prediction_horizons"] if not args.horizons or h in args.horizons]
        if args.models or args.horizons:
            run_tag = f"h{'-'.join(map(str, horizons))}_{'-'.join(models)}"
    layout = stage_layout(paths, args.stage, run_tag)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm.setup_logging(layout["logs"] / f"run_monitoring_{args.stage.replace('-', '_')}{'_' + run_tag if run_tag else ''}_{stamp}.log")
    started = time.time()
    modes = list(monitoring["reference_time"]["modes"])
    timepoints = list(monitoring["time_points_hours"])
    min_obs = int(monitoring["eligibility"]["min_timeseries_observations"])
    LOGGER.info("stage=%s models=%s horizons=%s modes=%s timepoints=%s", args.stage, models, horizons, modes, timepoints)

    checks: list[dict] = []

    def record(name: str, passed: bool, detail, fatal: bool = True) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
        LOGGER.info("[%s] %s: %s", "PASS" if passed else "FAIL", name, detail)
        if fatal and not passed:
            raise cm.MonitoringAbort(f"check failed: {name}: {detail}")

    # 1. Artifacts, strictly via manifest.
    artifacts, selection = cm.select_artifacts(monitoring, models, horizons)
    selection.to_csv(layout["summary"] / "artifact_selection.csv", index=False)
    record("manifest_formal_artifacts_unique", True, f"15 formal full artifacts; selected {len(artifacts)}")
    record("no_smoke_checkpoint_selected", all(a["scope"] == "full" and a["used_for_final_evaluation"].lower() == "true" and "/smoke/" not in a["checkpoint_path"] for a in artifacts),
           f"excluded non-final manifest rows={int(selection.selection.eq('excluded_smoke').sum())}")

    # 2. Frozen test cohort.
    master, test, cohort_checks = cm.load_and_verify_cohort(paths, monitoring, artifacts)
    for item in cohort_checks:
        record(item["check"], item["passed"], item["detail"])

    # 3. Baseline encoder and upstream reference-time matching.
    baseline = cm.import_baseline(paths["baseline_root"])
    upstream = cm.import_upstream(paths["paper_repo_root"], layout["metrics"] / "upstream_analysis", layout["figures"] / "upstream_analysis", ["experiments.step1_reference_time_matching"])
    references: dict[str, pd.DataFrame] = {}
    ref_dir = layout["predictions"] / "reference_times"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        reference, matching = cm.compute_reference_times(master, test, mode, upstream.step1_reference_time_matching, monitoring)
        references[mode] = reference
        reference.to_parquet(ref_dir / f"{mode}.parquet", index=False)
        if matching is not None:
            matching.to_csv(ref_dir / "matching_full_included_cohort.csv", index=False)
            aki_ratios = upstream.step1_reference_time_matching.compute_onset_ratios(master).to_numpy()
            ks = stats.ks_2samp(aki_ratios, matching.effective_ratio.to_numpy())
            pd.DataFrame([{"hospital": "mimic-iv", "n_aki": len(aki_ratios), "n_non_aki": len(matching), "aki_ratio_median": float(np.median(aki_ratios)), "non_aki_ratio_median": float(matching.effective_ratio.median()), "ks_statistic": ks.statistic, "ks_pvalue": ks.pvalue}]).to_csv(ref_dir / "ks_test_summary.csv", index=False)
            qc_dir = layout["figures"] / "reference_time"
            qc_dir.mkdir(parents=True, exist_ok=True)
            upstream.step1_reference_time_matching.create_qc_histogram(aki_ratios, matching.effective_ratio.to_numpy(), "mimic-iv", ks.statistic, ks.pvalue, qc_dir / "qc_histogram_mimic-iv.pdf")
        LOGGER.info("reference mode %s: median reference hours since intime (non-AKI)=%.1f", mode, reference.loc[reference.label.eq(0), "reference_hours_since_intime"].median())

    # 4. Event superset for test stays (ICU intime..outtime); slices are cut per time point.
    events = cm.extract_stay_events(paths["baseline_data_root"] / "interim", test)
    events_by_stay = {int(key): group for key, group in events.groupby("stay_id", sort=False)}
    empty_events = events.iloc[0:0]
    LOGGER.info("event superset: %d events for %d stays", len(events), len(events_by_stay))

    integrity, static_rows, encoding_rows, slice_rows, metric_records, audit_frames = [], [], [], [], [], []
    for horizon in horizons:
        horizon_artifacts = [a for a in artifacts if int(a["prediction_window_hours"]) == horizon]
        frozen = {a["model"]: cm.FrozenModel(a, baseline, paths, monitoring["device"]) for a in horizon_artifacts}
        for model in frozen.values():
            LOGGER.info("loaded %s threshold=%.6g fingerprint=%s", model.artifact["artifact_id"], model.threshold, model.fingerprint_before[:16])
        processed_dir = paths["baseline_processed_data_root"] / "mimiciv" / "full" / f"horizon_{horizon}"
        scale = pd.read_parquet(processed_dir / "scaling_info.parquet")
        record(f"h{horizon}_scaling_tables_frozen", all(m.scale.equals(scale) for m in frozen.values()), "model-dir scaling_info == processed train-only scaling_info")
        if horizon == horizons[0]:
            self_test = cm.validator_self_test(baseline, scale)
            record("leakage_validator_negative_control", self_test["passed"], self_test)

        # 5. Static reproduction gate: baseline cutoff must reproduce saved inputs and predictions.
        processed_master = pd.read_parquet(processed_dir / "master.parquet")
        processed_test = processed_master[processed_master.split.eq("test")].reset_index(drop=True)
        expected_cutoff = test.aki_dt.where(test.label.eq(1) & test.aki_dt.notna(), test.outtime)
        expected_cutoff = expected_cutoff.where(test.label.eq(0), test.aki_dt - pd.Timedelta(hours=horizon))
        record(f"h{horizon}_baseline_cutoff_definition", pd.to_datetime(processed_test.cutoff_dt).equals(pd.to_datetime(expected_cutoff)), "cases: onset - horizon; controls: ICU outtime")
        static_inputs, static_audit = cm.build_timepoint_inputs(test, processed_test.cutoff_dt, events_by_stay, empty_events, scale, baseline, apply_eligibility=False, min_observations=min_obs)
        record(f"h{horizon}_static_slice_no_future_events", int(static_audit.future_events_detected.sum()) == 0, f"future={int(static_audit.future_events_detected.sum())}")
        encoding = cm.compare_with_baseline_inputs(test.stay_id.astype(int).tolist(), static_inputs, processed_dir, list(baseline.config.ALL_FEATURES))
        encoding_rows.append({"prediction_horizon": horizon, **encoding})
        record(f"h{horizon}_reencoded_inputs_match_baseline", sum(v for k, v in encoding.items() if k.endswith("mismatches")) == 0, encoding)
        for name, model in frozen.items():
            probability = model.predict(test.stay_id.astype(int).tolist(), static_inputs)
            saved = pd.read_parquet(model.artifact["test_prediction_path"]).set_index("stay_id").loc[test.stay_id]
            diff = float(np.max(np.abs(probability - saved.y_prob.to_numpy(float))))
            tolerance = monitoring["static_reproduction_tolerance"]["ml" if name in cm.ML_MODELS else "deep"]
            same_labels = bool(np.array_equal(saved.y_true.to_numpy(int), test.label.to_numpy(int)))
            auroc_saved = cm.threshold_metrics(saved.y_true.to_numpy(int), saved.y_prob.to_numpy(float), model.threshold)["auroc"]
            auroc_new = cm.threshold_metrics(test.label.to_numpy(int), probability, model.threshold)["auroc"]
            static_rows.append({"artifact_id": model.artifact["artifact_id"], "model": name, "prediction_horizon": horizon, "n": len(probability), "max_abs_prob_diff": diff, "tolerance": tolerance, "auroc_saved": auroc_saved, "auroc_reproduced": auroc_new, "labels_identical": same_labels, "passed": diff <= tolerance and same_labels})
            record(f"{model.artifact['artifact_id']}_static_reproduction", diff <= tolerance and same_labels, f"max|dp|={diff:.3g}, AUROC saved={auroc_saved:.4f} reproduced={auroc_new:.4f}")
        del static_inputs

        # 6. Monitoring time slices.
        for mode in modes:
            reference = references[mode]
            slices = {name: [] for name in frozen}
            for index, timepoint in enumerate(timepoints):
                cutoffs = reference.reference_dt + pd.Timedelta(hours=timepoint)
                next_cutoffs = reference.reference_dt + pd.Timedelta(hours=timepoints[index + 1]) if index + 1 < len(timepoints) else None
                inputs, audit = cm.build_timepoint_inputs(test, cutoffs, events_by_stay, empty_events, scale, baseline, apply_eligibility=True, min_observations=min_obs, reference_dt=reference.reference_dt, next_cutoffs=next_cutoffs)
                summary = slice_summary(audit, reference, mode, horizon, timepoint)
                slice_rows.append(summary)
                audit_frames.append(audit.assign(reference_mode=mode, prediction_horizon=horizon, monitoring_timepoint=timepoint))
                LOGGER.info("h=%d mode=%s t=%+d eligible=%d (AKI %d) events=%d max_rel_ref=%.2f future=%d private_mem=%.2fGB", horizon, mode, timepoint, summary["n_patients_eligible"], summary["n_aki_eligible"], summary["n_events_before_cutoff"], summary["max_event_time_used_h_rel_reference"], summary["future_events_detected"], cm.process_memory_gb())
                if not summary["passed"]:
                    pd.DataFrame(slice_rows).to_csv(layout["summary"] / "time_slice_validation_FAILED.csv", index=False)
                    raise cm.MonitoringAbort(f"time-slice leakage validation failed at h={horizon} mode={mode} t={timepoint}: {summary}")
                eligible_ids = audit.loc[audit.eligible, "stay_id"].astype(int).tolist()
                for name, model in frozen.items():
                    probability = model.predict(eligible_ids, inputs)
                    slices[name].append((timepoint, audit, dict(zip(eligible_ids, probability))))
                del inputs
                gc.collect()  # release this slice's encoded arrays before building the next one
            for name, model in frozen.items():
                pred = assemble_predictions(model, reference, slices[name])
                out = layout["predictions"] / mode / f"horizon_{horizon}"
                out.mkdir(parents=True, exist_ok=True)
                pred.to_parquet(out / f"{name}.parquet", index=False)
                reloaded = pd.read_parquet(out / f"{name}.parquet")
                record(f"{model.artifact['artifact_id']}_{mode}_predictions_saved", len(reloaded) == len(test) * len(timepoints) and reloaded.y_prob.notna().sum() == reloaded.eligible.sum(), f"rows={len(reloaded)}, scored={int(reloaded.y_prob.notna().sum())}")
                metric_records.extend(metric_rows(pred, timepoints))

        for model in frozen.values():
            status = model.integrity_after()
            integrity.append(status)
            record(f"{model.artifact['artifact_id']}_parameters_unchanged", status["parameters_unchanged"], status["fingerprint_after"][:16])
            record(f"{model.artifact['artifact_id']}_artifact_files_unchanged", status["artifact_files_unchanged"], "checkpoint/preprocessing/scaling/config SHA-256 identical")
        del frozen

    # 7. Persist validation, metrics and run record.
    slice_table = pd.DataFrame(slice_rows)
    slice_table.to_csv(layout["summary"] / "time_slice_validation.csv", index=False)
    audit_dir = layout["metrics"] / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(audit_frames, ignore_index=True).to_parquet(audit_dir / "time_slice_patient_audit.parquet", index=False)
    pd.DataFrame(static_rows).to_csv(layout["metrics"] / "static_reproduction_check.csv", index=False)
    pd.DataFrame(encoding_rows).to_csv(layout["metrics"] / "static_input_encoding_check.csv", index=False)
    pd.DataFrame(integrity).to_csv(layout["metrics"] / "model_integrity_check.csv", index=False)
    metrics = pd.DataFrame(metric_records)
    metrics.to_csv(layout["metrics"] / "continuous_monitoring_metrics.csv", index=False)
    record("all_time_slices_no_future_events", int(slice_table.future_events_detected.sum()) == 0 and slice_table.passed.all(), f"slices={len(slice_table)}, future_events={int(slice_table.future_events_detected.sum())}")
    primary = metrics[metrics.reference_mode.eq(monitoring["reference_time"]["primary"]) & metrics.cohort.eq("all_eligible")]
    record("metrics_computed_every_timepoint", len(primary) == len(artifacts) * len(timepoints) and primary.auroc.notna().all(), f"rows={len(primary)}, finite AUROC={int(primary.auroc.notna().sum())}")
    boundary = slice_table[slice_table.reference_mode.eq(monitoring["reference_time"]["primary"])]
    for timepoint, next_timepoint in ((-72, -60), (-48, -36), (0, None)):
        rows = boundary[boundary.monitoring_timepoint.eq(timepoint)]
        limit = timepoint
        ok = bool((rows.max_event_time_used_h_rel_reference <= limit).all() and rows.events_after_next_timepoint.eq(0).all() and rows.future_events_detected.eq(0).all())
        detail = f"max event time used={rows.max_event_time_used_h_rel_reference.max():.3f} h rel. reference (limit {limit})"
        if next_timepoint is not None:
            detail += f"; events after {next_timepoint} h = {int(rows.events_after_next_timepoint.sum())}"
        record(f"boundary_t{timepoint}", ok, detail)

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(layout["summary"] / "run_checks.csv", index=False)
    run_record = {
        "stage": args.stage,
        "run_tag": run_tag,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started, 1),
        "models": models, "prediction_horizons": horizons, "reference_modes": modes, "timepoints": timepoints,
        "n_artifacts": len(artifacts), "n_model_horizon_timepoint": len(artifacts) * len(timepoints),
        "checks_passed": int(checks_frame.passed.sum()), "checks_total": len(checks_frame),
        "artifacts": [{k: a[k] for k in ("artifact_id", "checkpoint_path", "checkpoint_sha256")} for a in artifacts],
    }
    (layout["summary"] / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    lines = [
        f"# Continuous monitoring {args.stage} report", "",
        f"Finished (UTC): {run_record['finished_utc']}  ",
        f"Artifacts: {', '.join(a['artifact_id'] for a in artifacts)}  ",
        f"Reference modes: {', '.join(modes)} (primary: {monitoring['reference_time']['primary']})  ",
        f"Checks: **{run_record['checks_passed']}/{run_record['checks_total']} passed**", "",
        "## Checks", "", markdown_table(checks_frame), "",
        "## Time-slice validation", "",
        markdown_table(slice_table[["reference_mode", "prediction_horizon", "monitoring_timepoint", "n_patients_eligible", "n_aki_eligible", "n_non_aki_eligible", "n_events_before_cutoff", "max_event_time_used_h_rel_reference", "events_after_next_timepoint", "source_events_after_cutoff_excluded", "future_events_detected", "passed"]], "{:.3f}"), "",
        "## Static reproduction gate", "", markdown_table(pd.DataFrame(static_rows), "{:.3g}"), "",
        "## Metrics (primary reference, all eligible)", "",
        markdown_table(primary[["model", "prediction_horizon", "monitoring_timepoint", "N", "N_positive", "auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "nne", "metric_notes"]]),
    ]
    (layout["summary"] / f"{args.stage.replace('-', '_')}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("done in %.1f s; checks %d/%d", time.time() - started, run_record["checks_passed"], run_record["checks_total"])


if __name__ == "__main__":
    try:
        main()
    except cm.MonitoringAbort as error:
        LOGGER.error("ABORTED: %s", error)
        raise SystemExit(2)
