"""Merge the per-(horizon, model) full-stage runs into the phase-two result tables.

From the repository root, after all 15 runs of
    python Step2_Continuous_Monitoring/scripts/run_monitoring.py --stage full --horizons H --models M
then:
    python Step2_Continuous_Monitoring/scripts/merge_monitoring_runs.py

The full stage is split into one process per checkpoint only to stay within host memory; the
computation per checkpoint is identical to a single-process run. This script fails if any
checkpoint is missing or duplicated, if any run check failed, or if model-independent outputs
(time-slice audits and input-encoding checks) disagree between runs.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import cm_adapter as cm  # noqa: E402
from run_monitoring import markdown_table  # noqa: E402

SLICE_KEYS = ["reference_mode", "prediction_horizon", "monitoring_timepoint"]


def fail(message: str) -> None:
    raise SystemExit(f"MERGE FAILED: {message}")


def identical_by_key(frame: pd.DataFrame, keys: list[str], label: str) -> pd.DataFrame:
    """Keep one copy of model-independent rows after proving every run produced the same values."""
    values = frame.drop(columns=["run_tag"])
    # include_groups=False already removes the key columns from each group.
    duplicates = values.groupby(keys, dropna=False).apply(lambda g: g.drop_duplicates().shape[0], include_groups=False)
    if (duplicates != 1).any():
        fail(f"{label} differs between runs for {duplicates[duplicates != 1].index.tolist()}")
    return values.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)


def main() -> None:
    paths, monitoring = cm.load_config()
    timepoints, modes = monitoring["time_points_hours"], monitoring["reference_time"]["modes"]
    summary_runs, metrics_runs = paths["summary_root"] / "runs", paths["metrics_root"] / "runs"
    tags = sorted(p.name for p in summary_runs.iterdir() if p.is_dir())
    records = {tag: json.loads((summary_runs / tag / "run_record.json").read_text(encoding="utf-8")) for tag in tags}

    expected = {(h, m) for h in cm.HORIZONS for m in cm.MODELS}
    covered = [(h, m) for r in records.values() for h in r["prediction_horizons"] for m in r["models"]]
    if sorted(covered) != sorted(expected) or len(covered) != len(set(covered)):
        fail(f"runs cover {len(set(covered))} unique of 15 checkpoints ({len(covered)} total): missing={sorted(expected - set(covered))}")

    def collect(root: Path, name: str) -> pd.DataFrame:
        return pd.concat([pd.read_csv(root / tag / name).assign(run_tag=tag) for tag in tags], ignore_index=True)

    checks = collect(summary_runs, "run_checks.csv")
    if not checks.passed.all():
        fail(f"failed run checks: {checks.loc[~checks.passed, ['run_tag', 'check', 'detail']].to_dict('records')}")
    slices = identical_by_key(collect(summary_runs, "time_slice_validation.csv"), SLICE_KEYS, "time-slice validation")
    if len(slices) != len(modes) * len(cm.HORIZONS) * len(timepoints) or not slices.passed.all() or slices.future_events_detected.sum() != 0:
        fail("time-slice validation incomplete or failed")
    encoding = identical_by_key(collect(metrics_runs, "static_input_encoding_check.csv"), ["prediction_horizon"], "input-encoding check")
    static = collect(metrics_runs, "static_reproduction_check.csv")
    integrity = collect(metrics_runs, "model_integrity_check.csv")
    metrics = collect(metrics_runs, "continuous_monitoring_metrics.csv").drop(columns="run_tag")
    if len(static) != 15 or not static.passed.all():
        fail("static reproduction incomplete or failed")
    if len(integrity) != 15 or not (integrity.parameters_unchanged.all() and integrity.artifact_files_unchanged.all()):
        fail("model/artifact integrity incomplete or failed")
    expected_rows = 15 * len(timepoints) * len(modes)
    if len(metrics) != expected_rows or metrics.duplicated(["reference_mode", "cohort", "model", "prediction_horizon", "monitoring_timepoint"]).any():
        fail(f"metrics rows {len(metrics)} != {expected_rows} or duplicated")

    prediction_rows = 0
    for mode in modes:
        for horizon, model in sorted(expected):
            pred = pd.read_parquet(paths["predictions_root"] / mode / f"horizon_{horizon}" / f"{model}.parquet", columns=["stay_id", "monitoring_timepoint", "eligible", "y_prob"])
            if len(pred) != monitoring["expected_test_stays"] * len(timepoints) or (pred.eligible != pred.y_prob.notna()).any():
                fail(f"prediction file incomplete: {mode}/{horizon}/{model}")
            prediction_rows += len(pred)

    audits = []
    for horizon in cm.HORIZONS:
        tag = next(t for t in tags if horizon in records[t]["prediction_horizons"])
        audit = pd.read_parquet(metrics_runs / tag / "audit" / "time_slice_patient_audit.parquet")
        audits.append(audit[audit.prediction_horizon.eq(horizon)])
    audit_dir = paths["metrics_root"] / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(audits, ignore_index=True).to_parquet(audit_dir / "time_slice_patient_audit.parquet", index=False)

    order = lambda frame: frame.sort_values(["prediction_horizon", "model"], key=lambda s: s.map(cm.MODELS.index) if s.name == "model" else s).reset_index(drop=True)
    slices.to_csv(paths["summary_root"] / "time_slice_validation.csv", index=False)
    encoding.to_csv(paths["metrics_root"] / "static_input_encoding_check.csv", index=False)
    order(static.drop(columns="run_tag")).to_csv(paths["metrics_root"] / "static_reproduction_check.csv", index=False)
    integrity.drop(columns="run_tag").to_csv(paths["metrics_root"] / "model_integrity_check.csv", index=False)
    metrics.sort_values(["reference_mode", "cohort", "prediction_horizon", "model", "monitoring_timepoint"]).to_csv(paths["metrics_root"] / "continuous_monitoring_metrics.csv", index=False)
    checks.to_csv(paths["summary_root"] / "run_checks.csv", index=False)
    collect(summary_runs, "artifact_selection.csv").query("selection == 'selected'").drop(columns="run_tag").to_csv(paths["summary_root"] / "artifact_selection.csv", index=False)

    merged = {
        "stage": "full",
        "execution": "one process per (horizon, model) checkpoint, sequential, to stay within host memory",
        "runs": tags,
        "n_artifacts": 15,
        "n_model_horizon_timepoint": 15 * len(timepoints),
        "reference_modes": modes,
        "prediction_rows_written": prediction_rows,
        "checks_passed": int(checks.passed.sum()), "checks_total": len(checks),
        "runtime_seconds_total": round(sum(r["runtime_seconds"] for r in records.values()), 1),
        "slice_future_events_total": int(slices.future_events_detected.sum()),
    }
    (paths["summary_root"] / "run_record.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    per_run = checks.groupby("run_tag").passed.agg(["sum", "count"]).reset_index().rename(columns={"sum": "passed", "count": "total"})
    lines = [
        "# Continuous monitoring full run report", "",
        f"Checkpoints: 15 (5 models × 3 horizons) × {len(timepoints)} time points = {15 * len(timepoints)} evaluations per reference mode ({', '.join(modes)}).  ",
        f"Execution: {merged['execution']}; total runtime {merged['runtime_seconds_total'] / 60:.1f} min.  ",
        f"Checks: **{merged['checks_passed']}/{merged['checks_total']} passed** across runs; merge consistency checks passed.", "",
        "## Checks per run", "", markdown_table(per_run), "",
        "## Time-slice validation", "",
        markdown_table(slices[["reference_mode", "prediction_horizon", "monitoring_timepoint", "n_patients_eligible", "n_aki_eligible", "n_non_aki_eligible", "n_events_before_cutoff", "max_event_time_used_h_rel_reference", "events_after_next_timepoint", "source_events_after_cutoff_excluded", "future_events_detected", "passed"]], "{:.3f}"), "",
        "## Static reproduction gate", "", markdown_table(order(static.drop(columns="run_tag")), "{:.3g}"), "",
        "## Input re-encoding vs saved baseline inputs", "", markdown_table(encoding), "",
        "## Model and artifact integrity", "", markdown_table(integrity[["artifact_id", "parameters_unchanged", "artifact_files_unchanged"]]), "",
    ]
    (paths["summary_root"] / "full_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
