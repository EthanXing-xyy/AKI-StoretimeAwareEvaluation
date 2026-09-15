"""Null control: charttime availability through the st_patch path must reproduce ContinuousMonitoring.

Checks all five architectures at horizon 0, both reference modes. Writes
charttime/summary/null_control_check.{json,csv}; run_stress.py refuses the storetime and
onset_creatinine_only variants until this reports all_passed.

From the repository root after the five charttime monitor runs:
    python Step4_Storetime_Aware_Stress_Test/scripts/check_null_control.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_patch as st  # noqa: E402

cm = st.cm
CM_PREDICTIONS = st.WORK_DIR / "Step2_Continuous_Monitoring" / "predictions"
EXACT_COLUMNS = [
    "model", "prediction_horizon", "reference_mode", "monitoring_timepoint", "time_to_event", "subject_id", "hadm_id",
    "y_true", "eligible", "ineligible_reason", "reference_dt", "cutoff_dt", "cutoff_hours_since_intime", "n_events_used",
    "last_event_dt", "threshold", "y_pred", "alert", "n_eligible_timepoints", "n_alerts_patient", "ever_alerted",
    "first_alert_timepoint", "lead_time_hours",
]


def main() -> None:
    _, monitoring = st._ORIGINAL["load_config"]()
    root = st.STRESS_ROOT / "charttime"
    rows = []
    for model in cm.MODELS:
        tag = f"h0_{model}"
        tolerance = monitoring["static_reproduction_tolerance"]["ml" if model in cm.ML_MODELS else "deep"]
        run_dir = root / "metrics" / "runs" / tag / "availability"
        checks_path = root / "summary" / "runs" / tag / "run_checks.csv"
        if not checks_path.exists():
            rows.append({"run_tag": tag, "passed": False, "detail": "run missing"})
            continue
        run_checks = pd.read_csv(checks_path)
        extraction = json.loads((run_dir / "extraction_record.json").read_text(encoding="utf-8"))
        removed = pd.read_csv(run_dir / "removed_rows_by_slice.csv")
        row = {
            "run_tag": tag, "tolerance": tolerance,
            "run_checks_passed": f"{int(run_checks.passed.sum())}/{len(run_checks)}",
            "row_table_reaggregates_to_original_events": bool(extraction["row_table_reaggregates_to_original_events"]),
            "rows_removed_total": int(removed.n_rows_removed.sum()), "n_slices": len(removed),
        }
        ok = bool(run_checks.passed.all()) and row["row_table_reaggregates_to_original_events"] and row["rows_removed_total"] == 0 and len(removed) == 14
        for mode in monitoring["reference_time"]["modes"]:
            key = ["stay_id", "monitoring_timepoint"]
            new = pd.read_parquet(root / "predictions" / mode / "horizon_0" / f"{model}.parquet").sort_values(key).reset_index(drop=True)
            old = pd.read_parquet(CM_PREDICTIONS / mode / "horizon_0" / f"{model}.parquet").sort_values(key).reset_index(drop=True)
            same_shape = new.shape == old.shape and list(new.columns) == list(old.columns) and new[key].equals(old[key])
            mismatched = [c for c in EXACT_COLUMNS if not new[c].equals(old[c])] if same_shape else ["shape"]
            diff = float(np.nanmax(np.abs(new.y_prob.to_numpy(float) - old.y_prob.to_numpy(float)))) if same_shape else np.inf
            nan_same = same_shape and bool((new.y_prob.isna() == old.y_prob.isna()).all())
            row[f"{mode}_max_abs_prob_diff"] = diff
            row[f"{mode}_mismatched_columns"] = ";".join(mismatched)
            row[f"{mode}_n_scored"] = int(new.y_prob.notna().sum())
            ok = ok and same_shape and not mismatched and nan_same and diff <= tolerance
        row["passed"] = bool(ok)
        rows.append(row)
    table = pd.DataFrame(rows)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "null_control_check.csv", index=False)
    record = {"all_passed": bool(len(table) == 5 and table.passed.all()), "checkpoints": table.replace({np.nan: None}).to_dict("records")}
    (out / "null_control_check.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(table.T.to_string())
    print(json.dumps({"all_passed": record["all_passed"]}))
    if not record["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
