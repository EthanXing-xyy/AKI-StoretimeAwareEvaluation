"""Pre-flight check of st_patch slicing without any model inference.

For one availability variant: builds the row-level event table (the bit-identical re-aggregation gate
runs inside), then replays the matched slices at t = -12 h and 0 h with the encoder stubbed out, and
compares removed in-scope laboratory rows on eligible stays with Step 3 pending-row counts.

From the repository root:
    python Step4_Storetime_Aware_Stress_Test/scripts/check_patch_offline.py --availability storetime
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_patch as st  # noqa: E402

cm = st.cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--availability", choices=st.AVAILABILITY, required=True)
    args = parser.parse_args()
    st.activate(args.availability)
    paths, monitoring = cm.load_config()
    master = pd.read_parquet(paths["baseline_cohort_root"] / "master_include.parquet")
    test = master[master.split.eq("test")].reset_index(drop=True)
    if cm.cohort_sha256(test) != monitoring["expected_test_cohort_sha256"]:
        raise SystemExit("test cohort hash mismatch")

    started = time.time()
    events = cm.extract_stay_events(paths["baseline_data_root"] / "interim", test)
    extraction_seconds = time.time() - started
    captured = {}
    st._ORIGINAL["build_timepoint_inputs"] = lambda frame, cutoffs, sliced, *rest: (captured.update(sliced=sliced), None)[1]

    scope = pd.read_csv(st.AUDIT_ROOT / "tables" / "lab_feature_scope.csv")
    scope_features = set(scope.loc[scope.iloc[:, 0].notna(), "feature"]) if "feature" in scope else set()
    pending = pd.read_csv(st.AUDIT_ROOT / "tables" / "pending_by_timepoint_feature_group.csv")
    grid = pd.read_parquet(st.WORK_DIR / "Step2_Continuous_Monitoring/predictions/matched/horizon_0/logistic_regression.parquet",
                           columns=["stay_id", "monitoring_timepoint", "reference_dt", "cutoff_dt", "eligible"])
    rows = []
    for timepoint in (-12, 0):
        g = grid[grid.monitoring_timepoint.eq(timepoint)].set_index("stay_id").loc[test.stay_id].reset_index()
        st.State.references = {"matched": pd.DataFrame({"stay_id": test.stay_id, "reference_dt": pd.to_datetime(g.reference_dt).to_numpy()})}
        cutoffs = pd.Series(pd.to_datetime(g.cutoff_dt).to_numpy(), index=test.index)
        t0 = time.time()
        cm.build_timepoint_inputs(test, cutoffs, None, events.iloc[0:0], None, None, True, 1, reference_dt=st.State.references["matched"].reference_dt, next_cutoffs=None)
        seconds = time.time() - t0
        removed = st.State.removed[-1]
        eligible_ids = set(g.loc[g.eligible, "stay_id"])
        on_eligible = removed[removed.stay_id.isin(eligible_ids) & removed.is_lab]
        audit = pending[pending["mode"].eq("matched") & pending.cohort.eq("all_eligible") & pending.monitoring_timepoint.eq(timepoint) & pending.grp.eq("all") & pending.feature.eq("ALL_LABS")]
        n_events_sliced = sum(len(v) for v in captured["sliced"].values())
        rows.append({
            "availability": args.availability, "monitoring_timepoint": timepoint, "slice_seconds": round(seconds, 1),
            "removed_lab_rows_eligible_all_features": int(on_eligible.n_rows_removed.sum()),
            "removed_lab_rows_eligible_scope_features": int(on_eligible[on_eligible.feature.isin(scope_features)].n_rows_removed.sum()) if scope_features else None,
            "audit_pending_rows": int(audit.n_pending_rows.iloc[0]),
            "removed_nonlab_rows": int(removed.loc[~removed.is_lab, "n_rows_removed"].sum()),
            "removed_features": sorted(on_eligible.feature.dropna().unique().tolist()),
            "aggregated_events_after_removal": n_events_sliced, "aggregated_events_original": len(events),
        })
    print(json.dumps({"extraction": st.State.extraction, "extraction_seconds": round(extraction_seconds, 1), "slices": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
