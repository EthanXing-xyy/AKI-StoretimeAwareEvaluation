"""Measure charttime-visible laboratory inputs pending at each cutoff.

For every eligible (stay, monitoring timepoint) of phase two, a lab row is *visible* when
intime <= charttime <= cutoff (the retrospective slice) and *pending* when additionally
cutoff < storetime. Rows with missing storetime are counted as *unknown* and treated as available
(so pending counts are a lower bound); they are reported, not dropped.

Three levels:
- rows and model observations (stay, charttime, feature), by feature x timepoint x AKI group;
- stays with >= 1 pending lab / pending creatinine;
- last observed value per (stay, feature) at the cutoff with and without pending rows.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as ac  # noqa: E402

KEYS = "mode, cohort, stay_id, monitoring_timepoint"


def build_grid(test: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for mode in ac.MODES:
        grid = ac.monitoring_grid(mode)
        grid = grid[grid.eligible].drop(columns="eligible")
        if set(grid.stay_id) - set(test.stay_id):
            raise RuntimeError(f"{mode}: grid stays outside the test cohort")
        frames.append(grid.assign(cohort="all_eligible", mode=mode))
    grid = pd.concat(frames, ignore_index=True)
    grid["grp"] = grid.y_true.map(ac.GROUP_LABELS)
    return grid[["mode", "cohort", "stay_id", "monitoring_timepoint", "cutoff_dt", "grp"]]


def label_sets(frame: pd.DataFrame) -> pd.DataFrame:
    frame["grp"] = frame["grp"].fillna("all")
    if "feature" in frame:
        frame["feature"] = frame["feature"].fillna("ALL_LABS")
    return frame.sort_values([c for c in ["mode", "cohort", "monitoring_timepoint", "grp", "feature"] if c in frame]).reset_index(drop=True)


def consistency_check(con, test: pd.DataFrame, grid: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    """Lab observation counts must equal phase two's event extraction (cm_adapter.EVENT_SQL)."""
    mapping = pd.read_csv(ac.INTERIM / "feature_itemid_mapping.csv")
    lab_only = sorted(set(scope.feature) - set(mapping.loc[mapping.source.eq("chartevents"), "feature"]))
    sample = test.sample(n=80, random_state=0)
    events = ac.cm.extract_stay_events(ac.BASELINE_DATA_ROOT / "interim", sample)
    events = events[events.feature.isin(lab_only)]
    g = grid[grid["mode"].eq("matched") & grid.cohort.eq("all_eligible") & grid.stay_id.isin(sample.stay_id)]
    merged = g.merge(events, on="stay_id")
    expected = merged[merged.charttime <= merged.cutoff_dt].groupby(["stay_id", "monitoring_timepoint"]).size().rename("cm_event_sql_obs")
    con.register("lab_only", pd.DataFrame({"feature": lab_only}))
    ours = con.execute("""
        SELECT stay_id, monitoring_timepoint, count(*) AS audit_obs FROM obs
        WHERE mode='matched' AND cohort='all_eligible' AND feature IN (SELECT feature FROM lab_only)
        GROUP BY 1, 2
    """).df().set_index(["stay_id", "monitoring_timepoint"]).audit_obs
    keys = g.set_index(["stay_id", "monitoring_timepoint"]).index
    table = pd.DataFrame({"cm_event_sql_obs": expected.reindex(keys).fillna(0).astype(int), "audit_obs": ours.reindex(keys).fillna(0).astype(int)}).reset_index()
    table["match"] = table.cm_event_sql_obs.eq(table.audit_obs)
    return table


def main() -> None:
    scope, _, _ = ac.lab_feature_scope()
    test = ac.load_test_cohort()
    grid = build_grid(test)
    con = ac.connect("4GB")
    ac.register_test_labs(con, test, scope)
    con.register("grid", grid)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE vis AS
        SELECT g.mode, g.cohort, g.stay_id, g.monitoring_timepoint, g.grp, g.cutoff_dt,
               l.feature, l.labevent_id, l.charttime, l.valuenum, l.storetime,
               coalesce(l.storetime > g.cutoff_dt, FALSE) AS pending,
               l.storetime IS NULL AS unknown,
               date_diff('second', g.cutoff_dt, l.storetime) / 60.0 AS minutes_until_stored
        FROM grid g JOIN test_labs l ON g.stay_id = l.stay_id
        WHERE l.charttime <= g.cutoff_dt
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE obs AS
        SELECT {KEYS}, grp, feature, charttime,
               avg(valuenum) AS value_all,
               avg(valuenum) FILTER (WHERE NOT pending) AS value_available,
               bool_or(pending) AS any_pending, bool_or(unknown) AS any_unknown
        FROM vis GROUP BY ALL
    """)

    rows = con.execute(f"""
        -- Grouping-set totals have NULL grp/feature; label them before the join (NULL never matches NULL).
        WITH row_counts AS (
          SELECT mode, cohort, monitoring_timepoint, coalesce(grp, 'all') AS grp, coalesce(feature, 'ALL_LABS') AS feature,
                 count(*) AS n_visible_rows, sum(pending::INT) AS n_pending_rows, sum(unknown::INT) AS n_unknown_storetime_rows,
                 quantile_cont(minutes_until_stored, 0.5) FILTER (WHERE pending) AS median_minutes_until_stored_pending
          FROM vis GROUP BY GROUPING SETS ((mode, cohort, monitoring_timepoint, grp, feature), (mode, cohort, monitoring_timepoint, grp),
                                           (mode, cohort, monitoring_timepoint, feature), (mode, cohort, monitoring_timepoint))
        ), obs_counts AS (
          SELECT mode, cohort, monitoring_timepoint, coalesce(grp, 'all') AS grp, coalesce(feature, 'ALL_LABS') AS feature,
                 count(*) AS n_visible_observations, sum(any_pending::INT) AS n_observations_with_pending_row,
                 sum((value_available IS NULL)::INT) AS n_observations_fully_pending
          FROM obs GROUP BY GROUPING SETS ((mode, cohort, monitoring_timepoint, grp, feature), (mode, cohort, monitoring_timepoint, grp),
                                           (mode, cohort, monitoring_timepoint, feature), (mode, cohort, monitoring_timepoint))
        )
        SELECT * FROM row_counts FULL JOIN obs_counts USING (mode, cohort, monitoring_timepoint, grp, feature)
    """).df()
    rows = label_sets(rows)
    eligible = pd.concat([grid.groupby(["mode", "cohort", "monitoring_timepoint", "grp"]).size(),
                          grid.assign(grp="all").groupby(["mode", "cohort", "monitoring_timepoint", "grp"]).size()]).rename("n_eligible_stays").reset_index()
    rows = rows.merge(eligible, on=["mode", "cohort", "monitoring_timepoint", "grp"], how="left")
    rows["pct_pending_rows"] = 100 * rows.n_pending_rows / rows.n_visible_rows
    rows["pct_observations_with_pending_row"] = 100 * rows.n_observations_with_pending_row / rows.n_visible_observations
    ac.write(rows, "pending_by_timepoint_feature_group.csv")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE last_value AS
        SELECT {KEYS}, grp, feature,
               max(charttime) AS last_charttime,
               arg_max(value_all, charttime) AS last_value,
               max(charttime) FILTER (WHERE value_available IS NOT NULL) AS last_available_charttime,
               arg_max(value_available, charttime) FILTER (WHERE value_available IS NOT NULL) AS last_available_value,
               arg_max(any_unknown, charttime) AS last_observation_has_unknown_storetime
        FROM obs GROUP BY ALL
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE last_value_flags AS
        SELECT *,
               last_available_charttime IS NULL AS becomes_missing,
               (last_available_charttime IS NULL OR last_available_charttime <> last_charttime
                OR abs(last_available_value - last_value) > 1e-9) AS changed,
               date_diff('second', last_available_charttime, last_charttime) / 3600.0 AS hours_staler,
               abs(last_available_value - last_value) AS abs_value_change
        FROM last_value
    """)
    last = con.execute("""
        SELECT mode, cohort, monitoring_timepoint, grp, feature,
               count(*) AS n_stays_with_feature_visible, sum(changed::INT) AS n_last_value_changed,
               sum(becomes_missing::INT) AS n_becomes_missing,
               sum(last_observation_has_unknown_storetime::INT) AS n_last_observation_unknown_storetime,
               quantile_cont(hours_staler, 0.5) FILTER (WHERE changed AND NOT becomes_missing) AS median_hours_staler_when_changed,
               quantile_cont(abs_value_change, 0.5) FILTER (WHERE changed AND NOT becomes_missing) AS median_abs_value_change_when_changed
        FROM last_value_flags
        GROUP BY GROUPING SETS ((mode, cohort, monitoring_timepoint, grp, feature), (mode, cohort, monitoring_timepoint, feature))
    """).df()
    last = label_sets(last)
    last["pct_last_value_changed"] = 100 * last.n_last_value_changed / last.n_stays_with_feature_visible
    ac.write(last, "last_value_changed.csv")

    flags = con.execute(f"""
        WITH per_stay AS (
          SELECT {KEYS}, count(*) AS n_visible_rows, sum(pending::INT) AS n_pending_rows, sum(unknown::INT) AS n_unknown_rows,
                 bool_or(pending) AS any_pending, bool_or(pending AND feature='creatinine') AS creatinine_pending,
                 bool_or(feature='creatinine') AS creatinine_visible
          FROM vis GROUP BY ALL
        ), per_stay_last AS (
          SELECT {KEYS}, bool_or(changed) AS any_last_value_changed,
                 bool_or(changed AND feature='creatinine') AS creatinine_last_value_changed,
                 sum(changed::INT) AS n_features_changed
          FROM last_value_flags GROUP BY ALL
        )
        SELECT g.mode, g.cohort, g.stay_id, g.monitoring_timepoint, g.grp, g.cutoff_dt,
               coalesce(p.n_visible_rows, 0) AS n_visible_rows, coalesce(p.n_pending_rows, 0) AS n_pending_rows,
               coalesce(p.n_unknown_rows, 0) AS n_unknown_rows, coalesce(p.any_pending, FALSE) AS any_pending,
               coalesce(p.creatinine_visible, FALSE) AS creatinine_visible, coalesce(p.creatinine_pending, FALSE) AS creatinine_pending,
               coalesce(l.any_last_value_changed, FALSE) AS any_last_value_changed,
               coalesce(l.creatinine_last_value_changed, FALSE) AS creatinine_last_value_changed,
               coalesce(l.n_features_changed, 0) AS n_features_changed
        FROM grid g
        LEFT JOIN per_stay p USING (mode, cohort, stay_id, monitoring_timepoint)
        LEFT JOIN per_stay_last l USING (mode, cohort, stay_id, monitoring_timepoint)
    """).df()
    output_data = ac.WORK_ROOT / "Step3_Data_Availability_Audit/data"
    output_data.mkdir(parents=True, exist_ok=True)
    flags.to_parquet(output_data / "stay_timepoint_flags.parquet", index=False)

    def stay_summary(frame: pd.DataFrame) -> pd.Series:
        n = len(frame)
        return pd.Series({
            "n_eligible_stays": n,
            "n_stays_any_visible_lab": int(frame.n_visible_rows.gt(0).sum()),
            "n_stays_any_pending_lab": int(frame.any_pending.sum()), "pct_stays_any_pending_lab": 100 * frame.any_pending.mean(),
            "n_stays_creatinine_visible": int(frame.creatinine_visible.sum()),
            "n_stays_creatinine_pending": int(frame.creatinine_pending.sum()), "pct_stays_creatinine_pending": 100 * frame.creatinine_pending.mean(),
            "n_stays_any_last_value_changed": int(frame.any_last_value_changed.sum()), "pct_stays_any_last_value_changed": 100 * frame.any_last_value_changed.mean(),
            "n_stays_creatinine_last_value_changed": int(frame.creatinine_last_value_changed.sum()),
            "pct_stays_creatinine_last_value_changed": 100 * frame.creatinine_last_value_changed.mean(),
            "mean_pending_rows_per_stay": frame.n_pending_rows.mean(), "median_pending_rows_per_stay": frame.n_pending_rows.median(),
            "mean_visible_rows_per_stay": frame.n_visible_rows.mean(),
            "mean_features_changed_per_stay": frame.n_features_changed.mean(),
            "n_unknown_storetime_rows": int(frame.n_unknown_rows.sum()),
        })

    keys = ["mode", "cohort", "monitoring_timepoint"]
    stays = pd.concat([
        flags.groupby(keys + ["grp"]).apply(stay_summary, include_groups=False).reset_index(),
        flags.assign(grp="all").groupby(keys + ["grp"]).apply(stay_summary, include_groups=False).reset_index(),
    ], ignore_index=True)
    stays = label_sets(stays)
    ac.write(stays, "pending_stay_level.csv")

    check = consistency_check(con, test, grid, scope)
    con.close()
    ac.write(check, "consistency_vs_cm_event_sql.csv")
    print(f"consistency with cm_adapter.EVENT_SQL: {int(check.match.sum())}/{len(check)} stay-timepoints match")
    view = stays[stays["mode"].eq("matched") & stays.cohort.eq("all_eligible")]
    print(view[["monitoring_timepoint", "grp", "n_eligible_stays", "pct_stays_any_pending_lab", "pct_stays_creatinine_pending",
                "pct_stays_any_last_value_changed", "pct_stays_creatinine_last_value_changed", "mean_pending_rows_per_stay"]].round(2).to_string(index=False))
    if not check.match.all():
        raise SystemExit("consistency check failed: see tables/consistency_vs_cm_event_sql.csv")


if __name__ == "__main__":
    main()
