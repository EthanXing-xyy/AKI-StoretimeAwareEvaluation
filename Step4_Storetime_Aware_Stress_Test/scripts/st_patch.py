"""Storetime-aware slicing for the frozen continuous-monitoring pipeline (evaluation only).

Step 2 cm_adapter.py and run_monitoring.py are imported read-only. Four attributes of the
in-memory `cm_adapter` module are replaced; outputs remain under the Step 4 workspace:

  cm.load_config             output roots -> Step 4 workspace/<availability>/...
  cm.compute_reference_times unchanged result; recorded so each slice knows its reference mode
  cm.extract_stay_events     unchanged aggregated events (used by the static reproduction gate) plus a
                             row-level event table in DuckDB with an availability time per row
  cm.build_timepoint_inputs  monitoring slices only (apply_eligibility=True). Strict lab-input order:
                             storetime joined per row -> rows with charttime <= cutoff < avail_dt removed
                             -> avg(valuenum) aggregated by (stay_id, charttime, feature) -> original encoder

storetime is a proxy for when a result became available, not the exact clinician-view time.

Availability variants:
  charttime              avail_dt = charttime for every row (null control; must reproduce CM exactly)
  storetime              lab rows avail_dt = storetime; chartevents (vitals) rows avail_dt = charttime
  onset_creatinine_only  only the onset-defining creatinine rows of creatinine-onset AKI stays use storetime
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PUBLIC_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from project_paths import WORK_DIR  # noqa: E402

STRESS_ROOT = WORK_DIR / "Step4_Storetime_Aware_Stress_Test"
CONFIG_ROOT = PUBLIC_ROOT / "configs"
CM_SCRIPTS = REPO_ROOT / "Step2_Continuous_Monitoring" / "scripts"
AUDIT_ROOT = WORK_DIR / "Step3_Data_Availability_Audit"
STORETIME = AUDIT_ROOT / "data" / "lab_storetime.parquet"
ONSET_TABLE = AUDIT_ROOT / "tables" / "onset_creatinine_availability.csv"
AVAILABILITY = ("charttime", "storetime", "onset_creatinine_only")
CREATININE_ITEMID = 50912

sys.path.insert(0, str(CM_SCRIPTS))
import cm_adapter as cm  # noqa: E402
import run_monitoring  # noqa: E402

LOGGER = logging.getLogger("storetime_stress")
_ORIGINAL = {
    "load_config": cm.load_config,
    "compute_reference_times": cm.compute_reference_times,
    "extract_stay_events": cm.extract_stay_events,
    "build_timepoint_inputs": cm.build_timepoint_inputs,
}


class State:
    variant: str | None = None
    con: duckdb.DuckDBPyConnection | None = None
    references: dict[str, pd.DataFrame] = {}
    removed: list[pd.DataFrame] = []
    slices: list[dict] = []
    extraction: dict = {}


def quote(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


# Row-level twin of cm.EVENT_SQL: identical sources, joins, windows and value filters, without the
# aggregation. Lab rows carry labevent_id and the storetime joined on it (LEFT JOIN, unique key).
ROW_SQL = """
    WITH raw_events AS (
      SELECT c.stay_id, c.charttime, c.feature,
        CASE WHEN c.itemid=223761 THEN (c.valuenum-32)*5/9 ELSE c.valuenum END AS valuenum,
        FALSE AS is_lab, c.itemid, CAST(NULL AS BIGINT) AS labevent_id, CAST(NULL AS TIMESTAMP_NS) AS storetime
      FROM read_parquet('{chart_path}') c JOIN selected_stays s USING(stay_id)
      WHERE c.feature IS NOT NULL AND c.charttime BETWEEN s.intime AND s.upper_dt
      UNION ALL
      SELECT s.stay_id, l.charttime, l.feature, l.valuenum,
        TRUE AS is_lab, l.itemid, l.labevent_id, CAST(r.storetime AS TIMESTAMP_NS) AS storetime
      FROM read_parquet('{labs_path}') l JOIN selected_stays s USING(hadm_id)
      LEFT JOIN read_parquet('{storetime_path}') r ON l.labevent_id = r.labevent_id
      WHERE l.charttime BETWEEN s.intime AND s.upper_dt
    )
    SELECT * FROM raw_events
    WHERE valuenum IS NOT NULL AND isfinite(valuenum)
      AND (feature!='sbp' OR valuenum BETWEEN 40 AND 300)
      AND (feature!='dbp' OR valuenum BETWEEN 20 AND 200)
      AND (feature!='pulse' OR valuenum BETWEEN 20 AND 300)
      AND (feature!='resp' OR valuenum BETWEEN 1 AND 80)
      AND (feature!='spo2' OR valuenum BETWEEN 1 AND 100)
      AND (feature!='temp' OR valuenum BETWEEN 25 AND 45)
"""

# e = row-level event, o = onset-defining (stay_id, aki_dt) of creatinine-onset AKI stays.
AVAIL_EXPR = {
    "charttime": "e.charttime",
    # A lab row without storetime would be kept as available (counted in the extraction record).
    "storetime": "CASE WHEN e.is_lab THEN coalesce(e.storetime, e.charttime) ELSE e.charttime END",
    # Match the Step 3 defining-row rule (itemid 50912, 0.1-150 mg/dL, charttime = aki_dt).
    "onset_creatinine_only": (
        f"CASE WHEN e.is_lab AND e.itemid = {CREATININE_ITEMID} AND e.valuenum BETWEEN 0.1 AND 150 AND o.stay_id IS NOT NULL "
        "THEN coalesce(e.storetime, e.charttime) ELSE e.charttime END"
    ),
}

AGGREGATE_SQL = """
    SELECT stay_id, charttime, feature, avg(valuenum) AS valuenum FROM ({rows})
    GROUP BY stay_id, charttime, feature ORDER BY stay_id, charttime, feature
"""


def roots(variant: str) -> dict[str, Path]:
    base = STRESS_ROOT / variant
    return {
        "project_root": base, "predictions_root": base / "predictions", "metrics_root": base / "metrics",
        "figures_root": base / "figures", "summary_root": base / "summary", "logs_root": STRESS_ROOT / "logs" / variant,
    }


# ------------------------------------------------------------------------- patches
def load_config():
    paths, monitoring = _ORIGINAL["load_config"]()
    paths = dict(paths)
    paths.update(roots(State.variant))
    return paths, monitoring


def compute_reference_times(master, test, mode, step1, monitoring):
    frame, matching = _ORIGINAL["compute_reference_times"](master, test, mode, step1, monitoring)
    State.references[mode] = frame
    return frame, matching


def _connection() -> duckdb.DuckDBPyConnection:
    if State.con is None:
        tmp = STRESS_ROOT / "tmp_duckdb"
        tmp.mkdir(exist_ok=True)
        State.con = duckdb.connect()
        State.con.execute("PRAGMA threads=8")
        State.con.execute("PRAGMA memory_limit='3GB'")
        State.con.execute(f"PRAGMA temp_directory='{quote(tmp)}'")
    return State.con


def _to_events(frame: pd.DataFrame) -> pd.DataFrame:
    frame["charttime"] = pd.to_datetime(frame.charttime)
    return frame


def extract_stay_events(interim_root: Path, test: pd.DataFrame) -> pd.DataFrame:
    events = _ORIGINAL["extract_stay_events"](interim_root, test)
    events_dir = interim_root / "mimiciv" / "events"
    con = _connection()
    con.register("selected_stays", test[["stay_id", "hadm_id", "intime"]].assign(upper_dt=test.outtime))
    onset = pd.read_csv(ONSET_TABLE, usecols=["stay_id", "aki_dt", "creatinine_defines_onset"])
    onset = onset[onset.creatinine_defines_onset.astype(str).str.lower().eq("true")][["stay_id", "aki_dt"]].copy()
    onset["aki_dt"] = pd.to_datetime(onset.aki_dt)
    cohort_onset = test.loc[test.label.eq(1), ["stay_id", "aki_dt"]].assign(aki_dt=lambda f: pd.to_datetime(f.aki_dt))
    check = onset.merge(cohort_onset, on="stay_id", how="left", suffixes=("", "_cohort"))
    if check.aki_dt_cohort.isna().any() or not (check.aki_dt == check.aki_dt_cohort).all():
        raise cm.MonitoringAbort("onset-defining creatinine table does not match the frozen test cohort aki_dt")
    con.register("onset_defining", onset)
    rows = ROW_SQL.format(chart_path=quote(events_dir / "chartevents.parquet"), labs_path=quote(events_dir / "labevents.parquet"), storetime_path=quote(STORETIME))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE event_rows AS
        SELECT e.stay_id, e.charttime, e.feature, e.valuenum, e.is_lab, e.itemid, e.labevent_id, e.storetime,
               {AVAIL_EXPR[State.variant]} AS avail_dt
        FROM ({rows}) e LEFT JOIN onset_defining o ON e.stay_id = o.stay_id AND e.charttime = o.aki_dt
    """)
    stats = con.execute("""
        SELECT count(*) AS n_rows, sum(is_lab::INT) AS n_lab_rows,
               count(DISTINCT labevent_id) AS n_distinct_labevent_id,
               sum((is_lab AND storetime IS NULL)::INT) AS n_lab_rows_missing_storetime,
               sum((avail_dt > charttime)::INT) AS n_rows_avail_after_charttime,
               sum((avail_dt < charttime)::INT) AS n_rows_avail_before_charttime,
               sum((is_lab AND itemid = 50912 AND avail_dt != charttime)::INT) AS n_creatinine_rows_avail_shifted
        FROM event_rows
    """).df().iloc[0].to_dict()
    State.extraction = {"variant": State.variant, **{k: int(v) for k, v in stats.items()}}
    if State.variant == "charttime" and stats["n_rows_avail_after_charttime"] != 0:
        raise cm.MonitoringAbort("null control must not shift any availability time")

    # Gate: aggregating the row table without removal must reproduce the original aggregated events bit for bit.
    rebuilt = _to_events(con.execute(AGGREGATE_SQL.format(rows="SELECT * FROM event_rows")).df())
    identical = bool(rebuilt.reset_index(drop=True).equals(events.reset_index(drop=True)))
    State.extraction["row_table_reaggregates_to_original_events"] = identical
    State.extraction["n_aggregated_events"] = int(len(events))
    LOGGER.info("availability=%s row table: %s", State.variant, State.extraction)
    if not identical:
        raise cm.MonitoringAbort("row-level event table does not re-aggregate to cm.extract_stay_events output")
    del rebuilt
    return events


def _slice_identity(frame: pd.DataFrame, cutoffs: pd.Series, reference_dt: pd.Series | None) -> tuple[str, int]:
    if reference_dt is None:
        raise cm.MonitoringAbort("monitoring slice without reference_dt")
    ref = pd.to_datetime(reference_dt).reset_index(drop=True)
    offsets = ((pd.to_datetime(cutoffs).reset_index(drop=True) - ref).dt.total_seconds() / 3600).round(6).unique()
    if len(offsets) != 1:
        raise cm.MonitoringAbort(f"slice cutoffs are not a single offset from reference: {offsets[:5]}")
    modes = [m for m, r in State.references.items() if r.stay_id.reset_index(drop=True).equals(frame.stay_id.reset_index(drop=True)) and pd.to_datetime(r.reference_dt).reset_index(drop=True).equals(ref)]
    if len(modes) != 1:
        raise cm.MonitoringAbort(f"cannot identify reference mode for slice (matches={modes})")
    return modes[0], int(offsets[0])


def build_timepoint_inputs(frame, cutoffs, events_by_stay, empty_events, scale, baseline, apply_eligibility, min_observations, reference_dt=None, next_cutoffs=None):
    if not apply_eligibility:  # static reproduction gate: original, charttime-only events
        return _ORIGINAL["build_timepoint_inputs"](frame, cutoffs, events_by_stay, empty_events, scale, baseline, apply_eligibility, min_observations, reference_dt, next_cutoffs)
    mode, timepoint = _slice_identity(frame, cutoffs, reference_dt)
    con = _connection()
    con.register("slice_cutoffs", pd.DataFrame({"stay_id": frame.stay_id.astype("int64").to_numpy(), "cutoff_dt": pd.to_datetime(cutoffs).to_numpy()}))
    unavailable = "r.charttime <= c.cutoff_dt AND r.avail_dt > c.cutoff_dt"
    # 1) row level: storetime already joined; 2) remove rows not yet available at this stay's cutoff
    kept_rows = f"SELECT r.* FROM event_rows r JOIN slice_cutoffs c USING (stay_id) WHERE NOT ({unavailable})"
    removed = con.execute(f"""
        SELECT r.stay_id, r.feature, r.is_lab, count(*) AS n_rows_removed,
               count(DISTINCT r.charttime) AS n_charttimes_removed,
               median(date_diff('second', c.cutoff_dt, r.avail_dt) / 60.0) AS median_minutes_until_available
        FROM event_rows r JOIN slice_cutoffs c USING (stay_id)
        WHERE {unavailable}
        GROUP BY ALL
    """).df()
    # 3) aggregate the remaining rows exactly as cm.EVENT_SQL does
    events = _to_events(con.execute(AGGREGATE_SQL.format(rows=kept_rows)).df())
    con.unregister("slice_cutoffs")
    sliced = {int(key): group for key, group in events.groupby("stay_id", sort=False)}
    State.removed.append(removed.assign(reference_mode=mode, monitoring_timepoint=timepoint))
    State.slices.append({"reference_mode": mode, "monitoring_timepoint": timepoint, "n_aggregated_events_superset": int(len(events)),
                         "n_rows_removed": int(removed.n_rows_removed.sum()), "n_lab_rows_removed": int(removed.loc[removed.is_lab, "n_rows_removed"].sum()),
                         "n_stays_with_removed_rows": int(removed.stay_id.nunique())})
    LOGGER.info("availability=%s mode=%s t=%+d removed rows=%d (lab %d) stays=%d", State.variant, mode, timepoint, State.slices[-1]["n_rows_removed"], State.slices[-1]["n_lab_rows_removed"], State.slices[-1]["n_stays_with_removed_rows"])
    # 4) encode with the unchanged baseline encoder
    return _ORIGINAL["build_timepoint_inputs"](frame, cutoffs, sliced, empty_events, scale, baseline, apply_eligibility, min_observations, reference_dt, next_cutoffs)


def activate(variant: str) -> None:
    if variant not in AVAILABILITY:
        raise ValueError(f"availability must be one of {AVAILABILITY}")
    State.variant = variant
    cm.load_config = load_config
    cm.compute_reference_times = compute_reference_times
    cm.extract_stay_events = extract_stay_events
    cm.build_timepoint_inputs = build_timepoint_inputs


def write_run_records(run_tag: str) -> None:
    out = roots(State.variant)["metrics_root"] / "runs" / run_tag / "availability"
    out.mkdir(parents=True, exist_ok=True)
    removed = pd.concat(State.removed, ignore_index=True) if State.removed else pd.DataFrame()
    removed.to_parquet(out / "removed_rows_by_stay_feature.parquet", index=False)
    pd.DataFrame(State.slices).to_csv(out / "removed_rows_by_slice.csv", index=False)
    (out / "extraction_record.json").write_text(json.dumps(State.extraction, indent=2), encoding="utf-8")
