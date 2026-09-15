"""Construct the corrected MIMIC-IV AKI cohort and subject-level split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    CHART_ITEMIDS,
    CREATININE_LABEL_ITEMID,
    HEIGHT_ITEMIDS,
    WEIGHT_ITEMIDS,
    load_baseline_config,
    load_paths,
)


def creatinine_stages(labs: pd.DataFrame, stays: pd.DataFrame, baseline_hours: int) -> pd.DataFrame:
    cr = labs.loc[
        (labs.itemid == CREATININE_LABEL_ITEMID) & labs.valuenum.between(0.1, 150),
        ["hadm_id", "charttime", "valuenum"],
    ].copy()
    joined = cr.merge(stays[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id", how="inner")
    joined = joined[
        (joined.charttime >= joined.intime - pd.Timedelta(hours=baseline_hours))
        & (joined.charttime <= joined.outtime)
    ]
    joined = (
        joined.groupby(["stay_id", "charttime"], as_index=False)
        .valuenum.mean()
        .rename(columns={"valuenum": "creatinine"})
        .sort_values(["stay_id", "charttime"])
    )
    output = []
    window_seconds = baseline_hours * 3600
    for stay_id, group in joined.groupby("stay_id", sort=False):
        group = group.sort_values("charttime")
        times = group.charttime.astype("int64").to_numpy() // 10**9
        values = group.creatinine.to_numpy(float)
        candidates: deque[tuple[int, float]] = deque()
        for timestamp, value, charttime in zip(times, values, group.charttime):
            while candidates and candidates[0][0] < timestamp - window_seconds:
                candidates.popleft()
            baseline = min((x[1] for x in candidates), default=np.nan)
            stage = 0
            if np.isfinite(baseline):
                delta = value - baseline
                ratio = value / baseline if baseline > 0 else np.nan
                if ratio >= 3 or (value >= 4 and (delta >= 0.3 or ratio >= 1.5)):
                    stage = 3
                elif ratio >= 2:
                    stage = 2
                elif delta >= 0.3 or ratio >= 1.5:
                    stage = 1
            output.append((stay_id, charttime, value, baseline, stage))
            candidates.append((timestamp, value))
    return pd.DataFrame(output, columns=["stay_id", "charttime", "creatinine", "creatinine_baseline_48h", "aki_stage_creatinine"])


def prepare_weights(chart: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    weights = chart[chart.itemid.isin(WEIGHT_ITEMIDS) & chart.valuenum.notna()].copy()
    weights.loc[weights.itemid == 226531, "valuenum"] *= 0.45359237
    weights = weights[weights.valuenum.between(20, 500)].sort_values(["stay_id", "charttime"])
    output = {}
    for stay_id, group in weights.groupby("stay_id", sort=False):
        output[int(stay_id)] = (
            group.charttime.astype("int64").to_numpy() // 10**9,
            group.valuenum.to_numpy(float),
        )
    return output


def urine_stages(outputs: pd.DataFrame, chart: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    urine = (
        outputs.groupby(["stay_id", "charttime"], as_index=False).value.sum()
        .rename(columns={"value": "urineoutput"})
        .sort_values(["stay_id", "charttime"])
    )
    intime_map = stays.set_index("stay_id").intime.to_dict()
    weights = prepare_weights(chart)
    rows = []
    for stay_id, group in urine.groupby("stay_id", sort=False):
        stay_id = int(stay_id)
        if stay_id not in weights:
            continue
        wt_times, wt_values = weights[stay_id]
        times = group.charttime.astype("int64").to_numpy() // 10**9
        values = group.urineoutput.to_numpy(float)
        history: deque[tuple[int, float, float]] = deque()
        previous = None
        for timestamp, value, charttime in zip(times, values, group.charttime):
            hours_previous = 1.0 if previous is None else (timestamp - previous) / 3600.0
            previous = timestamp
            history.append((timestamp, value, hours_previous))
            while history and history[0][0] < timestamp - 24 * 3600:
                history.popleft()
            wt_idx = np.searchsorted(wt_times, timestamp, side="right") - 1
            if wt_idx < 0:
                wt_idx = 0
            weight = float(wt_values[wt_idx])
            rates = {}
            coverage = {}
            for hours in (6, 12, 24):
                selected = [entry for entry in history if entry[0] >= timestamp - hours * 3600]
                coverage[hours] = sum(entry[2] for entry in selected)
                total = sum(entry[1] for entry in selected)
                rates[hours] = total / weight / coverage[hours] if coverage[hours] >= hours else np.nan
            stage = 0
            if charttime > intime_map[stay_id] + pd.Timedelta(hours=6):
                if (coverage[24] >= 24 and rates[24] < 0.3) or (coverage[12] >= 12 and rates[12] == 0):
                    stage = 3
                elif coverage[12] >= 12 and rates[12] < 0.5:
                    stage = 2
                elif coverage[6] >= 6 and rates[6] < 0.5:
                    stage = 1
            rows.append((stay_id, charttime, weight, rates[6], rates[12], rates[24], stage))
    return pd.DataFrame(rows, columns=["stay_id", "charttime", "weight", "uo_rate_6h", "uo_rate_12h", "uo_rate_24h", "aki_stage_urine"])


def first_bmi(chart: pd.DataFrame) -> pd.Series:
    weight = chart[chart.itemid.isin(WEIGHT_ITEMIDS)].copy()
    weight.loc[weight.itemid == 226531, "valuenum"] *= 0.45359237
    weight = weight[weight.valuenum.between(20, 500)].sort_values("charttime").groupby("stay_id").valuenum.first()
    height = chart[chart.itemid.isin(HEIGHT_ITEMIDS)].copy()
    height.loc[height.itemid == 226707, "valuenum"] *= 2.54
    height = height[height.valuenum.between(100, 250)].sort_values("charttime").groupby("stay_id").valuenum.first()
    bmi = weight / (height / 100) ** 2
    return bmi.where(bmi.between(10, 80))


def assign_patient_splits(master: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    subjects = master.groupby("subject_id", as_index=False).label.max()
    train, temp = train_test_split(subjects, test_size=0.30, random_state=seed, stratify=subjects.label)
    val, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=temp.label)
    split_frames = {"train": train, "val": val, "test": test}
    split_map = {int(subject_id): name for name, frame in split_frames.items() for subject_id in frame.subject_id}
    out = master.copy()
    out["split"] = out.subject_id.map(split_map)
    return out, split_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper-adapted MIMIC-IV AKI cohort")
    parser.parse_args()
    paths = load_paths()
    cfg = load_baseline_config()
    interim = paths["interim_data_root"] / "mimiciv"
    cohort_dir = paths["cohort_root"]
    cohort_dir.mkdir(parents=True, exist_ok=True)
    stays = pd.read_parquet(interim / "candidate_stays.parquet")
    for col in ["intime", "outtime"]:
        stays[col] = pd.to_datetime(stays[col])
    parquet = lambda path: str(path).replace("\\", "/").replace("'", "''")
    stays_path = parquet(interim / "candidate_stays.parquet")
    labs_path = parquet(interim / "events" / "labevents.parquet")
    chart_path = parquet(interim / "events" / "chartevents.parquet")
    output_path = parquet(interim / "events" / "outputevents.parquet")
    con = duckdb.connect(str(interim / "cohort.duckdb"))
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='10GB'")
    con.execute(f"CREATE OR REPLACE VIEW stays AS SELECT * FROM read_parquet('{stays_path}')")
    con.execute(f"CREATE OR REPLACE VIEW labs AS SELECT * FROM read_parquet('{labs_path}')")
    con.execute(f"CREATE OR REPLACE VIEW chart AS SELECT * FROM read_parquet('{chart_path}')")
    con.execute(f"CREATE OR REPLACE VIEW outputs AS SELECT * FROM read_parquet('{output_path}')")
    con.execute("""
        CREATE OR REPLACE TABLE kdigo_creatinine AS
        WITH cr AS (
          SELECT s.stay_id, l.charttime, avg(l.valuenum) AS creat
          FROM stays s JOIN labs l ON s.hadm_id=l.hadm_id
          WHERE l.itemid=50912 AND l.valuenum BETWEEN 0.1 AND 150
            AND l.charttime >= s.intime - INTERVAL 48 HOUR AND l.charttime <= s.outtime
          GROUP BY s.stay_id, l.charttime
        ), base AS (
          SELECT cr.stay_id, cr.charttime, cr.creat, min(prev.creat) AS baseline
          FROM cr LEFT JOIN cr prev ON cr.stay_id=prev.stay_id
            AND prev.charttime < cr.charttime AND prev.charttime >= cr.charttime - INTERVAL 48 HOUR
          GROUP BY cr.stay_id, cr.charttime, cr.creat
        )
        SELECT *, CASE
          WHEN creat >= baseline*3 THEN 3
          WHEN creat >= 4 AND (creat >= baseline+0.3 OR creat >= baseline*1.5) THEN 3
          WHEN creat >= baseline*2 THEN 2
          WHEN creat >= baseline+0.3 OR creat >= baseline*1.5 THEN 1
          ELSE 0 END AS aki_stage_creatinine
        FROM base
    """)
    con.execute("""
        CREATE OR REPLACE TABLE kdigo_urine AS
        WITH weights AS (
          SELECT stay_id, charttime,
            CASE WHEN itemid=226531 THEN valuenum*0.45359237 ELSE valuenum END AS weight
          FROM chart WHERE itemid IN (224639,226512,226531) AND valuenum > 0
        ), first_weight AS (
          SELECT stay_id, arg_min(weight, charttime) AS first_weight FROM weights
          WHERE weight BETWEEN 20 AND 500 GROUP BY stay_id
        ), uo AS (
          SELECT stay_id, charttime, sum(value) AS urineoutput FROM outputs GROUP BY stay_id, charttime
        ), u1 AS (
          SELECT s.stay_id, s.intime, u.charttime, u.urineoutput,
            epoch(u.charttime)-epoch(s.intime) AS sec,
            coalesce(date_diff('second', lag(u.charttime) OVER (PARTITION BY s.stay_id ORDER BY u.charttime), u.charttime)/3600.0,1.0) AS hours_prev
          FROM stays s JOIN uo u USING(stay_id)
          -- 2026-09-15 fix: urine rows must lie within their own ICU stay. Rows charted after
          -- outtime otherwise turned the post-discharge gap into urine observation time.
          WHERE u.charttime >= s.intime AND u.charttime <= s.outtime
        ), u2 AS (
          SELECT *,
            sum(urineoutput) OVER (PARTITION BY stay_id ORDER BY sec RANGE BETWEEN 21600 PRECEDING AND CURRENT ROW) AS uo6,
            sum(urineoutput) OVER (PARTITION BY stay_id ORDER BY sec RANGE BETWEEN 43200 PRECEDING AND CURRENT ROW) AS uo12,
            sum(urineoutput) OVER (PARTITION BY stay_id ORDER BY sec RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW) AS uo24,
            sum(hours_prev) OVER (PARTITION BY stay_id ORDER BY sec RANGE BETWEEN 21600 PRECEDING AND CURRENT ROW) AS tm6,
            sum(hours_prev) OVER (PARTITION BY stay_id ORDER BY sec RANGE BETWEEN 43200 PRECEDING AND CURRENT ROW) AS tm12,
            sum(hours_prev) OVER (PARTITION BY stay_id ORDER BY sec RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW) AS tm24
          FROM u1
        ), with_weight AS (
          SELECT u2.*, coalesce(w.weight, fw.first_weight) AS weight
          FROM u2 ASOF LEFT JOIN weights w ON u2.stay_id=w.stay_id AND u2.charttime>=w.charttime
          LEFT JOIN first_weight fw ON u2.stay_id=fw.stay_id
        ), rates AS (
          SELECT *, CASE WHEN tm6>=6 THEN uo6/weight/tm6 END AS rt6,
            CASE WHEN tm12>=12 THEN uo12/weight/tm12 END AS rt12,
            CASE WHEN tm24>=24 THEN uo24/weight/tm24 END AS rt24
          FROM with_weight
        )
        SELECT stay_id, charttime, weight, rt6 AS uo_rate_6h, rt12 AS uo_rate_12h, rt24 AS uo_rate_24h,
          CASE WHEN charttime <= intime + INTERVAL 6 HOUR OR weight IS NULL THEN 0
            WHEN tm24>=24 AND rt24<0.3 THEN 3
            WHEN tm12>=12 AND rt12=0 THEN 3
            WHEN tm12>=12 AND rt12<0.5 THEN 2
            WHEN tm6>=6 AND rt6<0.5 THEN 1 ELSE 0 END AS aki_stage_urine
        FROM rates
    """)
    cr_path = parquet(interim / "kdigo_creatinine.parquet")
    uo_path = parquet(interim / "kdigo_urine.parquet")
    for path in [interim / "kdigo_creatinine.parquet", interim / "kdigo_urine.parquet"]:
        if path.exists():
            path.unlink()
    con.execute(f"COPY kdigo_creatinine TO '{cr_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY kdigo_urine TO '{uo_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    onsets = con.execute("""
        SELECT stay_id, min(charttime) AS aki_dt FROM (
          SELECT stay_id, charttime FROM kdigo_creatinine WHERE aki_stage_creatinine>0
          UNION ALL SELECT stay_id, charttime FROM kdigo_urine WHERE aki_stage_urine>0
        ) GROUP BY stay_id
    """).df().set_index("stay_id").aki_dt

    master = stays.copy().set_index("stay_id")
    master["aki_dt"] = onsets
    master["loo"] = (master.aki_dt - master.intime).dt.total_seconds() / 3600
    master["los"] = (master.outtime - master.intime).dt.total_seconds() / 3600
    master["label"] = master.aki_dt.notna().astype(int)
    master["sex"] = master.gender
    bmi = con.execute("""
        WITH anthropometrics AS (
          SELECT stay_id,
            arg_min(CASE WHEN itemid=226531 THEN valuenum*0.45359237 ELSE valuenum END, charttime)
              FILTER (WHERE itemid IN (224639,226512,226531)) AS weight,
            arg_min(CASE WHEN itemid=226707 THEN valuenum*2.54 ELSE valuenum END, charttime)
              FILTER (WHERE itemid IN (226707,226730)) AS height
          FROM chart GROUP BY stay_id
        ) SELECT stay_id, weight/pow(height/100,2) AS bmi FROM anthropometrics
        WHERE weight BETWEEN 20 AND 500 AND height BETWEEN 100 AND 250
          AND weight/pow(height/100,2) BETWEEN 10 AND 80
    """).df().set_index("stay_id").bmi
    master["bmi"] = bmi
    master["visit_id"] = master.index.astype("int64")
    master["admission_dt"] = master.intime
    master["discharge_dt"] = master.outtime

    master_for_sql = master.reset_index()[["stay_id", "intime", "outtime", "aki_dt"]].copy()
    master_for_sql["reference_dt"] = master_for_sql.aki_dt.fillna(master_for_sql.outtime)
    con.register("master_for_sql", master_for_sql)
    vital_counts = con.execute("""
        SELECT m.stay_id, count(DISTINCT c.feature) AS n_vitals
        FROM master_for_sql m JOIN chart c ON m.stay_id=c.stay_id
        WHERE c.feature IN ('sbp','dbp','pulse','resp','spo2','temp')
          AND c.charttime BETWEEN m.intime AND m.reference_dt
        GROUP BY m.stay_id
    """).df().set_index("stay_id").n_vitals
    vital_ok = vital_counts.eq(len(cfg["cohort"]["vital_features"]))
    master["vital_yn"] = vital_ok.reindex(master.index, fill_value=False).astype(int)
    master["error_yn"] = 0

    early_case = master.label.eq(1) & master.loo.lt(cfg["cohort"]["minimum_case_hours_before_onset"])
    short_control = master.label.eq(0) & master.los.lt(cfg["cohort"]["minimum_control_hours"])
    missing_vitals = master.vital_yn.eq(0)
    master["exclusion_reason"] = np.select(
        [early_case, short_control, missing_vitals],
        ["aki_onset_before_96h", "control_stay_under_24h", "missing_required_vital"],
        default="included",
    )
    master["inclusion_yn"] = master.exclusion_reason.eq("included").astype(int)
    master = master.reset_index()
    included = master.query("inclusion_yn == 1").copy()
    included, split_frames = assign_patient_splits(included, cfg["seed"])
    excluded = master.query("inclusion_yn == 0").copy()
    included.to_parquet(cohort_dir / "master_include.parquet", index=False)
    excluded.to_parquet(cohort_dir / "master_exclude.parquet", index=False)
    for split, frame in split_frames.items():
        frame[["subject_id"]].sort_values("subject_id").to_csv(cohort_dir / f"{split}_subjects.csv", index=False)

    summary = {
        "n_subjects": int(included.subject_id.nunique()),
        "n_admissions": int(included.hadm_id.nunique()),
        "n_stays": int(len(included)),
        "n_aki": int(included.label.sum()),
        "n_non_aki": int((included.label == 0).sum()),
        "prevalence": float(included.label.mean()),
        "splits": {
            split: {
                "subjects": int(frame.subject_id.nunique()),
                "stays": int((included.split == split).sum()),
                "aki": int(included.loc[included.split == split, "label"].sum()),
            }
            for split, frame in split_frames.items()
        },
        "exclusion_reasons": excluded.exclusion_reason.value_counts().to_dict(),
    }
    (cohort_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    con.close()


if __name__ == "__main__":
    main()
