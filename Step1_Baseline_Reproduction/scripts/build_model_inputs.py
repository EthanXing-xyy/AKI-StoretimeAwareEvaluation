"""Build reusable static and temporal inputs for the five baseline models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    ALL_FEATURES,
    DEMO_FEATURES,
    FEATURE_TO_ID,
    TS_FEATURES,
    load_baseline_config,
    load_paths,
)


VITAL_RANGES = {
    "sbp": (40, 300), "dbp": (20, 200), "pulse": (20, 300),
    "resp": (1, 80), "spo2": (1, 100), "temp": (25, 45),
}


def select_smoke(master: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(master) <= n:
        return master.copy()
    pieces = []
    grouped = master.groupby(["split", "label"], sort=False)
    for _, group in grouped:
        take = max(2, int(round(n * len(group) / len(master))))
        pieces.append(group.sample(n=min(take, len(group)), random_state=seed))
    out = pd.concat(pieces).drop_duplicates("stay_id")
    return out.sort_values("stay_id").reset_index(drop=True)


def prepare_events(master: pd.DataFrame, labs: pd.DataFrame, chart: pd.DataFrame) -> pd.DataFrame:
    keys = master[["stay_id", "hadm_id", "intime", "cutoff_dt"]]
    chart_events = chart[chart.stay_id.isin(set(master.stay_id)) & chart.feature.notna()][
        ["stay_id", "charttime", "feature", "itemid", "valuenum"]
    ].copy()
    chart_events.loc[chart_events.itemid == 223761, "valuenum"] = (
        chart_events.loc[chart_events.itemid == 223761, "valuenum"] - 32
    ) * 5 / 9
    chart_events = chart_events.merge(keys[["stay_id", "intime", "cutoff_dt"]], on="stay_id")
    chart_events = chart_events[chart_events.charttime.between(chart_events.intime, chart_events.cutoff_dt)]
    for feature, (low, high) in VITAL_RANGES.items():
        invalid = chart_events.feature.eq(feature) & ~chart_events.valuenum.between(low, high)
        chart_events = chart_events[~invalid]

    lab_events = labs[labs.hadm_id.isin(set(master.hadm_id))][
        ["hadm_id", "charttime", "feature", "itemid", "valuenum"]
    ].merge(keys, on="hadm_id")
    lab_events = lab_events[lab_events.charttime.between(lab_events.intime, lab_events.cutoff_dt)]
    events = pd.concat([
        chart_events[["stay_id", "charttime", "feature", "valuenum"]],
        lab_events[["stay_id", "charttime", "feature", "valuenum"]],
    ], ignore_index=True)
    events = events.replace([np.inf, -np.inf], np.nan).dropna(subset=["valuenum"])
    return (
        events.groupby(["stay_id", "charttime", "feature"], as_index=False).valuenum.mean()
        .sort_values(["stay_id", "charttime", "feature"])
    )


def scaling_table(master: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    train_ids = set(master.loc[master.split == "train", "stay_id"])
    train_events = events[events.stay_id.isin(train_ids)]
    rows = []
    for feature in ALL_FEATURES:
        if feature == "age":
            values = master.loc[master.split == "train", "age"].dropna().to_numpy(float)
        elif feature == "sex":
            values = master.loc[master.split == "train", "sex_num"].dropna().to_numpy(float)
        elif feature == "bmi":
            values = master.loc[master.split == "train", "bmi"].dropna().to_numpy(float)
        else:
            values = train_events.loc[train_events.feature == feature, "valuenum"].dropna().to_numpy(float)
        if len(values):
            median = float(np.median(values))
            q1, q3 = np.percentile(values, [25, 75])
            iqr = float(q3 - q1)
            if not np.isfinite(iqr) or iqr == 0:
                iqr = 1.0
        else:
            median, iqr = 0.0, 1.0
        rows.append({"feature": feature, "feature_id": FEATURE_TO_ID[feature], "median": median, "iqr": iqr, "n_train_values": int(len(values))})
    return pd.DataFrame(rows)


def build_one(
    row,
    patient_events: pd.DataFrame,
    scale: pd.DataFrame,
    cnn_hours: int,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    params = scale.set_index("feature")[["median", "iqr"]].to_dict("index")
    demo_raw = np.array([row.age, row.sex_num, row.bmi], dtype=np.float32)
    demo_norm = demo_raw.copy()
    for idx, feature in enumerate(DEMO_FEATURES):
        if feature != "sex" and np.isfinite(demo_norm[idx]):
            demo_norm[idx] = (demo_norm[idx] - params[feature]["median"]) / params[feature]["iqr"]
    demo_norm = np.nan_to_num(demo_norm, nan=0.0)

    event_rows = []
    for idx, feature in enumerate(DEMO_FEATURES):
        if np.isfinite(demo_raw[idx]):
            value = demo_raw[idx] if feature == "sex" else demo_norm[idx]
            event_rows.append([FEATURE_TO_ID[feature], 0.0, value])
    flat = {feature: np.nan for feature in TS_FEATURES}
    if not patient_events.empty:
        pe = patient_events.sort_values(["charttime", "feature"]).copy()
        pe["hours"] = (pe.charttime - row.intime).dt.total_seconds() / 3600
        for event in pe.itertuples(index=False):
            value = (event.valuenum - params[event.feature]["median"]) / params[event.feature]["iqr"]
            event_rows.append([FEATURE_TO_ID[event.feature], event.hours, value])
        last = pe.sort_values("charttime").groupby("feature").valuenum.last()
        flat.update(last.to_dict())
    irregular = np.asarray(event_rows, dtype=np.float32)
    if len(irregular) > max_events:
        demo_count = min(3, len(irregular))
        irregular = np.concatenate([irregular[:demo_count], irregular[-(max_events-demo_count):]], axis=0)

    cnn = np.full((cnn_hours, len(TS_FEATURES)), np.nan, dtype=np.float32)
    if not patient_events.empty:
        pe = patient_events.copy()
        pe["bin"] = np.floor((pe.charttime - row.cutoff_dt).dt.total_seconds() / 3600).astype(int) + cnn_hours - 1
        pe = pe[pe.bin.between(0, cnn_hours - 1)]
        hourly = pe.groupby(["bin", "feature"]).valuenum.mean()
        for (hour_bin, feature), value in hourly.items():
            cnn[int(hour_bin), TS_FEATURES.index(feature)] = (value - params[feature]["median"]) / params[feature]["iqr"]
    return irregular, cnn, demo_norm, flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable inputs for all five paper models")
    parser.add_argument("--horizon", type=int, choices=[0, 48, 72], required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = load_paths()
    cfg = load_baseline_config()
    master = pd.read_parquet(paths["cohort_root"] / "master_include.parquet")
    for col in ["intime", "outtime", "aki_dt"]:
        master[col] = pd.to_datetime(master[col])
    if args.scope == "smoke":
        master = select_smoke(master, cfg["processing"]["smoke_admissions"], cfg["seed"])
        master[["subject_id", "hadm_id", "stay_id", "label", "split"]].to_csv(paths["cohort_root"] / "smoke_stays.csv", index=False)
    master["sex_num"] = master.sex.map({"M": 0.0, "F": 1.0})
    master["cutoff_dt"] = np.where(
        master.label.eq(1),
        master.aki_dt - pd.to_timedelta(args.horizon, unit="h"),
        master.outtime,
    )
    master["cutoff_dt"] = pd.to_datetime(master.cutoff_dt)
    interim = paths["interim_data_root"] / "mimiciv" / "events"
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='8GB'")
    con.register("selected_stays", master[["stay_id", "hadm_id", "intime", "cutoff_dt"]])
    labs_path = str(interim / "labevents.parquet").replace("\\", "/").replace("'", "''")
    chart_path = str(interim / "chartevents.parquet").replace("\\", "/").replace("'", "''")
    events = con.execute(f"""
        WITH raw_events AS (
          SELECT c.stay_id, c.charttime, c.feature,
            CASE WHEN c.itemid=223761 THEN (c.valuenum-32)*5/9 ELSE c.valuenum END AS valuenum
          FROM read_parquet('{chart_path}') c JOIN selected_stays s USING(stay_id)
          WHERE c.feature IS NOT NULL AND c.charttime BETWEEN s.intime AND s.cutoff_dt
          UNION ALL
          SELECT s.stay_id, l.charttime, l.feature, l.valuenum
          FROM read_parquet('{labs_path}') l JOIN selected_stays s USING(hadm_id)
          WHERE l.charttime BETWEEN s.intime AND s.cutoff_dt
        )
        SELECT stay_id, charttime, feature, avg(valuenum) AS valuenum FROM raw_events
        WHERE valuenum IS NOT NULL AND isfinite(valuenum)
          AND (feature!='sbp' OR valuenum BETWEEN 40 AND 300)
          AND (feature!='dbp' OR valuenum BETWEEN 20 AND 200)
          AND (feature!='pulse' OR valuenum BETWEEN 20 AND 300)
          AND (feature!='resp' OR valuenum BETWEEN 1 AND 80)
          AND (feature!='spo2' OR valuenum BETWEEN 1 AND 100)
          AND (feature!='temp' OR valuenum BETWEEN 25 AND 45)
        GROUP BY stay_id, charttime, feature ORDER BY stay_id, charttime, feature
    """).df()
    con.close()
    events["charttime"] = pd.to_datetime(events.charttime)
    scale = scaling_table(master, events)

    out_dir = paths["processed_data_root"] / "mimiciv" / args.scope / f"horizon_{args.horizon}"
    sequence_dir = out_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    scale.to_parquet(out_dir / "scaling_info.parquet", index=False)
    master.to_parquet(out_dir / "master.parquet", index=False)
    (out_dir / "feature_list.json").write_text(json.dumps(ALL_FEATURES, indent=2), encoding="utf-8")
    grouped = {int(key): value for key, value in events.groupby("stay_id", sort=False)}
    flat_rows = []
    for idx, row in enumerate(master.itertuples(index=False), start=1):
        output = sequence_dir / f"{int(row.stay_id)}.npz"
        pe = grouped.get(int(row.stay_id), events.iloc[0:0])
        irregular, cnn, demo, flat = build_one(
            row, pe, scale,
            cfg["processing"]["cnn_hours"],
            cfg["processing"]["irregular_max_events"],
        )
        if args.force or not output.exists():
            np.savez_compressed(output, irregular=irregular, cnn=cnn, demo=demo)
        flat_rows.append({"stay_id": int(row.stay_id), **flat, "age": row.age, "sex": row.sex_num, "bmi": row.bmi})
        if idx % 250 == 0:
            print(f"built {idx:,}/{len(master):,} sequences", flush=True)
    pd.DataFrame(flat_rows).to_parquet(out_dir / "flat_features.parquet", index=False)
    print(f"[done] {out_dir}: {len(master):,} stays; events={len(events):,}")


if __name__ == "__main__":
    main()
