"""Shared read-only helpers for the charttime/storetime data-availability audit.

storetime is used as a proxy for when a lab result became available in the record system. It is
not the exact moment a clinician could see it. Inputs are read-only and regenerated outputs go
under the repository's ignored ``work`` directory.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import duckdb
import pandas as pd

AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUDIT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from project_paths import WORK_DIR  # noqa: E402

CM_ROOT = REPO_ROOT / "Step2_Continuous_Monitoring"
BASELINE_ROOT = REPO_ROOT / "Step1_Baseline_Reproduction"
WORK_ROOT = WORK_DIR
CM_WORK_ROOT = WORK_ROOT / "Step2_Continuous_Monitoring"
BASELINE_DATA_ROOT = WORK_ROOT / "Step1_Baseline_Reproduction" / "data"
sys.path.insert(0, str(CM_ROOT / "scripts"))
import cm_adapter as cm  # noqa: E402  (read-only helpers: config, cohort hash)

INTERIM = WORK_ROOT / "Step1_Baseline_Reproduction" / "data" / "interim" / "mimiciv"
INTERIM_LABS = INTERIM / "events" / "labevents.parquet"
STORETIME = WORK_ROOT / "Step3_Data_Availability_Audit" / "data" / "lab_storetime.parquet"
TABLES = WORK_ROOT / "Step3_Data_Availability_Audit" / "tables"
FIGURES = WORK_ROOT / "Step3_Data_Availability_Audit" / "figures"
SUMMARY = WORK_ROOT / "Step3_Data_Availability_Audit" / "summary"
TIMEPOINTS = [-72, -60, -48, -36, -24, -12, 0]
MODES = ["matched", "discharge"]
# Upstream-order label for stays; y_true from the frozen cohort.
GROUP_LABELS = {1: "AKI", 0: "non-AKI"}


def posix(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def lab_feature_scope() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Lab features the models use: feature_list.json (identical across horizons) ∩ lab itemid mapping."""
    lists = [json.loads((WORK_ROOT / f"Step1_Baseline_Reproduction/data/processed/mimiciv/full/horizon_{h}/feature_list.json").read_text()) for h in (0, 48, 72)]
    if not (lists[0] == lists[1] == lists[2]):
        raise RuntimeError("feature_list.json differs across horizons")
    mapping = pd.read_csv(INTERIM / "feature_itemid_mapping.csv")
    labs = mapping[mapping.source.eq("labevents")].copy()
    used = [f for f in lists[0] if f in set(labs.feature)]
    chart = set(mapping.loc[mapping.source.eq("chartevents"), "feature"])
    no_source = [f for f in lists[0] if f not in set(labs.feature) and f not in chart and f not in {"age", "sex", "bmi"}]
    return labs[labs.feature.isin(used)], used, no_source


def load_test_cohort() -> pd.DataFrame:
    paths, monitoring = cm.load_config()
    master = pd.read_parquet(paths["baseline_cohort_root"] / "master_include.parquet")
    test = master[master.split.eq("test")].reset_index(drop=True)
    digest = cm.cohort_sha256(test)
    if digest != monitoring["expected_test_cohort_sha256"] or len(test) != monitoring["expected_test_stays"] or int(test.label.sum()) != monitoring["expected_test_aki"]:
        raise RuntimeError(f"test cohort mismatch: sha={digest} stays={len(test)} aki={int(test.label.sum())}")
    for col in ["intime", "outtime", "aki_dt"]:
        test[col] = pd.to_datetime(test[col])
    return test


def monitoring_grid(mode: str) -> pd.DataFrame:
    """Eligible stay x timepoint cutoffs from phase-two predictions (independent of model/horizon)."""
    frames = []
    for model in ("logistic_regression", "lstm_attention"):
        p = pd.read_parquet(CM_WORK_ROOT / "predictions" / mode / "horizon_0" / f"{model}.parquet",
                            columns=["stay_id", "hadm_id", "monitoring_timepoint", "cutoff_dt", "eligible", "y_true"])
        frames.append(p.sort_values(["stay_id", "monitoring_timepoint"]).reset_index(drop=True))
    if not frames[0].equals(frames[1]):
        raise RuntimeError(f"{mode}: eligibility grid differs between models")
    grid = frames[0]
    grid["cutoff_dt"] = pd.to_datetime(grid.cutoff_dt)
    return grid


def connect(memory: str = "3GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA memory_limit='{memory}'")
    return con


def register_test_labs(con: duckdb.DuckDBPyConnection, test: pd.DataFrame, scope: pd.DataFrame) -> None:
    """View test_labs: in-scope lab rows joined to test stays by hadm_id within the ICU stay window.

    Mirrors the Baseline lab join (hadm_id, intime <= charttime <= bound). delay_min is storetime -
    charttime in minutes; NULL when storetime is missing or unparseable (kept, never dropped).
    """
    con.register("test_stays", test[["stay_id", "hadm_id", "intime", "outtime", "aki_dt", "label"]])
    con.register("scope_items", scope[["itemid", "feature"]].drop_duplicates())
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE test_labs AS
        SELECT s.stay_id, s.label, l.labevent_id, l.itemid, l.feature, l.charttime, l.valuenum,
               r.storetime, r.storetime_raw_missing,
               date_diff('second', l.charttime, r.storetime) / 60.0 AS delay_min
        FROM read_parquet('{posix(INTERIM_LABS)}') l
        JOIN scope_items i ON l.itemid = i.itemid
        JOIN test_stays s ON l.hadm_id = s.hadm_id
        LEFT JOIN read_parquet('{posix(STORETIME)}') r ON l.labevent_id = r.labevent_id
        WHERE l.charttime BETWEEN s.intime AND s.outtime AND isfinite(l.valuenum)
    """)


def write(frame: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / name
    frame.to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} rows)")
    return out
