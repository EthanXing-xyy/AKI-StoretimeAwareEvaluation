"""Recover storetime for every laboratory row used by the corrected baseline.

The interim laboratory table retains labevent_id but not storetime. This
streams hosp/labevents.csv from the raw MIMIC-IV zip (read-only), keeps the rows whose labevent_id
is in the interim table, and writes data/lab_storetime.parquet. Nothing is dropped silently:
missing/unparseable storetime is kept as NaT with a flag, and the join is checked for coverage and
charttime equality (tables/join_integrity.csv).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUDIT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from project_paths import MIMIC_IV_ZIP, WORK_DIR  # noqa: E402

BASELINE_ROOT = REPO_ROOT / "Step1_Baseline_Reproduction"
sys.path.insert(0, str(BASELINE_ROOT))
from src.data.mimic_zip import iter_table  # noqa: E402

MIMIC_ZIP = MIMIC_IV_ZIP
INTERIM_LABS = WORK_DIR / "Step1_Baseline_Reproduction/data/interim/mimiciv/events/labevents.parquet"
MAPPING = WORK_DIR / "Step1_Baseline_Reproduction/data/interim/mimiciv/feature_itemid_mapping.csv"
OUT = WORK_DIR / "Step3_Data_Availability_Audit/data/lab_storetime.parquet"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CHUNK_ROWS = 2_000_000


def main() -> None:
    started = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "Step3_Data_Availability_Audit/tables").mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPPING)
    lab_itemids = set(mapping.loc[mapping.source.eq("labevents"), "itemid"].astype(int))
    interim_ids = np.sort(pq.read_table(INTERIM_LABS, columns=["labevent_id"]).column(0).to_numpy())
    print(f"interim lab rows={len(interim_ids):,} unique ids={len(np.unique(interim_ids)):,} itemids={len(lab_itemids)}", flush=True)

    writer = None
    kept = scanned = raw_missing = unparseable = 0
    try:
        for idx, chunk in enumerate(iter_table(
            MIMIC_ZIP, "hosp/labevents.csv", CHUNK_ROWS,
            usecols=["labevent_id", "itemid", "charttime", "storetime"],
            dtype={"labevent_id": "int64", "itemid": "int32", "charttime": "string", "storetime": "string"},
        ), start=1):
            scanned += len(chunk)
            chunk = chunk[chunk.itemid.isin(lab_itemids)]
            chunk = chunk[np.isin(chunk.labevent_id.to_numpy(), interim_ids, assume_unique=False)]
            if chunk.empty:
                continue
            out = pd.DataFrame({
                "labevent_id": chunk.labevent_id.to_numpy(),
                "raw_itemid": chunk.itemid.to_numpy(),
                "raw_charttime": pd.to_datetime(chunk.charttime, format=TIME_FORMAT, errors="coerce").to_numpy(),
                "storetime": pd.to_datetime(chunk.storetime, format=TIME_FORMAT, errors="coerce").to_numpy(),
                "storetime_raw_missing": chunk.storetime.isna().to_numpy(),
            })
            raw_missing += int(out.storetime_raw_missing.sum())
            unparseable += int((out.storetime.isna() & ~out.storetime_raw_missing).sum())
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT, table.schema, compression="zstd")
            writer.write_table(table)
            kept += len(out)
            if idx % 5 == 0:
                print(f"  chunks={idx} scanned={scanned:,} kept={kept:,} elapsed={time.time()-started:.0f}s", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print(f"[extract] scanned={scanned:,} kept={kept:,} raw_missing_storetime={raw_missing:,} unparseable={unparseable:,}", flush=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='3GB'")
    row = con.execute(f"""
        WITH i AS (SELECT labevent_id, itemid, charttime FROM read_parquet('{INTERIM_LABS.as_posix()}')),
             r AS (SELECT * FROM read_parquet('{OUT.as_posix()}'))
        SELECT
          (SELECT count(*) FROM i) AS interim_rows,
          (SELECT count(DISTINCT labevent_id) FROM i) AS interim_unique_ids,
          (SELECT count(*) FROM r) AS extracted_rows,
          (SELECT count(DISTINCT labevent_id) FROM r) AS extracted_unique_ids,
          (SELECT count(*) FROM i LEFT JOIN r USING(labevent_id) WHERE r.labevent_id IS NULL) AS interim_ids_not_found,
          (SELECT count(*) FROM i JOIN r USING(labevent_id) WHERE i.charttime IS DISTINCT FROM r.raw_charttime) AS charttime_mismatch,
          (SELECT count(*) FROM i JOIN r USING(labevent_id) WHERE i.itemid <> r.raw_itemid) AS itemid_mismatch,
          (SELECT count(*) FROM r WHERE storetime_raw_missing) AS storetime_missing_raw,
          (SELECT count(*) FROM r WHERE storetime IS NULL AND NOT storetime_raw_missing) AS storetime_unparseable
    """).df()
    con.close()
    row["csv_rows_scanned"] = scanned
    row["elapsed_seconds"] = round(time.time() - started, 1)
    row.to_csv(WORK_DIR / "Step3_Data_Availability_Audit/tables/join_integrity.csv", index=False)
    print(json.dumps(row.iloc[0].to_dict(), default=str, indent=2))
    record = row.iloc[0]
    if record.interim_ids_not_found or record.charttime_mismatch or record.itemid_mismatch or record.extracted_unique_ids != record.extracted_rows:
        print("[warn] join integrity issues recorded in tables/join_integrity.csv; downstream steps report them")


if __name__ == "__main__":
    main()
