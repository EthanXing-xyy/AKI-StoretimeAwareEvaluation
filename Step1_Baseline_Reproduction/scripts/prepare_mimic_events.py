"""Stream required event tables from the local MIMIC-IV archive."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mimic_zip import find_member, iter_table, read_table
from src.utils.config import (
    CHART_ITEMIDS,
    HEIGHT_ITEMIDS,
    LAB_ITEMIDS,
    URINE_ITEMIDS,
    WEIGHT_ITEMIDS,
    load_baseline_config,
    load_paths,
    reverse_item_mapping,
)


def write_parquet_stream(chunks, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"[skip] {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    started = time.time()
    try:
        for idx, chunk in enumerate(chunks, start=1):
            if chunk.empty:
                continue
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            if idx % 10 == 0:
                print(f"  chunks={idx} kept_rows={rows:,} elapsed={time.time()-started:.1f}s", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"No rows selected for {output}")
    print(f"[done] {output}: {rows:,} rows in {time.time()-started:.1f}s")


def build_core(zip_path: Path, interim: Path, minimum_age: int, force: bool) -> pd.DataFrame:
    core_dir = interim / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "admissions": "hosp/admissions.csv",
        "patients": "hosp/patients.csv",
        "icustays": "icu/icustays.csv",
        "d_labitems": "hosp/d_labitems.csv",
        "d_items": "icu/d_items.csv",
    }
    frames = {}
    for name, suffix in tables.items():
        path = core_dir / f"{name}.parquet"
        if path.exists() and not force:
            frames[name] = pd.read_parquet(path)
        else:
            print(f"[read] {suffix}")
            frames[name] = read_table(zip_path, suffix, low_memory=False)
            frames[name].to_parquet(path, index=False)

    icu = frames["icustays"].copy()
    adm = frames["admissions"][["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    pat = frames["patients"][["subject_id", "gender", "anchor_age", "anchor_year"]].copy()
    for col in ["intime", "outtime"]:
        icu[col] = pd.to_datetime(icu[col])
    for col in ["admittime", "dischtime"]:
        adm[col] = pd.to_datetime(adm[col])
    candidate = icu.merge(adm, on=["subject_id", "hadm_id"], how="left").merge(pat, on="subject_id", how="left")
    candidate["age"] = candidate["anchor_age"] + candidate["intime"].dt.year - candidate["anchor_year"]
    candidate["icu_los_hours"] = (candidate["outtime"] - candidate["intime"]).dt.total_seconds() / 3600
    candidate = candidate.query("age >= @minimum_age and icu_los_hours >= 24").copy()
    candidate.to_parquet(interim / "candidate_stays.parquet", index=False)
    print(f"[core] {len(candidate):,} adult ICU stays >=24h; {candidate.subject_id.nunique():,} subjects")
    return candidate


def archive_inventory(paths: dict[str, Path], output: Path) -> None:
    records = []
    for key in ["mimic_zip"]:
        path = paths[key]
        with zipfile.ZipFile(path) as archive:
            records.append({
                "archive": str(path),
                "bytes": path.stat().st_size,
                "entries": len(archive.infolist()),
                "uncompressed_bytes": sum(info.file_size for info in archive.infolist()),
                "members": [info.filename for info in archive.infolist()],
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream only required MIMIC-IV events from the source ZIP")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = load_paths()
    config = load_baseline_config()
    zip_path = paths["mimic_zip"]
    interim = paths["interim_data_root"] / "mimiciv"
    archive_inventory(paths, paths["raw_data_root"] / "archive_inventory.json")
    candidate = build_core(zip_path, interim, config["cohort"]["minimum_age"], args.force)
    stay_ids = set(candidate.stay_id.astype("int64"))
    hadm_ids = set(candidate.hadm_id.astype("int64"))
    chunk_rows = config["processing"]["csv_chunk_rows"]

    lab_map = reverse_item_mapping(LAB_ITEMIDS)
    lab_ids = set(lab_map)
    def lab_chunks():
        for chunk in iter_table(
            zip_path, "hosp/labevents.csv", chunk_rows,
            usecols=["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
            dtype={"hadm_id": "Int64", "itemid": "int32", "valuenum": "float64"},
        ):
            out = chunk[chunk.itemid.isin(lab_ids) & chunk.hadm_id.isin(hadm_ids) & chunk.valuenum.notna()].copy()
            if not out.empty:
                out["feature"] = out.itemid.map(lab_map)
                out["charttime"] = pd.to_datetime(out.charttime)
            yield out
    write_parquet_stream(lab_chunks(), interim / "events" / "labevents.parquet", args.force)

    chart_map = reverse_item_mapping(CHART_ITEMIDS)
    selected_chart_ids = set(chart_map) | set(WEIGHT_ITEMIDS) | set(HEIGHT_ITEMIDS)
    def chart_chunks():
        for chunk in iter_table(
            zip_path, "icu/chartevents.csv", chunk_rows,
            usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"],
            dtype={"stay_id": "Int64", "itemid": "int32", "valuenum": "float64"},
        ):
            out = chunk[chunk.itemid.isin(selected_chart_ids) & chunk.stay_id.isin(stay_ids) & chunk.valuenum.notna()].copy()
            if not out.empty:
                out["feature"] = out.itemid.map(chart_map)
                out["charttime"] = pd.to_datetime(out.charttime)
            yield out
    write_parquet_stream(chart_chunks(), interim / "events" / "chartevents.parquet", args.force)

    def urine_chunks():
        for chunk in iter_table(
            zip_path, "icu/outputevents.csv", chunk_rows,
            usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valueuom"],
            dtype={"stay_id": "Int64", "itemid": "int32", "value": "float64"},
        ):
            out = chunk[chunk.itemid.isin(URINE_ITEMIDS) & chunk.stay_id.isin(stay_ids) & chunk.value.notna()].copy()
            if not out.empty:
                out.loc[(out.itemid == 227488) & (out.value > 0), "value"] *= -1
                out["charttime"] = pd.to_datetime(out.charttime)
            yield out
    write_parquet_stream(urine_chunks(), interim / "events" / "outputevents.parquet", args.force)

    mapping_rows = []
    for feature, itemids in LAB_ITEMIDS.items():
        mapping_rows.extend({"source": "labevents", "feature": feature, "itemid": itemid} for itemid in itemids)
    for feature, itemids in CHART_ITEMIDS.items():
        mapping_rows.extend({"source": "chartevents", "feature": feature, "itemid": itemid} for itemid in itemids)
    pd.DataFrame(mapping_rows).to_csv(interim / "feature_itemid_mapping.csv", index=False)


if __name__ == "__main__":
    main()
