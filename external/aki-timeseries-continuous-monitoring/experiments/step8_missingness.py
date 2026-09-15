# %%
"""
Step 8: Variable-Level Missingness Analysis
Reviewer 1 #6 + Reviewer 3 Minor 5.
Computes missing rates for all 51 features per dataset.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from utils.config import (
    SEED, PROJECT_ROOT, HOSPITALS,
    VITAL_COL, LAB_COL, DEMO_COL, ALL_FEATURES,
    PROCESSED_DIR, RAW_DIR, RESULTS_DIR,
    DATASET_DISPLAY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_OUT = RESULTS_DIR / "missingness"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)

HOSPITAL_DISPLAY = {
    "ilsan": "NHIS",
    "cchlmc": "CSHH",
    "mimic-iv": "MIMIC-IV",
}

# Map raw data columns to our feature names
# Vitals and labs may be stored in long format (vital_code/lab_code columns)
# or wide format — we need to check both


# ── 8-A: Missingness from raw data ──────────────────────────────────

def compute_raw_missingness():
    """Compute feature-level missingness from raw parquet data per hospital."""
    logger.info("=" * 60)
    logger.info("8-A: Raw Data Missingness")
    logger.info("=" * 60)

    all_rows = []

    for hospital in HOSPITALS:
        logger.info(f"\n--- {hospital} ({HOSPITAL_DISPLAY[hospital]}) ---")

        master_path = PROCESSED_DIR / hospital / "cohort" / "master_include.parquet"
        if not master_path.exists():
            logger.warning(f"  No master file for {hospital}")
            continue
        master = pd.read_parquet(master_path)
        total_patients = len(master)
        visit_ids = set(master["visit_id"])
        logger.info(f"  Total patients: {total_patients}")

        # ── Vitals ──
        vital_path = RAW_DIR / hospital / "vitalsign.parquet"
        if vital_path.exists():
            vitals = pd.read_parquet(vital_path)
            vitals = vitals[vitals["visit_id"].isin(visit_ids)]
            logger.info(f"  Vital records: {len(vitals)}")

            # Check format: long (vital_code column) vs wide
            if "vital_code" in vitals.columns:
                # Long format
                for feat in VITAL_COL:
                    feat_data = vitals[vitals["vital_code"] == feat]
                    n_timepoints_total = len(vitals[vitals["vital_code"] == feat]) if "vital_code" in vitals.columns else 0
                    patients_with = feat_data["visit_id"].nunique()
                    patients_never = total_patients - patients_with

                    # Timepoint-level: missing if value is NaN
                    if "vital_value" in feat_data.columns:
                        n_missing_tp = feat_data["vital_value"].isna().sum()
                        n_total_tp = len(feat_data)
                    else:
                        n_missing_tp = 0
                        n_total_tp = 0

                    all_rows.append({
                        "feature_name": feat,
                        "feature_type": "vital",
                        "hospital": hospital,
                        "hospital_display": HOSPITAL_DISPLAY[hospital],
                        "n_total_patients": total_patients,
                        "n_patients_with_measurement": patients_with,
                        "n_patients_never_measured": patients_never,
                        "missing_rate_patient_pct": patients_never / total_patients * 100 if total_patients > 0 else 0,
                        "n_total_timepoints": n_total_tp,
                        "n_missing_timepoints": int(n_missing_tp),
                        "missing_rate_timepoint_pct": float(n_missing_tp / n_total_tp * 100) if n_total_tp > 0 else 100.0,
                    })
            else:
                # Wide format — each column is a vital sign
                for feat in VITAL_COL:
                    if feat in vitals.columns:
                        n_total_tp = len(vitals)
                        n_missing_tp = vitals[feat].isna().sum()
                        patients_with = vitals.dropna(subset=[feat])["visit_id"].nunique()
                        patients_never = total_patients - patients_with
                    else:
                        n_total_tp = 0
                        n_missing_tp = 0
                        patients_with = 0
                        patients_never = total_patients

                    all_rows.append({
                        "feature_name": feat,
                        "feature_type": "vital",
                        "hospital": hospital,
                        "hospital_display": HOSPITAL_DISPLAY[hospital],
                        "n_total_patients": total_patients,
                        "n_patients_with_measurement": patients_with,
                        "n_patients_never_measured": patients_never,
                        "missing_rate_patient_pct": patients_never / total_patients * 100,
                        "n_total_timepoints": n_total_tp,
                        "n_missing_timepoints": int(n_missing_tp),
                        "missing_rate_timepoint_pct": float(n_missing_tp / n_total_tp * 100) if n_total_tp > 0 else 100.0,
                    })

        # ── Labs ──
        lab_path = RAW_DIR / hospital / "laboratory.parquet"
        if lab_path.exists():
            labs = pd.read_parquet(lab_path)
            labs = labs[labs["visit_id"].isin(visit_ids)]
            logger.info(f"  Lab records: {len(labs)}")

            if "lab_code" in labs.columns:
                # Long format
                for feat in LAB_COL:
                    feat_data = labs[labs["lab_code"] == feat]
                    patients_with = feat_data["visit_id"].nunique()
                    patients_never = total_patients - patients_with

                    if "lab_value" in feat_data.columns:
                        n_missing_tp = feat_data["lab_value"].isna().sum()
                        n_total_tp = len(feat_data)
                    else:
                        n_missing_tp = 0
                        n_total_tp = 0

                    all_rows.append({
                        "feature_name": feat,
                        "feature_type": "lab",
                        "hospital": hospital,
                        "hospital_display": HOSPITAL_DISPLAY[hospital],
                        "n_total_patients": total_patients,
                        "n_patients_with_measurement": patients_with,
                        "n_patients_never_measured": patients_never,
                        "missing_rate_patient_pct": patients_never / total_patients * 100,
                        "n_total_timepoints": n_total_tp,
                        "n_missing_timepoints": int(n_missing_tp),
                        "missing_rate_timepoint_pct": float(n_missing_tp / n_total_tp * 100) if n_total_tp > 0 else 100.0,
                    })
            else:
                for feat in LAB_COL:
                    if feat in labs.columns:
                        n_total_tp = len(labs)
                        n_missing_tp = labs[feat].isna().sum()
                        patients_with = labs.dropna(subset=[feat])["visit_id"].nunique()
                        patients_never = total_patients - patients_with
                    else:
                        n_total_tp = 0
                        n_missing_tp = 0
                        patients_with = 0
                        patients_never = total_patients

                    all_rows.append({
                        "feature_name": feat,
                        "feature_type": "lab",
                        "hospital": hospital,
                        "hospital_display": HOSPITAL_DISPLAY[hospital],
                        "n_total_patients": total_patients,
                        "n_patients_with_measurement": patients_with,
                        "n_patients_never_measured": patients_never,
                        "missing_rate_patient_pct": patients_never / total_patients * 100,
                        "n_total_timepoints": n_total_tp,
                        "n_missing_timepoints": int(n_missing_tp),
                        "missing_rate_timepoint_pct": float(n_missing_tp / n_total_tp * 100) if n_total_tp > 0 else 100.0,
                    })

        # ── Demographics ──
        for feat in DEMO_COL:
            if feat in master.columns:
                n_missing = master[feat].isna().sum()
                patients_with = total_patients - n_missing
            else:
                n_missing = total_patients
                patients_with = 0

            all_rows.append({
                "feature_name": feat,
                "feature_type": "demographic",
                "hospital": hospital,
                "hospital_display": HOSPITAL_DISPLAY[hospital],
                "n_total_patients": total_patients,
                "n_patients_with_measurement": patients_with,
                "n_patients_never_measured": int(n_missing),
                "missing_rate_patient_pct": float(n_missing / total_patients * 100),
                "n_total_timepoints": total_patients,
                "n_missing_timepoints": int(n_missing),
                "missing_rate_timepoint_pct": float(n_missing / total_patients * 100),
            })

        logger.info(f"  {hospital}: {len([r for r in all_rows if r['hospital'] == hospital])} feature entries")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(RESULTS_OUT / "raw_missingness.csv", index=False)
    logger.info(f"\nSaved raw missingness: {len(df_out)} rows")
    return df_out


# ── 8-B: MIMIC-IV NaN investigation ─────────────────────────────────

def investigate_mimic_nan():
    """Check specific features flagged as NaN in MIMIC-IV Table S1."""
    logger.info("=" * 60)
    logger.info("8-B: MIMIC-IV NaN Feature Investigation")
    logger.info("=" * 60)

    flagged_features = ["cholesterol", "pdw", "procalcitonin"]
    report_lines = ["# MIMIC-IV NaN Feature Investigation\n"]

    lab_path = RAW_DIR / "mimic-iv" / "laboratory.parquet"
    if not lab_path.exists():
        report_lines.append("Laboratory data not found for MIMIC-IV.\n")
    else:
        labs = pd.read_parquet(lab_path)
        master = pd.read_parquet(PROCESSED_DIR / "mimic-iv" / "cohort" / "master_include.parquet")
        visit_ids = set(master["visit_id"])
        labs_cohort = labs[labs["visit_id"].isin(visit_ids)]

        if "lab_code" in labs.columns:
            all_lab_codes = labs["lab_code"].unique()
            cohort_lab_codes = labs_cohort["lab_code"].unique()

            for feat in flagged_features:
                report_lines.append(f"\n## {feat}")
                in_raw = feat in all_lab_codes
                in_cohort = feat in cohort_lab_codes

                if not in_raw:
                    report_lines.append(f"- **Status**: Feature `{feat}` does NOT exist in raw MIMIC-IV laboratory data")
                    report_lines.append(f"- **Reason**: This variable is not routinely measured or recorded in MIMIC-IV")
                    report_lines.append(f"- **Recommendation**: Report as 'Not available' in Table S1")
                elif not in_cohort:
                    report_lines.append(f"- **Status**: Feature `{feat}` exists in raw data but NOT in cohort subset")
                    n_raw = len(labs[labs["lab_code"] == feat])
                    report_lines.append(f"- Raw records (all patients): {n_raw}")
                    report_lines.append(f"- **Reason**: None of the cohort patients have this measurement")
                else:
                    feat_data = labs_cohort[labs_cohort["lab_code"] == feat]
                    n_records = len(feat_data)
                    n_patients = feat_data["visit_id"].nunique()
                    n_total = len(master)
                    pct = n_patients / n_total * 100
                    report_lines.append(f"- **Status**: Feature exists in cohort data")
                    report_lines.append(f"- Records: {n_records} across {n_patients}/{n_total} patients ({pct:.1f}%)")
                    if "lab_value" in feat_data.columns:
                        nan_pct = feat_data["lab_value"].isna().mean() * 100
                        report_lines.append(f"- Value-level NaN: {nan_pct:.1f}%")
        else:
            for feat in flagged_features:
                report_lines.append(f"\n## {feat}")
                if feat in labs.columns:
                    n_records = labs_cohort[feat].notna().sum()
                    report_lines.append(f"- Records with non-null values: {n_records}")
                else:
                    report_lines.append(f"- **Column `{feat}` not found in MIMIC-IV data**")

    report = "\n".join(report_lines)
    (RESULTS_OUT / "mimic_iv_nan_investigation.md").write_text(report)
    logger.info("Saved MIMIC-IV NaN investigation report")
    return report


# ── 8-C: Table S1 supplement (formatted) ─────────────────────────────

def generate_table_s1_supplement(miss_df):
    """Create formatted missing rate table for paper supplementary."""
    logger.info("=" * 60)
    logger.info("8-C: Table S1 Supplement")
    logger.info("=" * 60)

    # Pivot: feature × hospital → missing rate (patient level)
    pivot_patient = miss_df.pivot_table(
        index=["feature_name", "feature_type"],
        columns="hospital_display",
        values="missing_rate_patient_pct",
        aggfunc="first",
    ).round(1)

    # Reorder columns
    col_order = ["NHIS", "CSHH", "MIMIC-IV"]
    pivot_patient = pivot_patient[[c for c in col_order if c in pivot_patient.columns]]

    # Sort by feature type then name
    type_order = {"vital": 0, "lab": 1, "demographic": 2}
    pivot_patient = pivot_patient.reset_index()
    pivot_patient["sort_key"] = pivot_patient["feature_type"].map(type_order)
    pivot_patient = pivot_patient.sort_values(["sort_key", "feature_name"]).drop("sort_key", axis=1)

    # Save CSV
    pivot_patient.to_csv(RESULTS_OUT / "table_s1_missingness.csv", index=False)

    # Generate markdown table
    md_lines = [
        "| Feature | Type | NHIS (%) | CSHH (%) | MIMIC-IV (%) |",
        "|---------|------|----------|----------|--------------|",
    ]
    for _, row in pivot_patient.iterrows():
        vals = []
        for col in col_order:
            if col in pivot_patient.columns:
                v = row.get(col, "N/A")
                vals.append(f"{v:.1f}" if isinstance(v, (float, int)) and not pd.isna(v) else "N/A")
            else:
                vals.append("N/A")
        md_lines.append(f"| {row['feature_name']} | {row['feature_type']} | {' | '.join(vals)} |")

    md_text = "\n".join(md_lines)
    (RESULTS_OUT / "table_s1_missingness.md").write_text(md_text)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Feature-level patient missingness rates (\%)}",
        r"\label{tab:missingness}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Feature & Type & NHIS & CSHH & MIMIC-IV \\",
        r"\hline",
    ]
    for _, row in pivot_patient.iterrows():
        vals = []
        for col in col_order:
            if col in pivot_patient.columns:
                v = row.get(col, "")
                vals.append(f"{v:.1f}" if isinstance(v, (float, int)) and not pd.isna(v) else "N/A")
            else:
                vals.append("N/A")
        latex_lines.append(f"{row['feature_name']} & {row['feature_type']} & {' & '.join(vals)} \\\\")
    latex_lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    latex_text = "\n".join(latex_lines)
    (RESULTS_OUT / "table_s1_missingness.tex").write_text(latex_text)

    logger.info(f"Saved Table S1 supplement (CSV, markdown, LaTeX)")
    return pivot_patient


# ── Main ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Step 8: Missingness Analysis")
    logger.info("=" * 70)

    miss_df = compute_raw_missingness()
    investigate_mimic_nan()
    generate_table_s1_supplement(miss_df)

    logger.info("\nStep 8 Complete!")


if __name__ == "__main__":
    main()
