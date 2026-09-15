# %%
"""
Baseline Characteristics Table for ALL Features
================================================
Generates comprehensive Table 1: Baseline Characteristics comparing three cohorts:
- Ilsan (Development)
- CCHLMC (External validation)
- MIMIC-IV (External validation)

Includes: Demographics, ALL Vital Signs, ALL Laboratory Tests
Used for: Model Training/Evaluation Feature Summary
"""

import pandas as pd
import numpy as np
from tableone import TableOne
from typing import List, Optional
from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# %%
# =============================================================================
# FEATURE DEFINITIONS (Same as model training)
# =============================================================================

# Demographics features
DEMO_COL = ["age", "sex", "bmi"]

# Vital sign features
VITAL_COL = ['sbp', 'dbp', 'pulse', 'resp', 'spo2', 'temp']

# Laboratory features (ordered per professor's specification)
LAB_COL = [
    'wbc', 'hemoglobin', 'hematocrit', 'rbc', 'rdw', 'platelet', 'pdw',
    'pt_inr', 'aptt', 'bun', 'creatinine', 'ast', 'alt', 'ggt', 'alp',
    'bilirubin', 'glucose', 'protein', 'albumin', 'uric_acid',
    'calcium', 'phosphorus', 'sodium', 'potassium', 'chloride', 'tco2', 'magnesium',
    'cholesterol', 'hdl', 'triglyceride',
    'po2', 'hco3', 'pco2', 'hba1c', 'ph',
    'esr', 'crp', 'procalcitonin', 'ldh', 'lactate', 'ck', 'bnp',
]

ALL_FEATURES = VITAL_COL + LAB_COL

# Feature labels for display (using Permutation Importance style names)
FEATURE_LABELS = {
    # Demographics
    "age": "Age, years",
    "sex": "Sex, Male n (%)",
    "bmi": "BMI, kg/m²",
    "los_days": "Length of stay, days",
    # Vital Signs
    "sbp": "Systolic blood pressure, mmHg",
    "dbp": "Diastolic blood pressure, mmHg",
    "pulse": "Heart rate, bpm",
    "resp": "Respiratory rate, /min",
    "spo2": "Oxygen saturation, %",
    "temp": "Temperature, °C",
    # Laboratory
    "wbc": "WBC, ×10³/μL",
    "hemoglobin": "Haemoglobin, g/dL",
    "hematocrit": "Haematocrit, %",
    "rbc": "RBC, ×10⁶/μL",
    "rdw": "RDW, %",
    "platelet": "Platelet, ×10³/μL",
    "pdw": "PDW, %",
    "pt_inr": "PT INR",
    "aptt": "aPTT, sec",
    "bun": "BUN, mg/dL",
    "creatinine": "Creatinine, mg/dL",
    "ast": "AST, U/L",
    "alt": "ALT, U/L",
    "ggt": "GGT, U/L",
    "alp": "Alkaline phosphatase, U/L",
    "bilirubin": "Total bilirubin, mg/dL",
    "glucose": "Glucose, mg/dL",
    "protein": "Total protein, g/dL",
    "albumin": "Albumin, g/dL",
    "uric_acid": "Uric acid, mg/dL",
    "calcium": "Calcium, mg/dL",
    "phosphorus": "Phosphate, mg/dL",
    "sodium": "Sodium, mEq/L",
    "potassium": "Potassium, mEq/L",
    "chloride": "Chloride, mEq/L",
    "tco2": "tCO2, mEq/L",
    "magnesium": "Magnesium, mg/dL",
    "cholesterol": "Total cholesterol, mg/dL",
    "hdl": "HDL, mg/dL",
    "triglyceride": "Triglyceride, mg/dL",
    "po2": "pO2, mmHg",
    "hco3": "Bicarbonate, mEq/L",
    "pco2": "pCO2, mmHg",
    "hba1c": "HbA1c, %",
    "ph": "pH",
    "esr": "ESR, mm/hr",
    "crp": "CRP, mg/L",
    "procalcitonin": "Procalcitonin, ng/mL",
    "ldh": "LDH, U/L",
    "lactate": "Lactate, mmol/L",
    "ck": "CK, U/L",
    "bnp": "BNP, pg/mL",
    # Outcome
    "label": "AKI incidence, n (%)",
}

# %%
# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_master_data(
    hospital: str,
    data_root: str = "../../../data",
    prediction_window: int = 0,
) -> pd.DataFrame:
    """
    Load processed master table for a hospital.
    """
    path = f"{data_root}/processed/{hospital}/transformer/prediction_window_{prediction_window}/master.parquet"
    
    df = (
        pd.read_parquet(path)
        .query("inclusion_yn == 1 & vital_yn == 1")
    )
    
    return df


def load_all_lab_values(
    hospital: str,
    visit_ids: pd.Series,
    data_root: str = "../../../data",
) -> pd.DataFrame:
    """
    Load first (admission) laboratory values for each visit.
    """
    path = f"{data_root}/raw/{hospital}/laboratory.parquet"
    
    df_lab = (
        pd.read_parquet(path)
        .query("lab_code.isin(@LAB_COL)")
        .query("visit_id.isin(@visit_ids)")
        .sort_values(["visit_id", "lab_code", "lab_drawn_dt"])
        .groupby(["visit_id", "lab_code"])
        .first()
        .reset_index()
        .pivot(index="visit_id", columns="lab_code", values="lab_value")
        .reset_index()
    )
    
    return df_lab


def load_all_vital_values(
    hospital: str,
    visit_ids: pd.Series,
    data_root: str = "../../../data",
) -> pd.DataFrame:
    """
    Load first (admission) vital sign values for each visit.
    """
    path = f"{data_root}/raw/{hospital}/vitalsign.parquet"
    
    df_vitals = (
        pd.read_parquet(path)
        .query("vital_code.isin(@VITAL_COL)")
        .query("visit_id.isin(@visit_ids)")
        .sort_values(["visit_id", "vital_code", "vital_dt"])
        .groupby(["visit_id", "vital_code"])
        .first()
        .reset_index()
        .pivot(index="visit_id", columns="vital_code", values="vital_value")
        .reset_index()
    )
    
    return df_vitals


# %%
# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def prepare_all_features_data(
    hospital: str,
    data_root: str = "../../../data",
    prediction_window: int = 0,
) -> pd.DataFrame:
    """
    Prepare all feature data for a hospital.
    """
    # Load master data
    df_master = load_master_data(hospital, data_root, prediction_window)
    
    # Load admission labs
    df_labs = load_all_lab_values(hospital, df_master["visit_id"], data_root)
    
    # Load admission vitals
    df_vitals = load_all_vital_values(hospital, df_master["visit_id"], data_root)
    
    # Merge data
    df = (
        df_master
        .merge(df_labs, on="visit_id", how="left")
        .merge(df_vitals, on="visit_id", how="left")
        .assign(
            los_days=lambda x: x["los"] / 24,
            sex_display=lambda x: x["sex"].map({"M": "Male", "F": "Female"}),
            hospital="NHIS" if hospital == "ilsan" else ("CSHH" if hospital == "cchlmc" else "MIMIC-IV"),
        )
    )
    
    return df


def combine_hospital_data(
    data_root: str = "../../../data",
    prediction_window: int = 0,
) -> pd.DataFrame:
    """
    Combine all feature data from all three hospitals.
    """
    hospitals = ["ilsan", "cchlmc", "mimic-iv"]
    
    dfs = []
    for hospital in hospitals:
        print(f"Loading {hospital}...")
        df = prepare_all_features_data(hospital, data_root, prediction_window)
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    hospital_order = ["NHIS", "CSHH", "MIMIC-IV"]
    df_combined["hospital"] = pd.Categorical(
        df_combined["hospital"],
        categories=hospital_order,
        ordered=True
    )
    
    return df_combined


# %%
# =============================================================================
# STATISTICS CALCULATION FUNCTIONS
# =============================================================================

def calculate_feature_stats(
    df: pd.DataFrame,
    feature: str,
    is_categorical: bool = False,
) -> str:
    """
    Calculate statistics for a single feature.
    
    Returns:
        - For continuous: "median [Q1, Q3]" or "mean ± SD"
        - For categorical: "n (%)"
    """
    if is_categorical:
        n = df[feature].sum()
        pct = df[feature].mean() * 100
        return f"{int(n)} ({pct:.1f})"
    else:
        values = df[feature].dropna()
        if len(values) == 0:
            return "N/A"
        
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        return f"{median:.2f} [{q1:.2f}, {q3:.2f}]"


def calculate_feature_stats_mean(
    df: pd.DataFrame,
    feature: str,
) -> str:
    """
    Calculate mean ± SD for a feature.
    """
    values = df[feature].dropna()
    if len(values) == 0:
        return "N/A"
    
    mean = values.mean()
    std = values.std()
    return f"{mean:.2f} ± {std:.2f}"


def create_comprehensive_table_all_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create comprehensive baseline characteristics table with ALL features.
    """
    hospitals = ["NHIS", "CSHH", "MIMIC-IV"]
    
    results = []
    
    for hospital in hospitals:
        df_h = df.query("hospital == @hospital")
        n_total = len(df_h)
        
        row = {"Hospital": hospital, "N": n_total}
        
        # Demographics
        row["Age, years"] = calculate_feature_stats_mean(df_h, "age")
        n_male = (df_h["sex"] == "M").sum()
        pct_male = (df_h["sex"] == "M").mean() * 100
        row["Male, n (%)"] = f"{n_male} ({pct_male:.1f})"
        row["BMI, kg/m²"] = calculate_feature_stats_mean(df_h, "bmi")
        row["Length of stay, days"] = calculate_feature_stats(df_h, "los_days")
        
        # Vital Signs
        for feat in VITAL_COL:
            label = FEATURE_LABELS.get(feat, feat)
            row[label] = calculate_feature_stats_mean(df_h, feat)
        
        # Laboratory Tests
        for feat in LAB_COL:
            label = FEATURE_LABELS.get(feat, feat)
            row[label] = calculate_feature_stats(df_h, feat)  # Use median [IQR] for labs
        
        # Outcome
        n_aki = df_h["label"].sum()
        pct_aki = df_h["label"].mean() * 100
        row["AKI incidence, n (%)"] = f"{int(n_aki)} ({pct_aki:.1f})"
        
        results.append(row)
    
    df_result = pd.DataFrame(results).set_index("Hospital").T
    df_result.columns = [
        "Ilsan (Development)",
        "CCHLMC (External)",
        "MIMIC-IV (External)"
    ]
    
    return df_result


def create_tableone_all_features(
    df: pd.DataFrame,
) -> TableOne:
    """
    Create TableOne with ALL features.
    """
    df = df.copy()
    df["hospital"] = df["hospital"].astype(str)
    
    # All continuous columns
    continuous_cols = ["age", "bmi", "los_days"] + VITAL_COL + LAB_COL
    
    # All available columns in the dataset
    available_cols = [c for c in continuous_cols if c in df.columns]
    available_cols = ["sex_display"] + available_cols + ["label"]
    
    categorical = ["sex_display", "label"]
    
    # Define nonnormal variables (labs - use median [IQR])
    nonnormal = ["los_days"] + [c for c in LAB_COL if c in df.columns]
    
    # Create labels dictionary for available columns
    labels = {}
    for col in available_cols:
        if col in FEATURE_LABELS:
            labels[col] = FEATURE_LABELS[col]
        elif col == "sex_display":
            labels[col] = "Sex, Male n (%)"
        elif col == "los_days":
            labels[col] = "Length of stay, days"
    
    order = {
        "sex_display": ["Male", "Female"],
    }
    
    table = TableOne(
        data=df,
        columns=available_cols,
        categorical=categorical,
        nonnormal=nonnormal,
        groupby="hospital",
        pval=True,
        pval_adjust="bonferroni",
        htest_name=True,
        smd=True,
        missing=True,
        rename=labels,
        order=order,
    )
    
    return table


def calculate_missing_rates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate missing rates for each feature by hospital.
    """
    hospitals = ["NHIS", "CSHH", "MIMIC-IV"]
    
    results = []
    
    all_features = ["age", "bmi"] + VITAL_COL + LAB_COL
    
    for hospital in hospitals:
        df_h = df.query("hospital == @hospital")
        n_total = len(df_h)
        
        row = {"Hospital": hospital, "N": n_total}
        
        for feat in all_features:
            label = FEATURE_LABELS.get(feat, feat)
            if feat in df_h.columns:
                missing_rate = df_h[feat].isna().mean() * 100
                row[label] = f"{missing_rate:.1f}%"
            else:
                row[label] = "N/A"
        
        results.append(row)
    
    df_result = pd.DataFrame(results).set_index("Hospital").T
    df_result.columns = [
        "Ilsan (Development)",
        "CCHLMC (External)",
        "MIMIC-IV (External)"
    ]
    
    return df_result


# %%
# =============================================================================
# MEASUREMENT FREQUENCY FUNCTIONS
# =============================================================================

def calculate_measurement_frequency(
    hospital: str,
    visit_ids: pd.Series,
    data_root: str = "../../../data",
) -> pd.DataFrame:
    """
    Calculate per-visit measurement frequency for each feature.

    For each feature (vital sign / lab test), count the number of measurements
    per visit during the admission period.

    Args:
        hospital: Hospital name ('ilsan', 'cchlmc', 'mimic-iv')
        visit_ids: Series of visit_ids to include
        data_root: Root path to data directory

    Returns:
        DataFrame with visit_id as index, features as columns, values = measurement counts
    """
    # Load vital sign counts per visit per feature
    vs_path = f"{data_root}/raw/{hospital}/vitalsign.parquet"
    df_vs = (
        pd.read_parquet(vs_path)
        .query("vital_code.isin(@VITAL_COL)")
        .query("visit_id.isin(@visit_ids)")
        .groupby(["visit_id", "vital_code"])
        .size()
        .reset_index(name="count")
        .pivot(index="visit_id", columns="vital_code", values="count")
    )

    # Load lab counts per visit per feature
    lab_path = f"{data_root}/raw/{hospital}/laboratory.parquet"
    df_lab = (
        pd.read_parquet(lab_path)
        .query("lab_code.isin(@LAB_COL)")
        .query("visit_id.isin(@visit_ids)")
        .groupby(["visit_id", "lab_code"])
        .size()
        .reset_index(name="count")
        .pivot(index="visit_id", columns="lab_code", values="count")
    )

    # Combine: fill missing features with 0 (no measurements)
    df_freq = (
        pd.DataFrame(index=visit_ids)
        .join(df_vs, how="left")
        .join(df_lab, how="left")
        .fillna(0)
        .astype(int)
    )

    return df_freq


def calculate_measurement_frequency_table(
    data_root: str = "../../../data",
    prediction_window: int = 0,
) -> pd.DataFrame:
    """
    Calculate measurement frequency summary (median [IQR]) for each feature by hospital.

    Returns:
        DataFrame with features as rows and hospitals as columns.
        Each cell: "median [Q1, Q3]" of per-visit measurement counts.
    """
    hospitals_map = {
        "ilsan": "NHIS",
        "cchlmc": "CSHH",
        "mimic-iv": "MIMIC-IV",
    }

    all_features = VITAL_COL + LAB_COL
    results = []

    for hospital, label in hospitals_map.items():
        print(f"  Calculating measurement frequency for {hospital}...")
        df_master = load_master_data(hospital, data_root, prediction_window)
        visit_ids = df_master["visit_id"]

        df_freq = calculate_measurement_frequency(hospital, visit_ids, data_root)

        row = {"Hospital": label, "N": len(visit_ids)}

        for feat in all_features:
            if feat in df_freq.columns:
                counts = df_freq[feat]
                median = counts.median()
                q1 = counts.quantile(0.25)
                q3 = counts.quantile(0.75)
                n_measured = (counts > 0).sum()
                pct_measured = (counts > 0).mean() * 100
                row[FEATURE_LABELS.get(feat, feat)] = (
                    f"{median:.0f} [{q1:.0f}, {q3:.0f}] (n={n_measured}, {pct_measured:.1f}%)"
                )
            else:
                row[FEATURE_LABELS.get(feat, feat)] = "N/A"

        results.append(row)

    df_result = pd.DataFrame(results).set_index("Hospital").T
    df_result.columns = [
        "Ilsan (Development)",
        "CCHLMC (External)",
        "MIMIC-IV (External)",
    ]

    return df_result


# %%
# =============================================================================
# SAVE FUNCTIONS
# =============================================================================

def save_to_xlsx(
    table1: TableOne,
    comprehensive_table: pd.DataFrame,
    missing_table: pd.DataFrame,
    frequency_table: pd.DataFrame,
    savepath: str,
) -> None:
    """
    Save all baseline characteristics to xlsx file.
    """
    import os
    os.makedirs(savepath, exist_ok=True)

    filepath = f"{savepath}/table1_baseline_all_features.xlsx"

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: TableOne output
        table1_df = table1.tableone
        table1_df.to_excel(writer, sheet_name='TableOne')

        # Sheet 2: Comprehensive table
        comprehensive_table.to_excel(writer, sheet_name='Summary')

        # Sheet 3: Missing rates
        missing_table.to_excel(writer, sheet_name='Missing_Rates')

        # Sheet 4: Measurement frequency per feature
        frequency_table.to_excel(writer, sheet_name='Measurement_Frequency')

    print(f"\nBaseline characteristics (all features) saved to: {filepath}")


# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Set paths
    DATA_ROOT = "../../../data"
    SAVE_PATH = "../../../result/draft/table"
    
    # Combine data from all hospitals
    print("Loading and combining data from all hospitals...")
    df_combined = combine_hospital_data(DATA_ROOT, prediction_window=0)
    
    print(f"\nTotal samples: {len(df_combined)}")
    print(f"By hospital:\n{df_combined['hospital'].value_counts()}")
    
    # Create TableOne with ALL features
    print("\n" + "="*60)
    print("Table 1: Baseline Characteristics - ALL Features (TableOne)")
    print("="*60)
    table1 = create_tableone_all_features(df_combined)
    display(table1)
    
    # Create comprehensive table
    print("\n" + "="*60)
    print("Comprehensive Baseline Characteristics (ALL Features)")
    print("="*60)
    comprehensive_table = create_comprehensive_table_all_features(df_combined)
    display(comprehensive_table)
    
    # Calculate missing rates
    print("\n" + "="*60)
    print("Missing Rates by Feature")
    print("="*60)
    missing_table = calculate_missing_rates(df_combined)
    display(missing_table)
    
    # Calculate measurement frequency per feature
    print("\n" + "="*60)
    print("Measurement Frequency by Feature (per visit)")
    print("="*60)
    frequency_table = calculate_measurement_frequency_table(DATA_ROOT, prediction_window=0)
    display(frequency_table)

    # Save to xlsx
    save_to_xlsx(table1, comprehensive_table, missing_table, frequency_table, SAVE_PATH)

    print("\nDone!")

# %%
