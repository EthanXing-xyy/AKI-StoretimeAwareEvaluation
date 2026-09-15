# %%
"""
Baseline Characteristics Table for AKI Prediction Paper
========================================================
Generates Table 1: Baseline Characteristics comparing three cohorts:
- Ilsan (Development)
- CCHLMC (External validation)
- MIMIC-IV (External validation)

Using TableOne package with SMD for multi-group comparison.
Includes: Demographics, Vitals, AKI incidence, AKI stage, and Urine output criteria.
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
# DATA LOADING FUNCTIONS
# =============================================================================

def load_master_data(
    hospital: str,
    data_root: str = "../../../data",
    prediction_window: int = 0,
) -> pd.DataFrame:
    """
    Load processed master table for a hospital.
    
    Args:
        hospital: Hospital name ('ilsan', 'cchlmc', 'mimic-iv')
        data_root: Root path to data directory
        prediction_window: Prediction window (0, 48, 72)
        
    Returns:
        DataFrame with master data
    """
    path = f"{data_root}/processed/{hospital}/transformer/prediction_window_{prediction_window}/master.parquet"
    
    df = (
        pd.read_parquet(path)
        .query("inclusion_yn == 1 & vital_yn == 1")
    )
    
    return df


def load_admission_creatinine(
    hospital: str,
    visit_ids: pd.Series,
    data_root: str = "../../../data",
) -> pd.DataFrame:
    """
    Load first (admission) creatinine value for each visit.
    
    Args:
        hospital: Hospital name
        visit_ids: Series of visit_ids to filter
        data_root: Root path to data directory
        
    Returns:
        DataFrame with visit_id and admission_creatinine
    """
    path = f"{data_root}/raw/{hospital}/laboratory.parquet"
    
    df_cr = (
        pd.read_parquet(path)
        .query("lab_code == 'creatinine'")
        .query("visit_id.isin(@visit_ids)")
        .sort_values(["visit_id", "lab_drawn_dt"])
        .groupby("visit_id")
        .first()
        .reset_index()
        [["visit_id", "lab_value"]]
        .rename(columns={"lab_value": "admission_cr"})
    )
    
    return df_cr


def load_admission_vitals(
    hospital: str,
    visit_ids: pd.Series,
    data_root: str = "../../../data",
) -> pd.DataFrame:
    """
    Load first (admission) vital sign values for each visit.
    
    Args:
        hospital: Hospital name
        visit_ids: Series of visit_ids to filter
        data_root: Root path to data directory
        
    Returns:
        DataFrame with visit_id and vital sign values
    """
    path = f"{data_root}/raw/{hospital}/vitalsign.parquet"
    
    # Common vital signs across all hospitals
    vital_codes = ['sbp', 'dbp', 'pulse', 'resp', 'spo2', 'temp']
    
    df_vitals = (
        pd.read_parquet(path)
        .query("vital_code.isin(@vital_codes)")
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

def prepare_baseline_data(
    hospital: str,
    data_root: str = "../../../data",
    prediction_window: int = 0,
) -> pd.DataFrame:
    """
    Prepare baseline characteristics data for a hospital.
    
    Args:
        hospital: Hospital name
        data_root: Root path to data directory
        prediction_window: Prediction window
        
    Returns:
        DataFrame with all baseline characteristics
    """
    # Load master data
    df_master = load_master_data(hospital, data_root, prediction_window)
    
    # Load admission creatinine
    df_cr = load_admission_creatinine(hospital, df_master["visit_id"], data_root)
    
    # Load admission vital signs
    df_vitals = load_admission_vitals(hospital, df_master["visit_id"], data_root)
    
    # Merge data
    df = (
        df_master
        .merge(df_cr, on="visit_id", how="left")
        .merge(df_vitals, on="visit_id", how="left")
        .assign(
            # Convert LOS from hours to days
            los_days=lambda x: x["los"] / 24,
            # Sex as Male indicator (use string for clearer TableOne output)
            sex_male=lambda x: x["sex"].map({"M": "Male", "F": "Female"}),
            # AKI incidence (use string for clearer TableOne output)
            aki_incidence=lambda x: x["label"].map({1: "Yes", 0: "No"}),
            # AKI Stage as categorical (for all patients, NA for non-AKI)
            aki_stage_cat=lambda x: x.apply(
                lambda row: f"Stage {int(row['aki_stage'])}" if row['label'] == 1 and pd.notna(row['aki_stage']) else np.nan,
                axis=1
            ),
            # Urine output criteria met (for AKI patients only)
            urine_criteria=lambda x: x.apply(
                lambda row: "Yes" if row['label'] == 1 and row['urine_yn'] == 1 else 
                           ("No" if row['label'] == 1 else np.nan),
                axis=1
            ),
            # Hospital indicator for grouping
            hospital="NHIS" if hospital == "ilsan" else ("CSHH" if hospital == "cchlmc" else "MIMIC-IV"),
        )
    )
    
    return df


def combine_hospital_data(
    data_root: str = "../../../data",
    prediction_window: int = 0,
) -> pd.DataFrame:
    """
    Combine baseline data from all three hospitals.
    
    Args:
        data_root: Root path to data directory
        prediction_window: Prediction window
        
    Returns:
        Combined DataFrame with hospital indicator
    """
    hospitals = ["ilsan", "cchlmc", "mimic-iv"]
    
    dfs = []
    for hospital in hospitals:
        df = prepare_baseline_data(hospital, data_root, prediction_window)
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Create ordered categorical for hospital
    hospital_order = ["NHIS", "CSHH", "MIMIC-IV"]
    df_combined["hospital"] = pd.Categorical(
        df_combined["hospital"],
        categories=hospital_order,
        ordered=True
    )
    
    return df_combined


# %%
# =============================================================================
# TABLE ONE GENERATION
# =============================================================================

def create_baseline_table(
    df: pd.DataFrame,
) -> TableOne:
    """
    Create Table 1: Baseline Characteristics using TableOne.
    
    Args:
        df: Combined DataFrame with all hospitals
        
    Returns:
        TableOne object
    """
    # Convert hospital to string to avoid categorical issues
    df = df.copy()
    df["hospital"] = df["hospital"].astype(str)
    
    # Define columns for the table
    columns = [
        # Demographics
        "age",
        "sex_male",
        "bmi",
        # Hospitalization
        "los_days",
        # Vital Signs
        "sbp",
        "dbp",
        "pulse",
        "resp",
        "spo2",
        "temp",
        # Baseline Kidney
        "admission_cr",
        # Outcomes
        "aki_incidence",
        "aki_stage_cat",
        "urine_criteria",
    ]
    
    # Define categorical variables
    categorical = [
        "sex_male",
        "aki_incidence",
        "aki_stage_cat",
        "urine_criteria",
    ]
    
    # Define nonnormal variables (will use median [IQR])
    nonnormal = [
        "los_days",
        "admission_cr",
    ]
    
    # Define labels for better display
    labels = {
        "age": "Age, years",
        "sex_male": "Sex",
        "bmi": "BMI, kg/m²",
        "los_days": "Length of stay, days",
        "sbp": "SBP, mmHg",
        "dbp": "DBP, mmHg",
        "pulse": "Heart rate, bpm",
        "resp": "Respiratory rate, /min",
        "spo2": "SpO2, %",
        "temp": "Temperature, °C",
        "admission_cr": "Admission creatinine, mg/dL",
        "aki_incidence": "AKI incidence",
        "aki_stage_cat": "AKI stage (among AKI patients)",
        "urine_criteria": "Urine output criteria (among AKI patients)",
    }
    
    # Create order mapping for categorical variables
    order = {
        "sex_male": ["Male", "Female"],
        "aki_incidence": ["Yes", "No"],
        "aki_stage_cat": ["Stage 1", "Stage 2", "Stage 3"],
        "urine_criteria": ["Yes", "No"],
    }
    
    # Create TableOne with SMD for comparison
    table = TableOne(
        data=df,
        columns=columns,
        categorical=categorical,
        nonnormal=nonnormal,
        groupby="hospital",
        pval=True,
        pval_adjust="bonferroni",
        htest_name=True,
        smd=True,
        missing=False,  # Don't show missing for cleaner output
        rename=labels,
        order=order,
    )
    
    return table


def create_comprehensive_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create comprehensive baseline characteristics table with AKI stages and urine output.
    
    Args:
        df: Combined DataFrame with all hospitals
        
    Returns:
        DataFrame with all characteristics
    """
    hospitals = ["NHIS", "CSHH", "MIMIC-IV"]
    
    results = []
    
    for hospital in hospitals:
        df_h = df.query("hospital == @hospital")
        df_aki = df_h.query("label == 1")
        
        n_total = len(df_h)
        n_aki = len(df_aki)
        
        n_male = (df_h['sex'] == 'M').sum()
        pct_male = (df_h['sex'] == 'M').mean() * 100
        
        # Urine output criteria (among AKI patients)
        n_urine = df_aki['urine_yn'].sum() if n_aki > 0 else 0
        pct_urine = (df_aki['urine_yn'].mean() * 100) if n_aki > 0 else 0.0
        
        row = {
            "Hospital": hospital,
            "N": n_total,
            # Demographics
            "Age, mean (SD)": f"{df_h['age'].mean():.1f} ({df_h['age'].std():.1f})",
            "Male, n (%)": f"{n_male} ({pct_male:.1f}%)",
            "BMI, mean (SD)": f"{df_h['bmi'].mean():.1f} ({df_h['bmi'].std():.1f})",
            # Hospitalization
            "LOS, median [IQR]": f"{df_h['los_days'].median():.1f} [{df_h['los_days'].quantile(0.25):.1f}-{df_h['los_days'].quantile(0.75):.1f}]",
            # Vital Signs
            "SBP, mean (SD)": f"{df_h['sbp'].mean():.1f} ({df_h['sbp'].std():.1f})",
            "DBP, mean (SD)": f"{df_h['dbp'].mean():.1f} ({df_h['dbp'].std():.1f})",
            "Heart rate, mean (SD)": f"{df_h['pulse'].mean():.1f} ({df_h['pulse'].std():.1f})",
            "Respiratory rate, mean (SD)": f"{df_h['resp'].mean():.1f} ({df_h['resp'].std():.1f})",
            "SpO2, mean (SD)": f"{df_h['spo2'].mean():.1f} ({df_h['spo2'].std():.1f})",
            "Temperature, mean (SD)": f"{df_h['temp'].mean():.1f} ({df_h['temp'].std():.1f})",
            # Baseline Kidney
            "Admission Cr, median [IQR]": f"{df_h['admission_cr'].median():.2f} [{df_h['admission_cr'].quantile(0.25):.2f}-{df_h['admission_cr'].quantile(0.75):.2f}]",
            # Outcomes
            "AKI incidence, n (%)": f"{n_aki} ({df_h['label'].mean()*100:.1f}%)",
            # AKI Stages (among AKI patients)
            "  Stage 1, n (%)": f"{(df_aki['aki_stage'] == 1).sum()} ({(df_aki['aki_stage'] == 1).mean()*100:.1f}%)" if n_aki > 0 else "0 (0.0%)",
            "  Stage 2, n (%)": f"{(df_aki['aki_stage'] == 2).sum()} ({(df_aki['aki_stage'] == 2).mean()*100:.1f}%)" if n_aki > 0 else "0 (0.0%)",
            "  Stage 3, n (%)": f"{(df_aki['aki_stage'] == 3).sum()} ({(df_aki['aki_stage'] == 3).mean()*100:.1f}%)" if n_aki > 0 else "0 (0.0%)",
            # Urine output criteria (among AKI patients)
            "Urine output criteria, n (%)": f"{int(n_urine)} ({pct_urine:.1f}%)" if n_aki > 0 else "0 (0.0%)",
        }
        results.append(row)
    
    df_result = pd.DataFrame(results).set_index("Hospital").T
    
    # Add column headers to indicate development/external
    df_result.columns = [
        "Ilsan (Development)",
        "CCHLMC (External)",
        "MIMIC-IV (External)"
    ]
    
    return df_result


# %%
# =============================================================================
# SAVE TO SINGLE XLSX FILE
# =============================================================================

def save_to_single_xlsx(
    table1: TableOne,
    comprehensive_table: pd.DataFrame,
    savepath: str,
) -> None:
    """
    Save all baseline characteristics to a single xlsx file.
    
    Args:
        table1: TableOne object with baseline characteristics
        comprehensive_table: DataFrame with comprehensive baseline table
        savepath: Directory path to save the file
    """
    import os
    os.makedirs(savepath, exist_ok=True)
    
    filepath = f"{savepath}/table1_baseline_characteristics.xlsx"
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: TableOne output (main statistical table with p-values and SMD)
        table1_df = table1.tableone
        table1_df.to_excel(writer, sheet_name='TableOne')
        
        # Sheet 2: Comprehensive table (formatted summary)
        comprehensive_table.to_excel(writer, sheet_name='Summary')
    
    print(f"\nBaseline characteristics saved to: {filepath}")


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
    
    # Create TableOne (with AKI stage and urine output criteria)
    print("\n" + "="*60)
    print("Table 1: Baseline Characteristics (TableOne)")
    print("="*60)
    table1 = create_baseline_table(df_combined)
    display(table1)
    
    # Create comprehensive table (includes AKI stages and urine output)
    print("\n" + "="*60)
    print("Comprehensive Baseline Characteristics")
    print("="*60)
    comprehensive_table = create_comprehensive_table(df_combined)
    display(comprehensive_table)
    
    # Save all to a single xlsx file
    save_to_single_xlsx(table1, comprehensive_table, SAVE_PATH)
    
    print("\nDone!")

# %%

