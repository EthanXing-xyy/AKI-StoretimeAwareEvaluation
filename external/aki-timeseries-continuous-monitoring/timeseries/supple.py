# %%
"""
Supplementary Materials for AKI Prediction Paper
=================================================
Generates:
- Table S1: Input Features (51 features)
- Table S2: Feature Availability and Missingness
- Table S3: Site-specific External Validation Performance
- Table S4: Calibration Metrics Before and After Recalibration
- Figure S1: External Validation Discrimination (Site-specific)
- Figure S2: Calibration for All Models
- Figure S3: Online Simulation by Site (Expanded)
"""

import os
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

# Model configurations
MODEL_CONFIGS = {
    "LSTM": {
        "display_name": "LSTM-Attention",
        "base_path": f"{BASE_DIR}/result/predictions/ilsan/lstm_attention",
        "path_template": "prediction_window_{pw}/{dataset}.parquet",
    },
    "CNN": {
        "display_name": "MaskRMS-1D-CNN",
        "base_path": f"{BASE_DIR}/result/predictions/ilsan/mask_rms_cnn",
        "path_template": "prediction_window_{pw}/resolution_control/apply_prob_0.0/{dataset}.parquet",
    },
    "Transformer": {
        "display_name": "ITE-Transformer",
        "base_path": f"{BASE_DIR}/result/predictions/ilsan/ite_transformer",
        "path_template": "prediction_window_{pw}/{dataset}.parquet",
    },
}

# Dataset configurations
DATASET_CONFIGS = {
    "ilsan_test": {
        "display_name": "NHIS",
        "short_name": "NHIS",
        "type": "internal",
    },
    "cchlmc_external": {
        "display_name": "CSHH",
        "short_name": "CSHH",
        "type": "external",
    },
    "mimic-iv_external": {
        "display_name": "MIMIC-IV",
        "short_name": "MIMIC-IV",
        "type": "external",
    },
}

# Prediction horizons
PREDICTION_HORIZONS = [0, 48, 72]

# Bootstrap settings
N_BOOTSTRAP = 500
RANDOM_STATE = 42

# Color schemes
COLORS_PH = {
    0: "#2E86AB",    # Blue
    48: "#E94F37",   # Orange
    72: "#7D7D7D",   # Gray
}

COLORS_SITE = {
    "NHIS": "#1565C0",      # Blue
    "CSHH": "#E65100",      # Orange
    "MIMIC-IV": "#2E7D32",  # Green
}

# Line styles
LINE_STYLES_PH = {
    0: "-",
    48: "--",
    72: ":",
}

# Output paths
OUTPUT_TABLE_PATH = f"{BASE_DIR}/result/draft/supplementary/table"
OUTPUT_FIGURE_PATH = f"{BASE_DIR}/result/draft/supplementary/figure"


# %%
# =============================================================================
# FEATURE DEFINITIONS FOR TABLE S1
# =============================================================================

FEATURE_DEFINITIONS = {
    "Demographics": [
        ("Age", "years", "Patient age at hospital admission"),
        ("Sex", "-", "Biological sex (Male/Female)"),
        ("BMI", "kg/m²", "Body mass index"),
    ],
    "Vital Signs": [
        ("SBP", "mmHg", "Systolic blood pressure"),
        ("DBP", "mmHg", "Diastolic blood pressure"),
        ("Pulse", "/min", "Heart rate / pulse rate"),
        ("RR", "/min", "Respiratory rate"),
        ("SpO2", "%", "Peripheral oxygen saturation"),
        ("Temperature", "°C", "Body temperature"),
    ],
    "Laboratory (Kidney Function)": [
        ("Creatinine", "mg/dL", "Serum creatinine"),
        ("BUN", "mg/dL", "Blood urea nitrogen"),
        ("Uric Acid", "mg/dL", "Serum uric acid"),
    ],
    "Laboratory (Electrolytes)": [
        ("Sodium", "mEq/L", "Serum sodium"),
        ("Potassium", "mEq/L", "Serum potassium"),
        ("Chloride", "mEq/L", "Serum chloride"),
        ("Calcium", "mg/dL", "Serum calcium"),
        ("Magnesium", "mg/dL", "Serum magnesium"),
        ("Phosphorus", "mg/dL", "Serum phosphorus"),
    ],
    "Laboratory (Liver Function)": [
        ("ALT", "U/L", "Alanine aminotransferase"),
        ("AST", "U/L", "Aspartate aminotransferase"),
        ("ALP", "U/L", "Alkaline phosphatase"),
        ("GGT", "U/L", "Gamma-glutamyl transferase"),
        ("Bilirubin", "mg/dL", "Total bilirubin"),
        ("Albumin", "g/dL", "Serum albumin"),
        ("Protein", "g/dL", "Total protein"),
    ],
    "Laboratory (Hematology)": [
        ("WBC", "×10³/μL", "White blood cell count"),
        ("RBC", "×10⁶/μL", "Red blood cell count"),
        ("Hemoglobin", "g/dL", "Hemoglobin concentration"),
        ("Hematocrit", "%", "Hematocrit percentage"),
        ("Platelet", "×10³/μL", "Platelet count"),
        ("RDW", "%", "Red cell distribution width"),
        ("PDW", "fL", "Platelet distribution width"),
    ],
    "Laboratory (Coagulation)": [
        ("PT_INR", "-", "Prothrombin time (INR)"),
        ("aPTT", "seconds", "Activated partial thromboplastin time"),
    ],
    "Laboratory (Cardiac Markers)": [
        ("BNP", "pg/mL", "B-type natriuretic peptide"),
        ("CK", "U/L", "Creatine kinase"),
        ("LDH", "U/L", "Lactate dehydrogenase"),
    ],
    "Laboratory (Inflammatory Markers)": [
        ("CRP", "mg/L", "C-reactive protein"),
        ("ESR", "mm/hr", "Erythrocyte sedimentation rate"),
        ("Procalcitonin", "ng/mL", "Procalcitonin"),
    ],
    "Laboratory (Metabolic)": [
        ("Glucose", "mg/dL", "Blood glucose"),
        ("HbA1c", "%", "Glycated hemoglobin"),
        ("Cholesterol", "mg/dL", "Total cholesterol"),
        ("Triglyceride", "mg/dL", "Triglyceride"),
        ("HDL", "mg/dL", "High-density lipoprotein cholesterol"),
        ("Lactate", "mmol/L", "Blood lactate"),
    ],
    "Laboratory (Blood Gas)": [
        ("pH", "-", "Blood pH"),
        ("pCO2", "mmHg", "Partial pressure of carbon dioxide"),
        ("pO2", "mmHg", "Partial pressure of oxygen"),
        ("HCO3", "mEq/L", "Bicarbonate"),
        ("tCO2", "mEq/L", "Total carbon dioxide"),
    ],
}

# Feature code mapping (for data loading)
FEATURE_CODE_MAP = {
    "Age": "age",
    "Sex": "sex", 
    "BMI": "bmi",
    "SBP": "sbp",
    "DBP": "dbp",
    "Pulse": "pulse",
    "RR": "resp",
    "SpO2": "spo2",
    "Temperature": "temp",
    "Albumin": "albumin",
    "ALP": "alp",
    "ALT": "alt",
    "aPTT": "aptt",
    "AST": "ast",
    "Bilirubin": "bilirubin",
    "BNP": "bnp",
    "BUN": "bun",
    "Calcium": "calcium",
    "Chloride": "chloride",
    "Cholesterol": "cholesterol",
    "CK": "ck",
    "Creatinine": "creatinine",
    "CRP": "crp",
    "ESR": "esr",
    "GGT": "ggt",
    "Glucose": "glucose",
    "HbA1c": "hba1c",
    "HCO3": "hco3",
    "HDL": "hdl",
    "Hematocrit": "hematocrit",
    "Hemoglobin": "hemoglobin",
    "Lactate": "lactate",
    "LDH": "ldh",
    "Magnesium": "magnesium",
    "pCO2": "pco2",
    "PDW": "pdw",
    "pH": "ph",
    "Phosphorus": "phosphorus",
    "Platelet": "platelet",
    "pO2": "po2",
    "Potassium": "potassium",
    "Procalcitonin": "procalcitonin",
    "Protein": "protein",
    "PT_INR": "pt_inr",
    "RBC": "rbc",
    "RDW": "rdw",
    "Sodium": "sodium",
    "tCO2": "tco2",
    "Triglyceride": "triglyceride",
    "Uric Acid": "uric_acid",
    "WBC": "wbc",
}


# %%
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def decode_pred_proba(pred_proba_series: pd.Series) -> np.ndarray:
    """Decode pred_proba from bytes or float (logits) to probabilities."""
    def decode_single(x):
        if isinstance(x, bytes):
            if len(x) == 2:
                logit = struct.unpack('<e', x)[0]
                return sigmoid(logit)
            elif len(x) == 4:
                logit = struct.unpack('<f', x)[0]
                return sigmoid(logit)
            else:
                return np.nan
        elif isinstance(x, (float, np.floating, np.float32, np.float64)):
            if np.isnan(x):
                return np.nan
            return sigmoid(x)
        else:
            return np.nan
    
    decoded = pred_proba_series.apply(decode_single).values
    
    if np.any(np.isnan(decoded)):
        valid_median = np.nanmedian(decoded)
        decoded = np.where(np.isnan(decoded), valid_median, decoded)
    
    return decoded


def load_predictions(
    model_key: str,
    prediction_window: int,
    dataset: str,
) -> pd.DataFrame:
    """Load prediction data for a specific model, PW, and dataset."""
    config = MODEL_CONFIGS[model_key]
    path = f"{config['base_path']}/{config['path_template'].format(pw=prediction_window, dataset=dataset)}"
    
    df = pd.read_parquet(path, engine='fastparquet')
    df['pred_proba'] = decode_pred_proba(df['pred_proba'])
    
    return df.reset_index(drop=True)


def calculate_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def isotonic_recalibration_cv(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_splits: int = 5, 
    random_state: int = 42
) -> np.ndarray:
    """Apply Isotonic Regression recalibration using cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_prob_recal = np.zeros_like(y_prob)
    
    for train_idx, val_idx in kf.split(y_prob):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(y_prob[train_idx], y_true[train_idx])
        y_prob_recal[val_idx] = ir.transform(y_prob[val_idx])
    
    return y_prob_recal


def bootstrap_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict:
    """Calculate all metrics with bootstrap 95% CI."""
    np.random.seed(random_state)
    n = len(y_true)
    
    # Point estimates
    y_pred = (y_prob >= threshold).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        return None
    
    point_estimates = {
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
    }
    
    # Bootstrap sampling
    bootstrap_results = {k: [] for k in point_estimates.keys()}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        try:
            y_pred_boot = (y_prob_boot >= threshold).astype(int)
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_boot, y_pred_boot).ravel()
            
            bootstrap_results['auroc'].append(roc_auc_score(y_true_boot, y_prob_boot))
            bootstrap_results['auprc'].append(average_precision_score(y_true_boot, y_prob_boot))
            bootstrap_results['sensitivity'].append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)
            bootstrap_results['specificity'].append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
            bootstrap_results['ppv'].append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0)
            bootstrap_results['npv'].append(tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else 0)
        except:
            continue
    
    # Calculate confidence intervals
    alpha = 1 - ci_level
    results = {}
    
    for metric_name, point_value in point_estimates.items():
        boot_values = np.array(bootstrap_results[metric_name])
        
        if len(boot_values) > 0:
            ci_lower = np.percentile(boot_values, alpha / 2 * 100)
            ci_upper = np.percentile(boot_values, (1 - alpha / 2) * 100)
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        
        if metric_name in ['auroc', 'auprc']:
            formatted = f"{point_value:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
        else:
            formatted = f"{point_value*100:.1f} ({ci_lower*100:.1f}-{ci_upper*100:.1f})"
        
        results[metric_name] = {
            'value': point_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'formatted': formatted,
        }
    
    return results


# %%
# =============================================================================
# TABLE S1: INPUT FEATURES
# =============================================================================

def generate_table_s1(save_path: str = None) -> pd.DataFrame:
    """
    Generate Table S1: Input Features
    
    51 features organized by category with name, unit, and description.
    """
    print("\n" + "=" * 80)
    print("TABLE S1: INPUT FEATURES")
    print("=" * 80)
    
    rows = []
    feature_count = 0
    
    for category, features in FEATURE_DEFINITIONS.items():
        for feature_name, unit, description in features:
            rows.append({
                "Category": category,
                "Feature Name": feature_name,
                "Unit": unit,
                "Description": description,
            })
            feature_count += 1
    
    df = pd.DataFrame(rows)
    
    print(f"\nTotal features: {feature_count}")
    print(f"  - Demographics: {sum(1 for _, f in FEATURE_DEFINITIONS.items() if 'Demo' in _ for _ in f)}")
    
    # Count by category type
    demo_count = len(FEATURE_DEFINITIONS.get("Demographics", []))
    vital_count = len(FEATURE_DEFINITIONS.get("Vital Signs", []))
    lab_count = feature_count - demo_count - vital_count
    
    print(f"  - Demographics: {demo_count}")
    print(f"  - Vital Signs: {vital_count}")
    print(f"  - Laboratory: {lab_count}")
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_excel(f"{save_path}/table_s1_input_features.xlsx", index=False)
        df.to_csv(f"{save_path}/table_s1_input_features.csv", index=False)
        print(f"\nTable S1 saved to: {save_path}")
    
    return df


# %%
# =============================================================================
# TABLE S2: FEATURE AVAILABILITY AND MISSINGNESS
# =============================================================================

def calculate_missingness_by_site(
    hospital: str,
    data_root: str = None,
) -> Dict[str, float]:
    """
    Calculate missingness rate for each feature in a hospital.
    """
    if data_root is None:
        data_root = f"{BASE_DIR}/data"
    
    # Load master table to get visit IDs
    master_path = f"{data_root}/processed/{hospital}/transformer/prediction_window_0/master.parquet"
    
    try:
        df_master = (
            pd.read_parquet(master_path)
            .query("inclusion_yn == 1 & vital_yn == 1")
        )
    except Exception as e:
        print(f"Warning: Could not load master table for {hospital}: {e}")
        return {}
    
    visit_ids = set(df_master["visit_id"].unique())
    n_visits = len(df_master)
    
    missingness = {}
    
    # Demographics from master table
    for demo_feat in ["age", "sex", "bmi"]:
        if demo_feat in df_master.columns:
            missing_rate = df_master[demo_feat].isna().mean() * 100
        else:
            missing_rate = 100.0
        missingness[demo_feat] = missing_rate
    
    # Load vital signs
    try:
        vital_path = f"{data_root}/raw/{hospital}/vitalsign.parquet"
        df_vital = (
            pd.read_parquet(vital_path)
            .query("visit_id.isin(@visit_ids)")
        )
        
        vital_codes = ['sbp', 'dbp', 'pulse', 'resp', 'spo2', 'temp']
        for vital in vital_codes:
            has_vital = df_vital[df_vital['vital_code'] == vital]['visit_id'].nunique()
            missing_rate = (1 - has_vital / n_visits) * 100
            missingness[vital] = missing_rate
    except Exception as e:
        print(f"Warning: Could not load vitals for {hospital}: {e}")
        for vital in ['sbp', 'dbp', 'pulse', 'resp', 'spo2', 'temp']:
            missingness[vital] = np.nan
    
    # Load laboratory values
    try:
        lab_path = f"{data_root}/raw/{hospital}/laboratory.parquet"
        df_lab = (
            pd.read_parquet(lab_path)
            .query("visit_id.isin(@visit_ids)")
        )
        
        lab_codes = [
            'albumin', 'alp', 'alt', 'aptt', 'ast', 'bilirubin',
            'bnp', 'bun', 'calcium', 'chloride', 'cholesterol',
            'ck', 'creatinine', 'crp', 'esr', 'ggt', 'glucose',
            'hba1c', 'hco3', 'hdl', 'hematocrit', 'hemoglobin',
            'lactate', 'ldh', 'magnesium', 'pco2', 'pdw', 'ph',
            'phosphorus', 'platelet', 'po2', 'potassium',
            'procalcitonin', 'protein', 'pt_inr', 'rbc',
            'rdw', 'sodium', 'tco2', 'triglyceride',
            'uric_acid', 'wbc'
        ]
        
        for lab in lab_codes:
            has_lab = df_lab[df_lab['lab_code'] == lab]['visit_id'].nunique()
            missing_rate = (1 - has_lab / n_visits) * 100
            missingness[lab] = missing_rate
    except Exception as e:
        print(f"Warning: Could not load labs for {hospital}: {e}")
        for lab in lab_codes:
            missingness[lab] = np.nan
    
    return missingness


def generate_table_s2(save_path: str = None) -> pd.DataFrame:
    """
    Generate Table S2: Feature Availability and Missingness
    
    Shows missingness rates for each feature across all sites.
    """
    print("\n" + "=" * 80)
    print("TABLE S2: FEATURE AVAILABILITY AND MISSINGNESS")
    print("=" * 80)
    
    hospitals = {
        "ilsan": "NHIS",
        "cchlmc": "CSHH",
        "mimic-iv": "MIMIC-IV",
    }
    
    all_missingness = {}
    
    for hospital_code, hospital_name in hospitals.items():
        print(f"\nCalculating missingness for {hospital_name}...")
        missingness = calculate_missingness_by_site(hospital_code)
        all_missingness[hospital_name] = missingness
    
    # Build table
    rows = []
    
    for category, features in FEATURE_DEFINITIONS.items():
        for feature_name, unit, description in features:
            feature_code = FEATURE_CODE_MAP.get(feature_name, feature_name.lower())
            
            row = {
                "Category": category,
                "Feature": feature_name,
            }
            
            for hospital_name in hospitals.values():
                miss_rate = all_missingness.get(hospital_name, {}).get(feature_code, np.nan)
                if not np.isnan(miss_rate):
                    row[f"{hospital_name} Missing (%)"] = f"{miss_rate:.1f}"
                    row[f"{hospital_name} Available"] = "Yes" if miss_rate < 100 else "No"
                else:
                    row[f"{hospital_name} Missing (%)"] = "N/A"
                    row[f"{hospital_name} Available"] = "N/A"
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_excel(f"{save_path}/table_s2_feature_missingness.xlsx", index=False)
        df.to_csv(f"{save_path}/table_s2_feature_missingness.csv", index=False)
        print(f"\nTable S2 saved to: {save_path}")
    
    return df


# %%
# =============================================================================
# TABLE S3: SITE-SPECIFIC EXTERNAL VALIDATION PERFORMANCE
# =============================================================================

def generate_table_s3(save_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Table S3: Site-specific External Validation Performance
    
    Panel A: CSHH
    Panel B: MIMIC-IV
    
    Each panel shows performance for all models × all prediction horizons.
    """
    print("\n" + "=" * 80)
    print("TABLE S3: SITE-SPECIFIC EXTERNAL VALIDATION PERFORMANCE")
    print("=" * 80)
    
    external_datasets = {
        "CSHH": "cchlmc_external",
        "MIMIC-IV": "mimic-iv_external",
    }
    
    results = {}
    
    for site_name, dataset_key in external_datasets.items():
        print(f"\n--- {site_name} ---")
        results[site_name] = []
        
        for model_key in MODEL_CONFIGS.keys():
            for ph in PREDICTION_HORIZONS:
                try:
                    df = load_predictions(model_key, ph, dataset_key)
                    y_true = df['label'].values
                    y_prob = df['pred_proba'].values
                    
                    # Calculate threshold from internal validation set
                    df_internal = load_predictions(model_key, ph, "ilsan_test")
                    threshold = calculate_youden_threshold(
                        df_internal['label'].values, 
                        df_internal['pred_proba'].values
                    )
                    
                    # Calculate metrics with CI
                    metrics = bootstrap_metrics_with_ci(
                        y_true, y_prob, threshold,
                        n_bootstrap=N_BOOTSTRAP,
                        random_state=RANDOM_STATE,
                    )
                    
                    if metrics:
                        results[site_name].append({
                            "Model": MODEL_CONFIGS[model_key]["display_name"],
                            "Horizon": f"{ph}h",
                            "Site": site_name,
                            "N": len(df),
                            "Events": int(y_true.sum()),
                            "AUROC (95% CI)": metrics['auroc']['formatted'],
                            "AUPRC (95% CI)": metrics['auprc']['formatted'],
                            "Sensitivity (%)": metrics['sensitivity']['formatted'],
                            "Specificity (%)": metrics['specificity']['formatted'],
                            "PPV (%)": metrics['ppv']['formatted'],
                            "NPV (%)": metrics['npv']['formatted'],
                        })
                        
                        print(f"  {model_key} PH={ph}h: AUROC={metrics['auroc']['value']:.3f}")
                    
                except Exception as e:
                    print(f"  Error: {model_key} PH={ph}h - {e}")
    
    # Create separate DataFrames for each site
    df_cshh = pd.DataFrame(results["CSHH"])
    df_mimic = pd.DataFrame(results["MIMIC-IV"])
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        df_cshh.to_excel(f"{save_path}/table_s3a_cshh_external.xlsx", index=False)
        df_cshh.to_csv(f"{save_path}/table_s3a_cshh_external.csv", index=False)
        
        df_mimic.to_excel(f"{save_path}/table_s3b_mimiciv_external.xlsx", index=False)
        df_mimic.to_csv(f"{save_path}/table_s3b_mimiciv_external.csv", index=False)
        
        # Combined Excel with multiple sheets
        with pd.ExcelWriter(f"{save_path}/table_s3_external_validation.xlsx", engine='openpyxl') as writer:
            df_cshh.to_excel(writer, sheet_name='Panel A - CSHH', index=False)
            df_mimic.to_excel(writer, sheet_name='Panel B - MIMIC-IV', index=False)
        
        print(f"\nTable S3 saved to: {save_path}")
    
    return df_cshh, df_mimic


# %%
# =============================================================================
# TABLE S4: CALIBRATION METRICS
# =============================================================================

def generate_table_s4(save_path: str = None) -> pd.DataFrame:
    """
    Generate Table S4: Calibration Metrics Before and After Recalibration
    
    Shows Brier score before/after isotonic recalibration for all models × sites × horizons.
    """
    print("\n" + "=" * 80)
    print("TABLE S4: CALIBRATION METRICS BEFORE AND AFTER RECALIBRATION")
    print("=" * 80)
    
    datasets = {
        "NHIS": "ilsan_test",
        "CSHH": "cchlmc_external",
        "MIMIC-IV": "mimic-iv_external",
    }
    
    results = []
    
    for model_key in MODEL_CONFIGS.keys():
        for ph in PREDICTION_HORIZONS:
            print(f"\n{MODEL_CONFIGS[model_key]['display_name']} PH={ph}h:")
            
            for site_name, dataset_key in datasets.items():
                try:
                    df = load_predictions(model_key, ph, dataset_key)
                    y_true = df['label'].values
                    y_prob = df['pred_proba'].values
                    
                    # Calculate Brier score before recalibration
                    brier_before = brier_score_loss(y_true, y_prob)
                    
                    # Apply isotonic recalibration with CV
                    y_prob_recal = isotonic_recalibration_cv(y_true, y_prob)
                    
                    # Calculate Brier score after recalibration
                    brier_after = brier_score_loss(y_true, y_prob_recal)
                    
                    # Calculate improvement
                    improvement = (brier_before - brier_after) / brier_before * 100
                    
                    results.append({
                        "Model": MODEL_CONFIGS[model_key]["display_name"],
                        "Horizon": f"{ph}h",
                        "Site": site_name,
                        "Brier (Before)": f"{brier_before:.4f}",
                        "Brier (After)": f"{brier_after:.4f}",
                        "Improvement (%)": f"{improvement:.1f}%",
                    })
                    
                    print(f"  {site_name}: {brier_before:.4f} → {brier_after:.4f} ({improvement:.1f}%)")
                    
                except Exception as e:
                    print(f"  Error: {site_name} - {e}")
    
    df = pd.DataFrame(results)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_excel(f"{save_path}/table_s4_calibration_metrics.xlsx", index=False)
        df.to_csv(f"{save_path}/table_s4_calibration_metrics.csv", index=False)
        print(f"\nTable S4 saved to: {save_path}")
    
    return df


# %%
# =============================================================================
# FIGURE S1: EXTERNAL VALIDATION DISCRIMINATION (SITE-SPECIFIC)
# =============================================================================

def create_figure_s1(save_path: str = None) -> plt.Figure:
    """
    Create Figure S1: External Validation Discrimination (Site-specific)
    
    Layout: 4 rows × 3 columns = 12 panels
    - Row 1: CSHH ROC curves (LSTM, CNN, Transformer)
    - Row 2: CSHH PRC curves
    - Row 3: MIMIC-IV ROC curves
    - Row 4: MIMIC-IV PRC curves
    """
    print("\n" + "=" * 80)
    print("FIGURE S1: EXTERNAL VALIDATION DISCRIMINATION (SITE-SPECIFIC)")
    print("=" * 80)
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    
    models = list(MODEL_CONFIGS.keys())
    external_datasets = {
        "CSHH": "cchlmc_external",
        "MIMIC-IV": "mimic-iv_external",
    }
    
    panel_labels = [
        ['A', 'B', 'C'],
        ['D', 'E', 'F'],
        ['G', 'H', 'I'],
        ['J', 'K', 'L'],
    ]
    
    # Load all data
    print("Loading prediction data...")
    all_data = {}
    
    for site_name, dataset_key in external_datasets.items():
        all_data[site_name] = {}
        for model_key in models:
            all_data[site_name][model_key] = {}
            for ph in PREDICTION_HORIZONS:
                try:
                    df = load_predictions(model_key, ph, dataset_key)
                    all_data[site_name][model_key][ph] = df
                    print(f"  ✓ {site_name} {model_key} PH={ph}h: n={len(df)}")
                except Exception as e:
                    print(f"  ✗ {site_name} {model_key} PH={ph}h: {e}")
    
    # Plot
    row_idx = 0
    for site_idx, site_name in enumerate(external_datasets.keys()):
        # ROC curves
        for col_idx, model_key in enumerate(models):
            ax = axes[row_idx, col_idx]
            
            for ph in PREDICTION_HORIZONS:
                if ph in all_data[site_name][model_key]:
                    df = all_data[site_name][model_key][ph]
                    y_true = df['label'].values
                    y_prob = df['pred_proba'].values
                    
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    auroc = roc_auc_score(y_true, y_prob)
                    
                    ax.plot(fpr, tpr, color=COLORS_PH[ph], linestyle=LINE_STYLES_PH[ph],
                            linewidth=2, label=f"PH {ph}h ({auroc:.3f})")
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.05])
            ax.set_xlabel('1 - Specificity')
            ax.set_ylabel('Sensitivity')
            ax.set_title(f"{MODEL_CONFIGS[model_key]['display_name']}")
            ax.legend(loc='lower right', fontsize=7, title='AUROC')
            ax.set_aspect('equal')
            ax.text(-0.15, 1.05, panel_labels[row_idx][col_idx], 
                    transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
        
        row_idx += 1
        
        # PRC curves
        for col_idx, model_key in enumerate(models):
            ax = axes[row_idx, col_idx]
            
            for ph in PREDICTION_HORIZONS:
                if ph in all_data[site_name][model_key]:
                    df = all_data[site_name][model_key][ph]
                    y_true = df['label'].values
                    y_prob = df['pred_proba'].values
                    
                    precision, recall, _ = precision_recall_curve(y_true, y_prob)
                    auprc = average_precision_score(y_true, y_prob)
                    
                    ax.plot(recall, precision, color=COLORS_PH[ph], linestyle=LINE_STYLES_PH[ph],
                            linewidth=2, label=f"PH {ph}h ({auprc:.3f})")
            
            # Baseline (prevalence)
            baseline = all_data[site_name][model_key][0]['label'].mean()
            ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.4)
            
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f"{MODEL_CONFIGS[model_key]['display_name']}")
            ax.legend(loc='lower left', fontsize=7, title='AUPRC')
            ax.text(-0.15, 1.05, panel_labels[row_idx][col_idx], 
                    transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
        
        row_idx += 1
    
    # Add row labels
    fig.text(0.01, 0.88, 'CSHH\nROC', fontsize=10, fontweight='bold', rotation=90, va='center', ha='center')
    fig.text(0.01, 0.63, 'CSHH\nPRC', fontsize=10, fontweight='bold', rotation=90, va='center', ha='center')
    fig.text(0.01, 0.38, 'MIMIC-IV\nROC', fontsize=10, fontweight='bold', rotation=90, va='center', ha='center')
    fig.text(0.01, 0.13, 'MIMIC-IV\nPRC', fontsize=10, fontweight='bold', rotation=90, va='center', ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, top=0.94, hspace=0.35, wspace=0.30)
    
    # Title removed for publication
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        for fmt in ['png', 'pdf', 'svg', 'tiff']:
            fig.savefig(f"{save_path}/figure_s1_external_discrimination.{fmt}",
                        dpi=500, bbox_inches='tight', facecolor='white', format=fmt)
        print(f"\nFigure S1 saved to: {save_path} (png, pdf, svg, tiff)")
    
    return fig


# %%
# =============================================================================
# FIGURE S2: CALIBRATION FOR ALL MODELS
# =============================================================================

def create_figure_s2(save_path: str = None) -> plt.Figure:
    """
    Create Figure S2: Calibration and Recalibration for All Models
    
    Layout: 3 rows × 3 columns = 9 panels
    - Rows: LSTM, CNN, Transformer
    - Columns: PH 0h, PH 48h, PH 72h
    - Each panel: Before/After calibration curves for 3 sites
    """
    print("\n" + "=" * 80)
    print("FIGURE S2: CALIBRATION AND RECALIBRATION FOR ALL MODELS")
    print("=" * 80)
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 6,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    
    models = list(MODEL_CONFIGS.keys())
    datasets = {
        "NHIS": "ilsan_test",
        "CSHH": "cchlmc_external",
        "MIMIC-IV": "mimic-iv_external",
    }
    
    panel_labels = [
        ['A', 'B', 'C'],
        ['D', 'E', 'F'],
        ['G', 'H', 'I'],
    ]
    
    line_styles = {
        "before": "--",
        "after": "-",
    }
    
    print("Loading and processing data...")
    
    for row_idx, model_key in enumerate(models):
        for col_idx, ph in enumerate(PREDICTION_HORIZONS):
            ax = axes[row_idx, col_idx]
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, 
                    alpha=0.6, label='Perfect', zorder=1)
            
            for site_name, dataset_key in datasets.items():
                try:
                    df = load_predictions(model_key, ph, dataset_key)
                    y_true = df['label'].values
                    y_prob = df['pred_proba'].values
                    
                    # Calculate calibration curve before recalibration
                    prob_true_before, prob_pred_before = calibration_curve(
                        y_true, y_prob, n_bins=10, strategy='quantile'
                    )
                    brier_before = brier_score_loss(y_true, y_prob)
                    
                    # Apply recalibration
                    y_prob_recal = isotonic_recalibration_cv(y_true, y_prob)
                    
                    # Calculate calibration curve after recalibration
                    prob_true_after, prob_pred_after = calibration_curve(
                        y_true, y_prob_recal, n_bins=10, strategy='quantile'
                    )
                    brier_after = brier_score_loss(y_true, y_prob_recal)
                    
                    # Plot before (dashed)
                    ax.plot(prob_pred_before, prob_true_before, 
                            color=COLORS_SITE[site_name], linestyle='--', linewidth=1.5,
                            alpha=0.7, marker='o', markersize=4,
                            label=f"{site_name} Before ({brier_before:.3f})")
                    
                    # Plot after (solid)
                    ax.plot(prob_pred_after, prob_true_after,
                            color=COLORS_SITE[site_name], linestyle='-', linewidth=2,
                            alpha=1.0, marker='s', markersize=4,
                            label=f"{site_name} After ({brier_after:.3f})")
                    
                    print(f"  {model_key} PH={ph}h {site_name}: {brier_before:.4f} → {brier_after:.4f}")
                    
                except Exception as e:
                    print(f"  Error: {model_key} PH={ph}h {site_name} - {e}")
            
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.05])
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Observed Frequency')
            ax.set_title(f"PH {ph}h")
            ax.legend(loc='upper left', fontsize=5.5, ncol=1,
                     title='Site (Brier)', title_fontsize=6)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            ax.text(-0.15, 1.05, panel_labels[row_idx][col_idx], 
                    transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Add row labels
    for row_idx, model_key in enumerate(models):
        y_pos = 0.83 - row_idx * 0.31
        fig.text(0.01, y_pos, MODEL_CONFIGS[model_key]['display_name'], 
                 fontsize=11, fontweight='bold', rotation=90, va='center', ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.10, top=0.94, hspace=0.35, wspace=0.30)
    
    # Title removed for publication
    
    # Add legend for line styles
    custom_lines = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Before Recalibration'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='After Recalibration'),
    ]
    fig.legend(handles=custom_lines, loc='lower center', ncol=2, fontsize=10, 
               bbox_to_anchor=(0.5, 0.01), framealpha=0.95)
    
    plt.subplots_adjust(bottom=0.06)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        for fmt in ['png', 'pdf', 'svg', 'tiff']:
            fig.savefig(f"{save_path}/figure_s2_calibration_all_models.{fmt}",
                        dpi=500, bbox_inches='tight', facecolor='white', format=fmt)
        print(f"\nFigure S2 saved to: {save_path} (png, pdf, svg, tiff)")
    
    return fig


# %%
# =============================================================================
# FIGURE S3: ONLINE SIMULATION BY SITE (EXPANDED)
# =============================================================================

def load_online_performance_csv(dataset_key: str) -> pd.DataFrame:
    """Load online performance CSV for a dataset."""
    csv_files = {
        "ilsan_test": f"{BASE_DIR}/result/tables/online_ilsan_test_performance.csv",
        "cchlmc_external": f"{BASE_DIR}/result/tables/online_cchlmc_external_performance.csv",
        "mimic-iv_external": f"{BASE_DIR}/result/tables/online_mimic_iv_external_performance.csv",
    }
    
    path = csv_files.get(dataset_key)
    if path and os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


def create_figure_s3(save_path: str = None) -> plt.Figure:
    """
    Create Figure S3: Online Simulation by Site (Expanded)
    
    Layout: 1 row × 3 columns
    - Panel A: NHIS (Internal)
    - Panel B: CSHH (External)
    - Panel C: MIMIC-IV (External)
    
    Each panel: 9 lines (3 models × 3 prediction horizons) with 95% CI
    """
    print("\n" + "=" * 80)
    print("FIGURE S3: ONLINE SIMULATION BY SITE (EXPANDED)")
    print("=" * 80)
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    datasets = {
        "NHIS (Internal)": "ilsan_test",
        "CSHH (External)": "cchlmc_external",
        "MIMIC-IV (External)": "mimic-iv_external",
    }
    
    model_prefixes = {
        "ilsan_test": {
            "Transformer": "ITE Transformer - NHIS (Internal)",
            "LSTM": "LSTM-Attention - NHIS (Internal)",
            "CNN": "Masked CNN - NHIS (Internal)",
        },
        "cchlmc_external": {
            "Transformer": "ITE Transformer - CSHH (External)",
            "LSTM": "LSTM-Attention - CSHH (External)",
            "CNN": "Masked CNN - CSHH (External)",
        },
        "mimic-iv_external": {
            "Transformer": "ITE Transformer - MIMIC-IV (External)",
            "LSTM": "LSTM-Attention - MIMIC-IV (External)",
            "CNN": "Masked CNN - MIMIC-IV (External)",
        },
    }
    
    time_points = [0, 12, 24, 36, 48, 60, 72]
    
    line_styles_model = {
        "Transformer": "-",
        "LSTM": "--",
        "CNN": ":",
    }
    
    line_widths_model = {
        "Transformer": 2.5,
        "LSTM": 2.0,
        "CNN": 2.0,
    }
    
    markers_model = {
        "Transformer": "^",
        "LSTM": "o",
        "CNN": "s",
    }
    
    panel_labels = ['A', 'B', 'C']
    
    print("Loading online performance data...")
    
    for col_idx, (site_display, dataset_key) in enumerate(datasets.items()):
        ax = axes[col_idx]
        
        df_online = load_online_performance_csv(dataset_key)
        
        if df_online is not None:
            df_auroc = df_online[df_online['Metric'] == 'AUROC']
            prefixes = model_prefixes[dataset_key]
            
            # Plot each model × PH combination
            for ph in [72, 48, 0]:  # Reverse order so PH 0h is on top
                for model_key in ["Transformer", "LSTM", "CNN"]:
                    model_prefix = prefixes[model_key]
                    model_label = f"{model_prefix} (PW={ph}h)"
                    
                    row = df_auroc[df_auroc['Model'] == model_label]
                    
                    if len(row) > 0:
                        auroc_values = []
                        for tp in time_points:
                            col_name = f"{tp}h"
                            if col_name in row.columns:
                                auroc_values.append(float(row[col_name].values[0]))
                            else:
                                auroc_values.append(np.nan)
                        
                        # Line width and alpha based on PH
                        lw = line_widths_model[model_key] * (1.3 if ph == 0 else 1.0 if ph == 48 else 0.9)
                        alpha = 1.0 if ph == 0 else 0.6 if ph == 48 else 0.5
                        zorder = 10 if ph == 0 else 5
                        
                        ax.plot(time_points, auroc_values,
                                color=COLORS_PH[ph],
                                linestyle=line_styles_model[model_key],
                                linewidth=lw,
                                alpha=alpha,
                                zorder=zorder)
                        
                        # Add markers at key time points for PH 0h
                        if ph == 0:
                            key_indices = [0, 2, 4, 6]  # 0h, 24h, 48h, 72h
                            for idx in key_indices:
                                if idx < len(auroc_values) and not np.isnan(auroc_values[idx]):
                                    ax.scatter(time_points[idx], auroc_values[idx],
                                              color=COLORS_PH[ph], marker=markers_model[model_key],
                                              s=35, zorder=15, alpha=1.0, 
                                              edgecolors='white', linewidth=0.8)
        
        # Styling
        ax.set_xlim([74, -2])  # Reversed
        ax.set_ylim([0.5, 1.0])
        ax.set_xlabel('Hours before AKI onset')
        ax.set_ylabel('AUROC')
        ax.set_title(site_display)
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.set_xticks(time_points)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.text(-0.10, 1.05, panel_labels[col_idx], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')
    
    # Create custom legend
    model_legend = [
        Line2D([0], [0], color='#444444', linestyle=line_styles_model[m], 
               linewidth=line_widths_model[m], label=MODEL_CONFIGS[m]['display_name'])
        for m in ["Transformer", "LSTM", "CNN"]
    ]
    
    ph_legend = [
        Line2D([0], [0], color=COLORS_PH[0], linestyle='-', linewidth=3, label='PH 0h'),
        Line2D([0], [0], color=COLORS_PH[48], linestyle='-', linewidth=2, alpha=0.6, label='PH 48h'),
        Line2D([0], [0], color=COLORS_PH[72], linestyle='-', linewidth=2, alpha=0.5, label='PH 72h'),
    ]
    
    fig.legend(handles=model_legend, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.30, -0.02), title='Model Architecture', title_fontsize=9)
    fig.legend(handles=ph_legend, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.70, -0.02), title='Prediction Horizon', title_fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.90, wspace=0.25)
    
    # Title removed for publication
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        for fmt in ['png', 'pdf', 'svg', 'tiff']:
            fig.savefig(f"{save_path}/figure_s3_online_simulation_expanded.{fmt}",
                        dpi=500, bbox_inches='tight', facecolor='white', format=fmt)
        print(f"\nFigure S3 saved to: {save_path} (png, pdf, svg, tiff)")
    
    return fig


# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GENERATING SUPPLEMENTARY MATERIALS")
    print("=" * 80)
    
    # Create output directories
    os.makedirs(OUTPUT_TABLE_PATH, exist_ok=True)
    os.makedirs(OUTPUT_FIGURE_PATH, exist_ok=True)
    
    # Generate Tables
    print("\n" + "=" * 80)
    print("GENERATING SUPPLEMENTARY TABLES")
    print("=" * 80)
    
    # Table S1: Input Features
    df_s1 = generate_table_s1(save_path=OUTPUT_TABLE_PATH)
    display(df_s1.head(10))
    
    # Table S2: Feature Availability and Missingness
    df_s2 = generate_table_s2(save_path=OUTPUT_TABLE_PATH)
    display(df_s2.head(10))
    
    # Table S3: Site-specific External Validation
    df_s3_cshh, df_s3_mimic = generate_table_s3(save_path=OUTPUT_TABLE_PATH)
    print("\n--- CSHH ---")
    display(df_s3_cshh)
    print("\n--- MIMIC-IV ---")
    display(df_s3_mimic)
    
    # Table S4: Calibration Metrics
    df_s4 = generate_table_s4(save_path=OUTPUT_TABLE_PATH)
    display(df_s4)
    
    # Generate Figures
    print("\n" + "=" * 80)
    print("GENERATING SUPPLEMENTARY FIGURES")
    print("=" * 80)
    
    # Figure S1: External Validation Discrimination
    fig_s1 = create_figure_s1(save_path=OUTPUT_FIGURE_PATH)
    plt.show()
    
    # Figure S2: Calibration for All Models
    fig_s2 = create_figure_s2(save_path=OUTPUT_FIGURE_PATH)
    plt.show()
    
    # Figure S3: Online Simulation Expanded
    fig_s3 = create_figure_s3(save_path=OUTPUT_FIGURE_PATH)
    plt.show()
    
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY MATERIALS GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nTables saved to: {OUTPUT_TABLE_PATH}")
    print(f"Figures saved to: {OUTPUT_FIGURE_PATH}")
    print("=" * 80)

# %%

