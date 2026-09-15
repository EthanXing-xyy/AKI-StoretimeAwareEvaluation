# %%
"""
Model Performance Tables for AKI Prediction Paper
==================================================
Generates:
- Table 2: Model Performance in Development Setting (Internal Validation)
- Table 3: External Validation Performance (CCHLMC & MIMIC-IV)

All metrics calculated at Youden's index optimal threshold.
95% CI calculated using bootstrap resampling.
"""

import os
import struct
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)
from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configurations - using admission-level prediction files
MODEL_CONFIGS = {
    "LSTM": {
        "base_path": "../../../result/predictions/ilsan/lstm_attention",
        "path_template": "prediction_window_{pw}/{dataset}.parquet",
    },
    "CNN": {
        "base_path": "../../../result/predictions/ilsan/mask_rms_cnn",
        "path_template": "prediction_window_{pw}/resolution_control/apply_prob_0.0/{dataset}.parquet",
    },
    "Transformer": {
        "base_path": "../../../result/predictions/ilsan/ite_transformer",
        "path_template": "prediction_window_{pw}/{dataset}.parquet",
    },
}

# Dataset configurations
DATASET_CONFIGS = {
    "ilsan_test": {
        "display_name": "Internal (NHIS)",
        "type": "internal",
    },
    "cchlmc_external": {
        "display_name": "CSHH",
        "type": "external",
    },
    "mimic-iv_external": {
        "display_name": "MIMIC-IV",
        "type": "external",
    },
}

# Prediction horizons
PREDICTION_HORIZONS = [0, 48, 72]

# Bootstrap settings
N_BOOTSTRAP = 500  # Can increase to 1000 for final tables
RANDOM_STATE = 42

# Output path
OUTPUT_PATH = "../../../result/draft/table"


# %%
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def decode_pred_proba(pred_proba_series: pd.Series) -> np.ndarray:
    """
    Decode pred_proba from bytes or float (logits) to probabilities.
    
    The pred_proba column stores logit values that need sigmoid transformation.
    Values may be stored as:
    - 2-byte float16 (bytes)
    - 4-byte float32 (bytes or numpy.float32)
    
    All values are logits and require sigmoid to convert to probabilities.
    """
    def decode_single(x):
        if isinstance(x, bytes):
            if len(x) == 2:
                logit = struct.unpack('<e', x)[0]  # little-endian float16
                return sigmoid(logit)
            elif len(x) == 4:
                logit = struct.unpack('<f', x)[0]  # little-endian float32
                return sigmoid(logit)
            else:
                # Invalid byte length, return NaN
                return np.nan
        elif isinstance(x, (float, np.floating, np.float32, np.float64)):
            if np.isnan(x):
                return np.nan
            # All float values are logits - apply sigmoid
            return sigmoid(x)
        else:
            return np.nan
    
    decoded = pred_proba_series.apply(decode_single).values
    
    # Handle any remaining NaN by using median imputation
    if np.any(np.isnan(decoded)):
        valid_median = np.nanmedian(decoded)
        decoded = np.where(np.isnan(decoded), valid_median, decoded)
    
    return decoded


def calculate_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate optimal threshold using Youden's J statistic.
    
    Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict:
    """Compute all classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        return {
            'auroc': np.nan,
            'auprc': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan,
            'ppv': np.nan,
            'npv': np.nan,
        }
    
    return {
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
    }


def bootstrap_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict:
    """
    Calculate all metrics with bootstrap 95% CI.
    
    Returns dict with each metric containing:
    - value: point estimate
    - ci_lower: lower bound of CI
    - ci_upper: upper bound of CI
    - formatted: string formatted for publication
    """
    np.random.seed(random_state)
    n = len(y_true)
    
    # Point estimates
    point_estimates = compute_metrics_at_threshold(y_true, y_prob, threshold)
    
    # Bootstrap sampling
    bootstrap_results = {k: [] for k in point_estimates.keys()}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        try:
            boot_metrics = compute_metrics_at_threshold(y_true_boot, y_prob_boot, threshold)
            for k, v in boot_metrics.items():
                if not np.isnan(v):
                    bootstrap_results[k].append(v)
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
        
        # Format based on metric type
        if metric_name in ['auroc', 'auprc']:
            formatted = f"{point_value:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
        else:
            # Sensitivity, Specificity, PPV, NPV as percentages
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
# DATA LOADING
# =============================================================================

def load_predictions(
    model_name: str,
    config: Dict,
    prediction_window: int,
    dataset: str,
) -> pd.DataFrame:
    """
    Load prediction data for a specific model, PW, and dataset.
    
    Uses admission-level parquet files (ilsan_test.parquet, etc.)
    Decodes pred_proba from bytes/float16 logits to probabilities.
    
    Args:
        model_name: Model identifier
        config: Model configuration dict
        prediction_window: Prediction horizon (0, 48, 72)
        dataset: Dataset name (ilsan_test, cchlmc_external, mimic-iv_external)
        
    Returns:
        DataFrame with predictions (one row per admission)
    """
    path = f"{config['base_path']}/{config['path_template'].format(pw=prediction_window, dataset=dataset)}"
    
    # Use fastparquet engine for compatibility with bytes encoding
    df = pd.read_parquet(path, engine='fastparquet')
    
    # Decode pred_proba from bytes/logits to probabilities
    df['pred_proba'] = decode_pred_proba(df['pred_proba'])
    
    return df.reset_index(drop=True)


def load_all_predictions() -> Dict:
    """
    Load all predictions for all models, PHs, and datasets.
    
    Returns:
        Nested dict: {model: {dataset: {ph: DataFrame}}}
    """
    all_data = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        all_data[model_name] = {}
        
        for dataset_name in DATASET_CONFIGS.keys():
            all_data[model_name][dataset_name] = {}
            
            for ph in PREDICTION_HORIZONS:
                try:
                    df = load_predictions(model_name, config, ph, dataset_name)
                    all_data[model_name][dataset_name][ph] = df
                    
                    # Quick validation
                    auroc = roc_auc_score(df['label'], df['pred_proba'])
                    print(f"✓ {model_name} - {dataset_name} - PH={ph}h: {len(df):,} admissions, AUROC={auroc:.3f}")
                except Exception as e:
                    print(f"✗ Failed: {model_name} - {dataset_name} - PH={ph}h: {e}")
    
    return all_data


# %%
# =============================================================================
# TABLE GENERATION FUNCTIONS
# =============================================================================

def create_model_performance_table(
    all_data: Dict,
    dataset: str,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Create model performance table for a specific dataset.
    
    Structure:
    - Rows: AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV, Threshold
    - Columns: Model × PH (9 columns: LSTM 0h, LSTM 48h, ..., Transformer 72h)
    
    All metrics at Youden index cutoff.
    """
    # Define row order and display names
    metric_rows = [
        ('auroc', 'AUROC (95% CI)'),
        ('auprc', 'AUPRC (95% CI)'),
        ('sensitivity', 'Sensitivity (%)'),
        ('specificity', 'Specificity (%)'),
        ('ppv', 'PPV (%)'),
        ('npv', 'NPV (%)'),
        ('threshold', 'Threshold'),
    ]
    
    # Define column order
    models = ['LSTM', 'CNN', 'Transformer']
    prediction_horizons = [0, 48, 72]
    
    # Build column headers
    columns = ['Metric']
    for model in models:
        for ph in prediction_horizons:
            columns.append(f"{model}\nPH {ph}h")
    
    # Calculate metrics for each model-PH combination
    results_dict = {}  # {(model, ph): metrics_dict}
    thresholds_dict = {}  # {(model, ph): threshold}
    
    for model in models:
        for ph in prediction_horizons:
            try:
                df = all_data[model][dataset][ph]
                y_true = df['label'].values
                y_prob = df['pred_proba'].values
                
                # Calculate optimal threshold using Youden's index
                threshold = calculate_youden_threshold(y_true, y_prob)
                thresholds_dict[(model, ph)] = threshold
                
                # Calculate all metrics with CI
                metrics = bootstrap_metrics_with_ci(
                    y_true, y_prob, threshold,
                    n_bootstrap=n_bootstrap,
                    random_state=RANDOM_STATE,
                )
                results_dict[(model, ph)] = metrics
                
                print(f"  {model} PH={ph}h: threshold={threshold:.3f}, AUROC={metrics['auroc']['value']:.3f}")
                
            except Exception as e:
                print(f"  Error for {model} PH={ph}h: {e}")
                results_dict[(model, ph)] = None
                thresholds_dict[(model, ph)] = np.nan
    
    # Build table rows
    rows = []
    for metric_key, metric_display in metric_rows:
        row = {'Metric': metric_display}
        
        for model in models:
            for ph in prediction_horizons:
                col_name = f"{model}\nPH {ph}h"
                
                if metric_key == 'threshold':
                    # Threshold row
                    threshold = thresholds_dict.get((model, ph), np.nan)
                    row[col_name] = f"{threshold:.3f}" if not np.isnan(threshold) else "N/A"
                else:
                    # Metric rows
                    metrics = results_dict.get((model, ph))
                    if metrics and metric_key in metrics:
                        row[col_name] = metrics[metric_key]['formatted']
                    else:
                        row[col_name] = "N/A"
        
        rows.append(row)
    
    df_table = pd.DataFrame(rows)
    
    return df_table


def create_compact_performance_table(
    all_data: Dict,
    dataset: str,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Create compact table with PH as rows and Model as columns.
    
    Alternative structure that may be more compact:
    - Rows: PH 0h, PH 48h, PH 72h (each with sub-rows for metrics)
    - Columns: LSTM, CNN, Transformer
    """
    models = ['LSTM', 'CNN', 'Transformer']
    prediction_horizons = [0, 48, 72]
    
    metrics_list = [
        ('auroc', 'AUROC (95% CI)'),
        ('auprc', 'AUPRC (95% CI)'),
        ('sensitivity', 'Sensitivity (%)'),
        ('specificity', 'Specificity (%)'),
        ('ppv', 'PPV (%)'),
        ('npv', 'NPV (%)'),
        ('threshold', 'Threshold'),
    ]
    
    # Calculate all metrics
    results_dict = {}
    thresholds_dict = {}
    
    for model in models:
        for ph in prediction_horizons:
            try:
                df = all_data[model][dataset][ph]
                y_true = df['label'].values
                y_prob = df['pred_proba'].values
                
                threshold = calculate_youden_threshold(y_true, y_prob)
                thresholds_dict[(model, ph)] = threshold
                
                metrics = bootstrap_metrics_with_ci(
                    y_true, y_prob, threshold,
                    n_bootstrap=n_bootstrap,
                    random_state=RANDOM_STATE,
                )
                results_dict[(model, ph)] = metrics
                
            except Exception as e:
                results_dict[(model, ph)] = None
                thresholds_dict[(model, ph)] = np.nan
    
    # Build table
    rows = []
    for ph in prediction_horizons:
        for metric_key, metric_display in metrics_list:
            row = {
                'Prediction Horizon': f'{ph}h' if metric_key == 'auroc' else '',
                'Metric': metric_display,
            }
            
            for model in models:
                if metric_key == 'threshold':
                    threshold = thresholds_dict.get((model, ph), np.nan)
                    row[model] = f"{threshold:.3f}" if not np.isnan(threshold) else "N/A"
                else:
                    metrics = results_dict.get((model, ph))
                    if metrics and metric_key in metrics:
                        row[model] = metrics[metric_key]['formatted']
                    else:
                        row[model] = "N/A"
            
            rows.append(row)
    
    return pd.DataFrame(rows)


# %%
# =============================================================================
# TABLE 2: INTERNAL VALIDATION
# =============================================================================

def generate_table2(all_data: Dict, save_path: str = None) -> pd.DataFrame:
    """
    Generate Table 2: Model Performance in Development Setting (Internal Validation)
    
    Uses Ilsan test set data.
    """
    print("\n" + "=" * 80)
    print("TABLE 2: Model Performance in Development Setting (Internal Validation)")
    print("=" * 80)
    
    df_table2 = create_model_performance_table(
        all_data=all_data,
        dataset="ilsan_test",
        n_bootstrap=N_BOOTSTRAP,
    )
    
    # Add footer note
    print("\nNote: All metrics calculated at Youden's index optimal threshold")
    print(f"95% CI calculated using bootstrap resampling (n={N_BOOTSTRAP})")
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save as Excel
        df_table2.to_excel(f"{save_path}/table2_internal_validation.xlsx", index=False)
        
        # Save as CSV
        df_table2.to_csv(f"{save_path}/table2_internal_validation.csv", index=False)
        
        print(f"\nTable 2 saved to: {save_path}")
    
    return df_table2


# %%
# =============================================================================
# TABLE 3: EXTERNAL VALIDATION
# =============================================================================

def generate_table3(all_data: Dict, save_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Table 3: External Validation Performance
    
    Panel A: CCHLMC
    Panel B: MIMIC-IV
    """
    print("\n" + "=" * 80)
    print("TABLE 3: External Validation Performance")
    print("=" * 80)
    
    # Panel A: CCHLMC
    print("\n--- Panel A: CCHLMC ---")
    df_table3a = create_model_performance_table(
        all_data=all_data,
        dataset="cchlmc_external",
        n_bootstrap=N_BOOTSTRAP,
    )
    
    # Panel B: MIMIC-IV
    print("\n--- Panel B: MIMIC-IV ---")
    df_table3b = create_model_performance_table(
        all_data=all_data,
        dataset="mimic-iv_external",
        n_bootstrap=N_BOOTSTRAP,
    )
    
    print("\nNote: All metrics calculated at Youden's index optimal threshold")
    print(f"95% CI calculated using bootstrap resampling (n={N_BOOTSTRAP})")
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save Panel A
        df_table3a.to_excel(f"{save_path}/table3a_cchlmc_external.xlsx", index=False)
        df_table3a.to_csv(f"{save_path}/table3a_cchlmc_external.csv", index=False)
        
        # Save Panel B
        df_table3b.to_excel(f"{save_path}/table3b_mimiciv_external.xlsx", index=False)
        df_table3b.to_csv(f"{save_path}/table3b_mimiciv_external.csv", index=False)
        
        # Save combined table with both panels
        with pd.ExcelWriter(f"{save_path}/table3_external_validation.xlsx", engine='openpyxl') as writer:
            df_table3a.to_excel(writer, sheet_name='Panel A - CCHLMC', index=False)
            df_table3b.to_excel(writer, sheet_name='Panel B - MIMIC-IV', index=False)
        
        print(f"\nTable 3 saved to: {save_path}")
    
    return df_table3a, df_table3b


# %%
# =============================================================================
# ADDITIONAL: SAMPLE SIZE SUMMARY
# =============================================================================

def generate_sample_summary(all_data: Dict) -> pd.DataFrame:
    """Generate sample size and event rate summary for each dataset."""
    print("\n" + "=" * 80)
    print("SAMPLE SIZE SUMMARY")
    print("=" * 80)
    
    rows = []
    
    for dataset, config in DATASET_CONFIGS.items():
        # Use first model (LSTM) and PH=0 for sample size (should be same across models)
        try:
            df = all_data['LSTM'][dataset][0]
            n_total = len(df)
            n_events = int(df['label'].sum())
            event_rate = df['label'].mean() * 100
            
            rows.append({
                'Dataset': config['display_name'],
                'Type': config['type'].capitalize(),
                'N (admissions)': f"{n_total:,}",
                'AKI events': f"{n_events:,}",
                'Event rate (%)': f"{event_rate:.1f}",
            })
        except Exception as e:
            print(f"Error for {dataset}: {e}")
    
    df_summary = pd.DataFrame(rows)
    display(df_summary)
    
    return df_summary


# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load all data
    print("Loading prediction data...")
    print("Using admission-level parquet files:")
    print("  - ilsan_test.parquet")
    print("  - cchlmc_external.parquet") 
    print("  - mimic-iv_external.parquet")
    print()
    
    all_data = load_all_predictions()
    
    # Generate sample summary
    df_sample_summary = generate_sample_summary(all_data)
    
    # Generate Table 2: Internal Validation
    df_table2 = generate_table2(all_data, save_path=OUTPUT_PATH)
    print("\n--- Table 2: Internal Validation ---")
    display(df_table2)
    
    # Generate Table 3: External Validation
    df_table3a, df_table3b = generate_table3(all_data, save_path=OUTPUT_PATH)
    print("\n--- Table 3A: CCHLMC ---")
    display(df_table3a)
    print("\n--- Table 3B: MIMIC-IV ---")
    display(df_table3b)
    
    print("\n" + "=" * 80)
    print("DONE! Tables saved to:", OUTPUT_PATH)
    print("=" * 80)

# %%
