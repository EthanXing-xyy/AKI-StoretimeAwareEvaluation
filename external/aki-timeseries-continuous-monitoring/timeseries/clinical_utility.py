# %%
"""
Figure 3: Clinical Utility of the Transformer Model (PH 0h)
============================================================
Generates a 2 rows × 2 columns figure:
- Panel A: Calibration BEFORE Recalibration (3 sites overlaid)
- Panel B: Calibration AFTER Isotonic Recalibration (3 sites overlaid)
- Panel C: Lead Time Distribution (Box plot by site) - spans full width

Data sources:
- Calibration: Single prediction data (ilsan_test, cchlmc_external, mimic-iv_external)
- Lead Time: Online simulation data (first alert → AKI, excluding post-onset predictions)

Output CSVs:
- figure3ab_calibration.csv: Calibration data before/after recalibration
- figure3c_leadtime.csv: Lead time statistics by site
"""

import os
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.model_selection import KFold
from typing import Tuple, Dict, List

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory (resolve relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

# Model configuration
PREDICTION_WINDOW = 0
BASE_PATH = f"{BASE_DIR}/result/predictions/ilsan/ite_transformer/prediction_window_{PREDICTION_WINDOW}"

# Site configurations
SITE_CONFIGS = {
    "NHIS": {
        "single_file": "ilsan_test.parquet",
        "online_file": "ilsan_test_online.parquet",
        "type": "Development",
        "color": "#1565C0",      # Blue
        "marker": "o",
        "label": "NHIS (Development)"
    },
    "CSHH": {
        "single_file": "cchlmc_external.parquet",
        "online_file": "cchlmc_external_online.parquet",
        "type": "External",
        "color": "#E65100",      # Orange
        "marker": "^",
        "label": "CSHH (External)"
    },
    "MIMIC-IV": {
        "single_file": "mimic-iv_external.parquet",
        "online_file": "mimic-iv_external_online.parquet",
        "type": "External",
        "color": "#2E7D32",      # Green
        "marker": "s",
        "label": "MIMIC-IV (External)"
    },
}

# Color scheme
COLORS = {
    "perfect": "#666666",       # Gray for perfect calibration line
}

# Output paths
OUTPUT_PATH = f"{BASE_DIR}/result/draft/figure"
TABLE_PATH = f"{BASE_DIR}/result/draft/table"


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


def load_single_predictions(site_name: str) -> pd.DataFrame:
    """
    Load single prediction data for calibration.
    """
    config = SITE_CONFIGS[site_name]
    path = f"{BASE_PATH}/{config['single_file']}"
    
    df = pd.read_parquet(path, engine='fastparquet')
    df['pred_proba'] = decode_pred_proba(df['pred_proba'])
    
    return df.reset_index(drop=True)


def load_online_predictions(site_name: str) -> pd.DataFrame:
    """
    Load online prediction data for lead time analysis.
    """
    config = SITE_CONFIGS[site_name]
    path = f"{BASE_PATH}/online/{config['online_file']}"
    
    df = pd.read_parquet(path, engine='fastparquet')
    
    return df.reset_index(drop=True)


# %%
# =============================================================================
# CALIBRATION FUNCTIONS
# =============================================================================

def stratified_bootstrap_indices(y_true: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
    """
    Generate stratified bootstrap indices that maintain class proportions.
    """
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    
    boot_idx_pos = random_state.choice(idx_pos, size=len(idx_pos), replace=True)
    boot_idx_neg = random_state.choice(idx_neg, size=len(idx_neg), replace=True)
    
    indices = np.concatenate([boot_idx_pos, boot_idx_neg])
    random_state.shuffle(indices)
    
    return indices


def isotonic_recalibration_cv(y_true: np.ndarray, y_prob: np.ndarray, 
                               n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Apply Isotonic Regression recalibration using cross-validation.
    
    Returns:
        Recalibrated probabilities
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    y_prob_recal = np.zeros_like(y_prob)
    
    for train_idx, val_idx in kf.split(y_prob):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(y_prob[train_idx], y_true[train_idx])
        y_prob_recal[val_idx] = ir.transform(y_prob[val_idx])
    
    return y_prob_recal


def calculate_calibration_with_ci(y_true: np.ndarray, y_prob: np.ndarray,
                                   n_bins: int = 10, strategy: str = 'quantile',
                                   n_bootstrap: int = 200, random_state: int = 42) -> dict:
    """
    Calculate calibration curve with 95% CI using stratified bootstrap.
    """
    # Main calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    
    # Stratified bootstrap for CI
    rng = np.random.RandomState(random_state)
    bootstrap_true = []
    
    for b in range(n_bootstrap):
        indices = stratified_bootstrap_indices(y_true, rng)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        try:
            prob_true_boot, _ = calibration_curve(y_true_boot, y_prob_boot, 
                                                   n_bins=n_bins, strategy=strategy)
            if len(prob_true_boot) == len(prob_true):
                bootstrap_true.append(prob_true_boot)
        except:
            continue
    
    if len(bootstrap_true) > 0:
        bootstrap_true = np.array(bootstrap_true)
        lower_ci = np.percentile(bootstrap_true, 2.5, axis=0)
        upper_ci = np.percentile(bootstrap_true, 97.5, axis=0)
    else:
        lower_ci = prob_true
        upper_ci = prob_true
    
    return {
        'prob_pred': prob_pred,
        'prob_true': prob_true,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
    }


# %%
# =============================================================================
# YOUDEN INDEX FOR OPTIMAL THRESHOLD
# =============================================================================

def calculate_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate optimal threshold using Youden Index.
    
    Youden Index = Sensitivity + Specificity - 1 = TPR - FPR
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
    
    Returns:
        Tuple of (optimal_threshold, sensitivity, specificity)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Youden Index = TPR - FPR (or equivalently, Sensitivity + Specificity - 1)
    youden_index = tpr - fpr
    
    # Find the threshold that maximizes Youden Index
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    
    return optimal_threshold, optimal_sensitivity, optimal_specificity


# %%
# =============================================================================
# LEAD TIME FUNCTIONS
# =============================================================================

def calculate_lead_times(df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """
    Calculate lead time for each AKI patient.
    
    Lead Time = hours_to_aki at FIRST alert (first prediction >= threshold)
    
    IMPORTANT: Excludes predictions made AFTER AKI onset (hours_to_aki <= 0)
    
    Args:
        df: Online prediction DataFrame with visit_id, label, pred_proba_1, hours_to_aki
        threshold: Alert threshold
    
    Returns:
        Array of lead times (in hours) for AKI patients who received an alert
    """
    # Filter to AKI patients only
    df_aki = df[df['label'] == 1].copy()
    
    if len(df_aki) == 0:
        return np.array([])
    
    # IMPORTANT: Exclude predictions made after AKI onset
    df_aki = df_aki[df_aki['hours_to_aki'] > 0]
    
    lead_times = []
    
    for visit_id, group in df_aki.groupby('visit_id'):
        # Sort by hours_to_aki descending (earliest prediction first)
        group = group.sort_values('hours_to_aki', ascending=False)
        
        # Find rows where alert was raised (before AKI onset)
        alerts = group[group['pred_proba_1'] >= threshold]
        
        if len(alerts) > 0:
            # First alert = maximum hours_to_aki among alerts
            first_alert_time = alerts['hours_to_aki'].max()
            lead_times.append(first_alert_time)
    
    return np.array(lead_times)


# %%
# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_figure_3():
    """
    Create Figure 3: Clinical Utility of the Transformer Model (PH 0h)
    
    Structure: 
    - Row 1: Panel A (Before Recal) | Panel B (After Recal)
    - Row 2: Panel C (Lead Time) spanning full width
    """
    # Set up figure with publication-quality settings
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8], hspace=0.35, wspace=0.30)
    
    ax_before = fig.add_subplot(gs[0, 0])  # Panel A
    ax_after = fig.add_subplot(gs[0, 1])   # Panel B
    ax_lead = fig.add_subplot(gs[1, :])    # Panel C spans both columns
    
    # Data storage for CSV export
    calibration_data = []
    leadtime_data = []
    
    # =========================================================================
    # Load data and apply recalibration for all sites
    # =========================================================================
    print("Loading data and applying recalibration...")
    
    site_data = {}
    for site_name, config in SITE_CONFIGS.items():
        print(f"  Processing {site_name}...")
        df = load_single_predictions(site_name)
        y_true = df['label'].values
        y_prob_orig = df['pred_proba'].values
        
        # Apply isotonic recalibration
        y_prob_recal = isotonic_recalibration_cv(y_true, y_prob_orig)
        
        # Calculate Brier scores
        brier_before = brier_score_loss(y_true, y_prob_orig)
        brier_after = brier_score_loss(y_true, y_prob_recal)
        
        site_data[site_name] = {
            'y_true': y_true,
            'y_prob_orig': y_prob_orig,
            'y_prob_recal': y_prob_recal,
            'brier_before': brier_before,
            'brier_after': brier_after,
            'n': len(df),
            'n_aki': y_true.sum(),
        }
        
        print(f"    N={len(df):,}, AKI={y_true.sum():,}")
        print(f"    Brier Before: {brier_before:.4f}, After: {brier_after:.4f}")
    
    # =========================================================================
    # Panel A: Calibration BEFORE Recalibration
    # =========================================================================
    print("\nGenerating Panel A: Calibration Before Recalibration...")
    
    # Perfect calibration line
    ax_before.plot([0, 1], [0, 1], color=COLORS['perfect'], linestyle='--', 
                   linewidth=1.5, label='Perfect', zorder=1)
    
    for site_name, config in SITE_CONFIGS.items():
        data = site_data[site_name]
        
        # Calculate calibration
        cal_data = calculate_calibration_with_ci(data['y_true'], data['y_prob_orig'], 
                                                  n_bins=10, strategy='quantile', n_bootstrap=200)
        
        # Store for CSV
        for i, (pred, obs, lower, upper) in enumerate(zip(
            cal_data['prob_pred'], cal_data['prob_true'], 
            cal_data['lower_ci'], cal_data['upper_ci']
        )):
            calibration_data.append({
                'Site': site_name,
                'Stage': 'Before',
                'Bin': i + 1,
                'Predicted_Prob': pred,
                'Observed_Prob': obs,
                'CI_Lower': lower,
                'CI_Upper': upper,
                'Brier': data['brier_before'],
                'N': data['n'],
                'N_AKI': data['n_aki'],
            })
        
        # Plot
        ax_before.errorbar(
            cal_data['prob_pred'], cal_data['prob_true'],
            yerr=[cal_data['prob_true'] - cal_data['lower_ci'], 
                  cal_data['upper_ci'] - cal_data['prob_true']],
            fmt=config['marker'], color=config['color'], 
            markersize=7, capsize=3, capthick=1.2,
            elinewidth=1.2, markeredgecolor='white', markeredgewidth=0.8,
            label=f"{site_name} (Brier={data['brier_before']:.3f})", zorder=3
        )
        ax_before.plot(cal_data['prob_pred'], cal_data['prob_true'],
                       color=config['color'], linestyle='-', linewidth=1.2,
                       alpha=0.5, zorder=2)
    
    ax_before.set_xlim([-0.02, 1.02])
    ax_before.set_ylim([-0.02, 1.05])
    ax_before.set_xlabel('Predicted Probability')
    ax_before.set_ylabel('Observed Frequency')
    ax_before.set_title('Before Recalibration')
    ax_before.legend(loc='upper left', framealpha=0.95, edgecolor='none', fontsize=8)
    ax_before.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax_before.set_aspect('equal')
    ax_before.text(-0.15, 1.05, 'A', transform=ax_before.transAxes,
                   fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel B: Calibration AFTER Recalibration
    # =========================================================================
    print("Generating Panel B: Calibration After Recalibration...")
    
    # Perfect calibration line
    ax_after.plot([0, 1], [0, 1], color=COLORS['perfect'], linestyle='--', 
                  linewidth=1.5, label='Perfect', zorder=1)
    
    for site_name, config in SITE_CONFIGS.items():
        data = site_data[site_name]
        
        # Calculate calibration on recalibrated probabilities
        cal_data = calculate_calibration_with_ci(data['y_true'], data['y_prob_recal'], 
                                                  n_bins=10, strategy='quantile', n_bootstrap=200)
        
        # Store for CSV
        for i, (pred, obs, lower, upper) in enumerate(zip(
            cal_data['prob_pred'], cal_data['prob_true'], 
            cal_data['lower_ci'], cal_data['upper_ci']
        )):
            calibration_data.append({
                'Site': site_name,
                'Stage': 'After',
                'Bin': i + 1,
                'Predicted_Prob': pred,
                'Observed_Prob': obs,
                'CI_Lower': lower,
                'CI_Upper': upper,
                'Brier': data['brier_after'],
                'N': data['n'],
                'N_AKI': data['n_aki'],
            })
        
        # Plot
        ax_after.errorbar(
            cal_data['prob_pred'], cal_data['prob_true'],
            yerr=[cal_data['prob_true'] - cal_data['lower_ci'], 
                  cal_data['upper_ci'] - cal_data['prob_true']],
            fmt=config['marker'], color=config['color'], 
            markersize=7, capsize=3, capthick=1.2,
            elinewidth=1.2, markeredgecolor='white', markeredgewidth=0.8,
            label=f"{site_name} (Brier={data['brier_after']:.3f})", zorder=3
        )
        ax_after.plot(cal_data['prob_pred'], cal_data['prob_true'],
                      color=config['color'], linestyle='-', linewidth=1.2,
                      alpha=0.5, zorder=2)
    
    ax_after.set_xlim([-0.02, 1.02])
    ax_after.set_ylim([-0.02, 1.05])
    ax_after.set_xlabel('Predicted Probability')
    ax_after.set_ylabel('Observed Frequency')
    ax_after.set_title('After Isotonic Recalibration')
    ax_after.legend(loc='upper left', framealpha=0.95, edgecolor='none', fontsize=8)
    ax_after.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax_after.set_aspect('equal')
    ax_after.text(-0.15, 1.05, 'B', transform=ax_after.transAxes,
                   fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel C: Lead Time Distribution (Box Plot)
    # =========================================================================
    print("\nGenerating Panel C: Lead Time Distribution...")
    
    # Calculate Youden Index threshold from Development set (NHIS) only
    dev_data = site_data["NHIS"]
    dev_threshold, dev_sens, dev_spec = calculate_youden_threshold(dev_data['y_true'], dev_data['y_prob_orig'])
    print(f"  Development set (NHIS) Youden threshold: {dev_threshold:.4f}")
    print(f"    Sensitivity: {dev_sens:.3f}, Specificity: {dev_spec:.3f}")
    print(f"  Applying this threshold to all sites...\n")
    
    lead_time_results = []
    site_names = []
    site_colors = []
    
    for site_name, config in SITE_CONFIGS.items():
        print(f"  Processing {site_name}...")
        
        # Load online data and calculate lead times
        df_online = load_online_predictions(site_name)
        
        # Total AKI patients
        total_aki = df_online.groupby('visit_id')['label'].max().sum()
        
        # Calculate lead times using Development set threshold (excluding post-onset predictions)
        lead_times = calculate_lead_times(df_online, threshold=dev_threshold)
        
        if len(lead_times) > 0:
            lead_time_results.append(lead_times)
            site_names.append(site_name)
            site_colors.append(config['color'])
            
            median_lt = np.median(lead_times)
            q1 = np.percentile(lead_times, 25)
            q3 = np.percentile(lead_times, 75)
            
            # Store for CSV
            leadtime_data.append({
                'Site': site_name,
                'Threshold': dev_threshold,
                'Threshold_Source': 'Ilsan (Development)',
                'N_AKI_Total': int(total_aki),
                'N_Alerted': len(lead_times),
                'Alert_Rate': len(lead_times) / total_aki if total_aki > 0 else 0,
                'Median': median_lt,
                'Q1': q1,
                'Q3': q3,
                'Min': np.min(lead_times),
                'Max': np.max(lead_times),
                'Mean': np.mean(lead_times),
                'Std': np.std(lead_times),
            })
            
            print(f"    AKI: {total_aki}, Alerted: {len(lead_times)} ({len(lead_times)/total_aki*100:.1f}%)")
            print(f"    Median: {median_lt:.1f}h, IQR: [{q1:.1f}, {q3:.1f}]")
    
    # Create box plot
    positions = range(1, len(site_names) + 1)
    bp = ax_lead.boxplot(lead_time_results, positions=positions, patch_artist=True,
                          widths=0.5, showfliers=True,
                          flierprops=dict(marker='o', markersize=3, alpha=0.3))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], site_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    
    # Style whiskers and caps
    for i in range(len(site_names)):
        bp['whiskers'][i*2].set(color=site_colors[i], linewidth=1.5)
        bp['whiskers'][i*2+1].set(color=site_colors[i], linewidth=1.5)
        bp['caps'][i*2].set(color=site_colors[i], linewidth=1.5)
        bp['caps'][i*2+1].set(color=site_colors[i], linewidth=1.5)
    
    for median in bp['medians']:
        median.set(color='white', linewidth=2.5)
    
    for flier, color in zip(bp['fliers'], site_colors):
        flier.set(markerfacecolor=color, markeredgecolor=color, alpha=0.3)
    
    # Add median annotations
    for i, (site_name, lead_times) in enumerate(zip(site_names, lead_time_results)):
        median_val = np.median(lead_times)
        q3_val = np.percentile(lead_times, 75)
        ax_lead.annotate(f'{median_val:.0f}h', 
                        xy=(i + 1, median_val), 
                        xytext=(i + 1, q3_val + 20),
                        fontsize=10, fontweight='bold',
                        ha='center', va='bottom',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                  edgecolor=site_colors[i], alpha=0.9))
    
    # Set x-axis labels
    ax_lead.set_xticks(positions)
    labels = [f"{name}\n(n={leadtime_data[i]['N_Alerted']})" for i, name in enumerate(site_names)]
    ax_lead.set_xticklabels(labels)
    
    # Styling
    ax_lead.set_ylabel('Lead Time (hours before AKI)')
    ax_lead.set_xlabel('')
    ax_lead.set_title(f'Lead Time Distribution (Threshold={dev_threshold:.2f} from Development Set)')
    ax_lead.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, axis='y')
    
    # Add reference lines
    for hours, days_text in [(24, '1 day'), (48, '2 days'), (72, '3 days')]:
        ax_lead.axhline(y=hours, color='#BDBDBD', linestyle='--', alpha=0.7, linewidth=1, zorder=0)
        ax_lead.text(len(site_names) + 0.6, hours, f'{hours}h ({days_text})',
                     fontsize=8, color='#757575', va='center', ha='left')
    
    ax_lead.set_xlim([0.4, len(site_names) + 0.9])
    ax_lead.set_ylim([0, 500])  # Limit y-axis to reduce outlier distortion
    ax_lead.text(-0.05, 1.08, 'C', transform=ax_lead.transAxes,
                 fontsize=14, fontweight='bold', va='top')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # Title removed for publication
    
    # Save CSV files
    os.makedirs(TABLE_PATH, exist_ok=True)
    
    df_cal = pd.DataFrame(calibration_data)
    df_cal.to_csv(f"{TABLE_PATH}/figure3ab_calibration.csv", index=False)
    df_cal.to_excel(f"{TABLE_PATH}/figure3ab_calibration.xlsx", index=False)
    print(f"\nCalibration data saved to: {TABLE_PATH}/figure3ab_calibration.csv")
    
    df_lead = pd.DataFrame(leadtime_data)
    df_lead.to_csv(f"{TABLE_PATH}/figure3c_leadtime.csv", index=False)
    df_lead.to_excel(f"{TABLE_PATH}/figure3c_leadtime.xlsx", index=False)
    print(f"Lead time data saved to: {TABLE_PATH}/figure3c_leadtime.csv")
    
    return fig


def create_figure_3_enhanced():
    """
    Create an enhanced version of Figure 3 with improved aesthetics.
    """
    # Set up figure with publication-quality settings
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.85], hspace=0.32, wspace=0.28)
    
    ax_before = fig.add_subplot(gs[0, 0])  # Panel A
    ax_after = fig.add_subplot(gs[0, 1])   # Panel B
    ax_lead = fig.add_subplot(gs[1, :])    # Panel C spans both columns
    
    # =========================================================================
    # Load data and apply recalibration for all sites
    # =========================================================================
    print("Loading data and applying recalibration...")
    
    site_data = {}
    for site_name, config in SITE_CONFIGS.items():
        print(f"  Processing {site_name}...")
        df = load_single_predictions(site_name)
        y_true = df['label'].values
        y_prob_orig = df['pred_proba'].values
        
        # Apply isotonic recalibration
        y_prob_recal = isotonic_recalibration_cv(y_true, y_prob_orig)
        
        # Calculate Brier scores
        brier_before = brier_score_loss(y_true, y_prob_orig)
        brier_after = brier_score_loss(y_true, y_prob_recal)
        
        site_data[site_name] = {
            'y_true': y_true,
            'y_prob_orig': y_prob_orig,
            'y_prob_recal': y_prob_recal,
            'brier_before': brier_before,
            'brier_after': brier_after,
            'n': len(df),
            'n_aki': y_true.sum(),
        }
        
        print(f"    N={len(df):,}, Brier: {brier_before:.4f} → {brier_after:.4f}")
    
    # =========================================================================
    # Panel A: Calibration BEFORE Recalibration
    # =========================================================================
    print("\nGenerating Panel A: Calibration Before Recalibration...")
    
    ax_before.fill_between([0, 1], [0, 1], [0, 1], color='gray', alpha=0.05, zorder=0)
    ax_before.plot([0, 1], [0, 1], color=COLORS['perfect'], linestyle='--', 
                   linewidth=1.5, label='Perfect', zorder=1)
    
    for site_name, config in SITE_CONFIGS.items():
        data = site_data[site_name]
        cal_data = calculate_calibration_with_ci(data['y_true'], data['y_prob_orig'], 
                                                  n_bins=10, strategy='quantile', n_bootstrap=500)
        
        ax_before.fill_between(cal_data['prob_pred'], cal_data['lower_ci'], cal_data['upper_ci'],
                               color=config['color'], alpha=0.12, zorder=2)
        ax_before.errorbar(
            cal_data['prob_pred'], cal_data['prob_true'],
            yerr=[cal_data['prob_true'] - cal_data['lower_ci'], 
                  cal_data['upper_ci'] - cal_data['prob_true']],
            fmt=config['marker'], color=config['color'], 
            markersize=8, capsize=3, capthick=1.2,
            elinewidth=1.2, markeredgecolor='white', markeredgewidth=1,
            label=f"{site_name}", zorder=4
        )
        ax_before.plot(cal_data['prob_pred'], cal_data['prob_true'],
                       color=config['color'], linestyle='-', linewidth=1.5,
                       alpha=0.6, zorder=3)
    
    ax_before.set_xlim([-0.02, 1.02])
    ax_before.set_ylim([-0.02, 1.05])
    ax_before.set_xlabel('Predicted Probability')
    ax_before.set_ylabel('Observed Frequency')
    ax_before.set_title('Before Recalibration')
    ax_before.legend(loc='upper left', framealpha=0.95, edgecolor='none', fontsize=8)
    ax_before.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax_before.set_aspect('equal')
    
    # Brier score box - below legend (upper left area)
    brier_text = "Brier Score:\n" + "\n".join([f"  {name}: {data['brier_before']:.3f}" 
                                                for name, data in site_data.items()])
    ax_before.text(0.02, 0.55, brier_text, transform=ax_before.transAxes, ha='left', va='top',
                   fontsize=7, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                         edgecolor='#cccccc', alpha=0.95), linespacing=1.2)
    
    ax_before.text(-0.15, 1.05, 'A', transform=ax_before.transAxes,
                   fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel B: Calibration AFTER Recalibration
    # =========================================================================
    print("Generating Panel B: Calibration After Recalibration...")
    
    ax_after.fill_between([0, 1], [0, 1], [0, 1], color='gray', alpha=0.05, zorder=0)
    ax_after.plot([0, 1], [0, 1], color=COLORS['perfect'], linestyle='--', 
                  linewidth=1.5, label='Perfect', zorder=1)
    
    for site_name, config in SITE_CONFIGS.items():
        data = site_data[site_name]
        cal_data = calculate_calibration_with_ci(data['y_true'], data['y_prob_recal'], 
                                                  n_bins=10, strategy='quantile', n_bootstrap=500)
        
        ax_after.fill_between(cal_data['prob_pred'], cal_data['lower_ci'], cal_data['upper_ci'],
                              color=config['color'], alpha=0.12, zorder=2)
        ax_after.errorbar(
            cal_data['prob_pred'], cal_data['prob_true'],
            yerr=[cal_data['prob_true'] - cal_data['lower_ci'], 
                  cal_data['upper_ci'] - cal_data['prob_true']],
            fmt=config['marker'], color=config['color'], 
            markersize=8, capsize=3, capthick=1.2,
            elinewidth=1.2, markeredgecolor='white', markeredgewidth=1,
            label=f"{site_name}", zorder=4
        )
        ax_after.plot(cal_data['prob_pred'], cal_data['prob_true'],
                      color=config['color'], linestyle='-', linewidth=1.5,
                      alpha=0.6, zorder=3)
    
    ax_after.set_xlim([-0.02, 1.02])
    ax_after.set_ylim([-0.02, 1.05])
    ax_after.set_xlabel('Predicted Probability')
    ax_after.set_ylabel('Observed Frequency')
    ax_after.set_title('After Isotonic Recalibration')
    ax_after.legend(loc='upper left', framealpha=0.95, edgecolor='none', fontsize=8)
    ax_after.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax_after.set_aspect('equal')
    
    # Brier score box - below legend (upper left area)
    brier_text = "Brier Score:\n" + "\n".join([f"  {name}: {data['brier_after']:.3f}" 
                                                for name, data in site_data.items()])
    ax_after.text(0.02, 0.55, brier_text, transform=ax_after.transAxes, ha='left', va='top',
                  fontsize=7, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor='#cccccc', alpha=0.95), linespacing=1.2)
    
    ax_after.text(-0.15, 1.05, 'B', transform=ax_after.transAxes,
                  fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel C: Lead Time Distribution (Box Plot)
    # =========================================================================
    print("\nGenerating Panel C: Lead Time Distribution...")
    
    # Calculate Youden Index threshold from Development set (NHIS) only
    dev_data = site_data["NHIS"]
    dev_threshold, dev_sens, dev_spec = calculate_youden_threshold(dev_data['y_true'], dev_data['y_prob_orig'])
    print(f"  Development set (NHIS) Youden threshold: {dev_threshold:.4f}")
    print(f"    Sensitivity: {dev_sens:.3f}, Specificity: {dev_spec:.3f}")
    print(f"  Applying this threshold to all sites...\n")
    
    lead_time_results = []
    site_names = []
    site_colors = []
    leadtime_stats = []
    
    for site_name, config in SITE_CONFIGS.items():
        print(f"  Processing {site_name}...")
        
        # Load online data and calculate lead times
        df_online = load_online_predictions(site_name)
        
        total_aki = df_online.groupby('visit_id')['label'].max().sum()
        lead_times = calculate_lead_times(df_online, threshold=dev_threshold)
        
        if len(lead_times) > 0:
            lead_time_results.append(lead_times)
            site_names.append(site_name)
            site_colors.append(config['color'])
            
            median_lt = np.median(lead_times)
            q1 = np.percentile(lead_times, 25)
            q3 = np.percentile(lead_times, 75)
            
            leadtime_stats.append({
                'name': site_name,
                'n': len(lead_times),
                'total': total_aki,
                'median': median_lt,
                'q1': q1,
                'q3': q3,
            })
            
            print(f"    Alerted: {len(lead_times)}/{total_aki} ({len(lead_times)/total_aki*100:.1f}%)")
            print(f"    Median: {median_lt:.1f}h, IQR: [{q1:.1f}, {q3:.1f}]")
    
    # Create box plot
    positions = range(1, len(site_names) + 1)
    bp = ax_lead.boxplot(lead_time_results, positions=positions, patch_artist=True,
                          widths=0.55, showfliers=True,
                          flierprops=dict(marker='o', markersize=3, alpha=0.2))
    
    # Color the boxes
    for i, (patch, color) in enumerate(zip(bp['boxes'], site_colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
    
    # Style whiskers and caps
    for i in range(len(site_names)):
        bp['whiskers'][i*2].set(color=site_colors[i], linewidth=1.5)
        bp['whiskers'][i*2+1].set(color=site_colors[i], linewidth=1.5)
        bp['caps'][i*2].set(color=site_colors[i], linewidth=1.5)
        bp['caps'][i*2+1].set(color=site_colors[i], linewidth=1.5)
    
    for median in bp['medians']:
        median.set(color='white', linewidth=3)
    
    for flier, color in zip(bp['fliers'], site_colors):
        flier.set(markerfacecolor=color, markeredgecolor=color, alpha=0.2)
    
    # Add median annotations with days
    for i, stats in enumerate(leadtime_stats):
        median_val = stats['median']
        days = median_val / 24
        ax_lead.annotate(f'{median_val:.0f}h\n({days:.1f} days)', 
                        xy=(i + 1, stats['q3'] + 10), 
                        fontsize=9, fontweight='bold',
                        ha='center', va='bottom',
                        color=site_colors[i],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  edgecolor=site_colors[i], alpha=0.9, linewidth=1.5))
    
    # Set x-axis labels with sample sizes
    ax_lead.set_xticks(positions)
    labels = [f"{stats['name']}\n(n={stats['n']}/{stats['total']})" for stats in leadtime_stats]
    ax_lead.set_xticklabels(labels, fontsize=10)
    
    # Styling
    ax_lead.set_ylabel('Lead Time (hours before AKI onset)', fontsize=11)
    ax_lead.set_xlabel('')
    ax_lead.set_title(f'Lead Time Distribution (Threshold={dev_threshold:.2f} from Development Set)', fontsize=12)
    ax_lead.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, axis='y')
    
    # Add reference lines
    for hours, days_text in [(24, '1 day'), (48, '2 days'), (72, '3 days'), (96, '4 days')]:
        ax_lead.axhline(y=hours, color='#E0E0E0', linestyle='-', alpha=0.8, linewidth=1, zorder=0)
        ax_lead.text(len(site_names) + 0.55, hours, f'{hours}h ({days_text})',
                     fontsize=8, color='#757575', va='center', ha='left')
    
    ax_lead.set_xlim([0.4, len(site_names) + 0.85])
    ax_lead.set_ylim([0, 500])  # Limit y-axis to reduce outlier distortion
    ax_lead.text(-0.05, 1.08, 'C', transform=ax_lead.transAxes,
                 fontsize=14, fontweight='bold', va='top')
    
    # Add key message annotation (dynamically calculated)
    median_values = [stats['median'] for stats in leadtime_stats]
    if len(median_values) >= 2:
        min_med, max_med = min(median_values[:2]), max(median_values[:2])  # Use Ilsan & CCHLMC
        message = f"Key: Median lead time {min_med:.0f}-{max_med:.0f}h ({min_med/24:.1f}-{max_med/24:.1f} days) → Sufficient intervention window"
    else:
        message = "Key: Sufficient intervention window for clinical action"
    ax_lead.text(0.5, -0.15, message, transform=ax_lead.transAxes,
                 ha='center', va='top', fontsize=9, fontstyle='italic', color='#424242')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.94, top=0.92, bottom=0.10)
    
    # Title removed for publication
    
    return fig


# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create output directories
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(TABLE_PATH, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 3: Clinical Utility of the Transformer Model (PH 0h)")
    print("=" * 80)
    print("\nAlert Threshold for Lead Time: Youden Index (per site)")
    print("Calibration: Using single prediction data (NOT online)")
    print("Lead Time: Using online simulation data")
    print("  - First alert → AKI onset")
    print("  - Excluding predictions made AFTER AKI onset (hours_to_aki <= 0)\n")
    
    # Helper function to save in multiple formats
    def save_figure_all_formats(fig, base_path, dpi=500):
        """Save figure in png, pdf, svg, and tiff formats."""
        for fmt in ['png', 'pdf', 'svg', 'tiff']:
            fig.savefig(
                f"{base_path}.{fmt}",
                dpi=dpi, bbox_inches='tight', facecolor='white',
                format=fmt
            )
        print(f"  Saved: {base_path}.{{png,pdf,svg,tiff}}")
    
    # Basic version
    fig3 = create_figure_3()
    save_figure_all_formats(fig3, f"{OUTPUT_PATH}/figure3_clinical_utility")
    print(f"\nFigure 3 (basic) saved to: {OUTPUT_PATH}")
    
    # Enhanced version
    fig3_enhanced = create_figure_3_enhanced()
    save_figure_all_formats(fig3_enhanced, f"{OUTPUT_PATH}/figure3_clinical_utility_enhanced")
    print(f"Figure 3 (enhanced) saved to: {OUTPUT_PATH}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

# %%
