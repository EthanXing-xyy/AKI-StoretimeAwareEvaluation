# %%
"""
Figure 1: Discrimination Performance in Development Setting
============================================================
Generates a 2 rows × 3 columns figure:
- Row 1 (AUROC): LSTM, CNN, Transformer ROC curves
- Row 2 (AUPRC): LSTM, CNN, Transformer PR curves

Each panel contains 3 curves for PH 0h, 48h, 72h.
Uses internal validation data (ilsan_test) only.
"""

import os
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Color scheme for prediction horizons
COLORS = {
    0: "#2E86AB",   # PH 0h: Blue
    48: "#E94F37",  # PH 48h: Orange
    72: "#7D7D7D",  # PH 72h: Gray
}

# Line styles for prediction horizons
LINE_STYLES = {
    0: "-",      # PH 0h: solid
    48: "--",    # PH 48h: dashed
    72: ":",     # PH 72h: dotted
}

# Model configurations
MODEL_CONFIGS = {
    "LSTM": {
        "base_path": "../../../result/predictions/ilsan/lstm_attention",
        "path_template": "prediction_window_{pw}/ilsan_test.parquet",
    },
    "CNN": {
        "base_path": "../../../result/predictions/ilsan/mask_rms_cnn",
        "path_template": "prediction_window_{pw}/resolution_control/apply_prob_0.0/ilsan_test.parquet",
    },
    "Transformer": {
        "base_path": "../../../result/predictions/ilsan/ite_transformer",
        "path_template": "prediction_window_{pw}/ilsan_test.parquet",
    },
}

# Prediction horizons
PREDICTION_HORIZONS = [0, 48, 72]

# Output path
OUTPUT_PATH = "../../../result/draft/figure"


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


def load_predictions(model_name: str, prediction_window: int) -> pd.DataFrame:
    """
    Load prediction data for a specific model and prediction window.
    Uses development (internal validation) data only - ilsan_test.parquet
    """
    config = MODEL_CONFIGS[model_name]
    path = f"{config['base_path']}/{config['path_template'].format(pw=prediction_window)}"
    
    df = pd.read_parquet(path, engine='fastparquet')
    df['pred_proba'] = decode_pred_proba(df['pred_proba'])
    
    return df.reset_index(drop=True)


# %%
# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_figure_1():
    """
    Create Figure 1: Discrimination Performance in Development Setting
    
    Structure: 2 rows × 3 columns
    - Row 1 (AUROC): LSTM, CNN, Transformer ROC curves
    - Row 2 (AUPRC): LSTM, CNN, Transformer PR curves
    """
    # Set up figure with publication-quality settings
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    models = ['LSTM', 'CNN', 'Transformer']
    
    # Load all data
    print("Loading prediction data...")
    all_data = {}
    for model in models:
        all_data[model] = {}
        for ph in PREDICTION_HORIZONS:
            df = load_predictions(model, ph)
            all_data[model][ph] = df
            auroc = roc_auc_score(df['label'], df['pred_proba'])
            auprc = average_precision_score(df['label'], df['pred_proba'])
            print(f"  {model} PH={ph}h: n={len(df):,}, AUROC={auroc:.3f}, AUPRC={auprc:.3f}")
    
    # Row 1: ROC Curves (AUROC)
    print("\nGenerating ROC curves...")
    for col_idx, model in enumerate(models):
        ax = axes[0, col_idx]
        
        for ph in PREDICTION_HORIZONS:
            df = all_data[model][ph]
            y_true = df['label'].values
            y_prob = df['pred_proba'].values
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auroc = roc_auc_score(y_true, y_prob)
            
            # Plot ROC curve
            ax.plot(
                fpr, tpr,
                color=COLORS[ph],
                linestyle=LINE_STYLES[ph],
                linewidth=2,
                label=f"PH {ph}h (AUROC={auroc:.3f})"
            )
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Reference')
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('1 - Specificity (FPR)')
        ax.set_ylabel('Sensitivity (TPR)')
        ax.set_title(f'{model}', fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Row 2: Precision-Recall Curves (AUPRC)
    print("Generating PR curves...")
    for col_idx, model in enumerate(models):
        ax = axes[1, col_idx]
        
        # Get baseline (prevalence) for reference
        df_ref = all_data[model][0]
        baseline_prevalence = df_ref['label'].mean()
        
        for ph in PREDICTION_HORIZONS:
            df = all_data[model][ph]
            y_true = df['label'].values
            y_prob = df['pred_proba'].values
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            
            # Plot PR curve
            ax.plot(
                recall, precision,
                color=COLORS[ph],
                linestyle=LINE_STYLES[ph],
                linewidth=2,
                label=f"PH {ph}h (AUPRC={auprc:.3f})"
            )
        
        # Baseline reference (horizontal line at prevalence)
        ax.axhline(y=baseline_prevalence, color='k', linestyle='--', 
                   linewidth=1, alpha=0.5, label=f'Baseline ({baseline_prevalence:.3f})')
        
        # Styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (PPV)')
        ax.set_title(f'{model}', fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    # Add row labels
    fig.text(0.02, 0.72, 'AUROC', fontsize=14, fontweight='bold', 
             rotation=90, va='center')
    fig.text(0.02, 0.28, 'AUPRC', fontsize=14, fontweight='bold', 
             rotation=90, va='center')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, top=0.92, hspace=0.25, wspace=0.25)
    
    # Title removed for publication
    
    return fig


def create_figure_1_enhanced():
    """
    Create an enhanced version of Figure 1 with improved aesthetics.
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
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    models = ['LSTM', 'CNN', 'Transformer']
    
    # Load all data
    print("Loading prediction data...")
    all_data = {}
    for model in models:
        all_data[model] = {}
        for ph in PREDICTION_HORIZONS:
            df = load_predictions(model, ph)
            all_data[model][ph] = df
    
    # Row 1: ROC Curves (AUROC)
    print("Generating ROC curves...")
    for col_idx, model in enumerate(models):
        ax = axes[0, col_idx]
        
        # Create legend entries
        legend_elements = []
        
        for ph in PREDICTION_HORIZONS:
            df = all_data[model][ph]
            y_true = df['label'].values
            y_prob = df['pred_proba'].values
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auroc = roc_auc_score(y_true, y_prob)
            
            line, = ax.plot(
                fpr, tpr,
                color=COLORS[ph],
                linestyle=LINE_STYLES[ph],
                linewidth=2.5,
            )
            legend_elements.append(
                Line2D([0], [0], color=COLORS[ph], linestyle=LINE_STYLES[ph],
                       linewidth=2, label=f"PH {ph}h: {auroc:.3f}")
            )
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
        
        # Fill area under curve for visual emphasis (optional, for PH 0h)
        # ax.fill_between(fpr_0h, tpr_0h, alpha=0.1, color=COLORS[0])
        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('1 - Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title(f'{model}')
        ax.legend(handles=legend_elements, loc='lower right', 
                  framealpha=0.95, edgecolor='none',
                  title='AUROC')
        ax.set_aspect('equal')
        
        # Add panel label
        panel_label = chr(65 + col_idx)  # A, B, C
        ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')
    
    # Row 2: Precision-Recall Curves (AUPRC)
    print("Generating PR curves...")
    for col_idx, model in enumerate(models):
        ax = axes[1, col_idx]
        
        df_ref = all_data[model][0]
        baseline_prevalence = df_ref['label'].mean()
        
        legend_elements = []
        
        for ph in PREDICTION_HORIZONS:
            df = all_data[model][ph]
            y_true = df['label'].values
            y_prob = df['pred_proba'].values
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            
            ax.plot(
                recall, precision,
                color=COLORS[ph],
                linestyle=LINE_STYLES[ph],
                linewidth=2.5,
            )
            legend_elements.append(
                Line2D([0], [0], color=COLORS[ph], linestyle=LINE_STYLES[ph],
                       linewidth=2, label=f"PH {ph}h: {auprc:.3f}")
            )
        
        # Baseline reference
        ax.axhline(y=baseline_prevalence, color='k', linestyle='--', 
                   linewidth=1, alpha=0.4)
        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model}')
        ax.legend(handles=legend_elements, loc='lower left',
                  framealpha=0.95, edgecolor='none',
                  title='AUPRC')
        
        # Add panel label
        panel_label = chr(68 + col_idx)  # D, E, F
        ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')
    
    # Add row labels on the left
    fig.text(0.01, 0.72, 'ROC Curves', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center')
    fig.text(0.01, 0.28, 'Precision-Recall Curves', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.10, top=0.92, hspace=0.30, wspace=0.30)
    
    # Title removed for publication
    
    return fig


# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Generate Figure 1 (basic version)
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 1: Discrimination Performance in Development Setting")
    print("=" * 80)
    print("\nUsing internal validation data (ilsan_test.parquet)")
    print("NOT using online data\n")
    
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
    fig1 = create_figure_1()
    save_figure_all_formats(fig1, f"{OUTPUT_PATH}/figure1_discrimination_develop")
    print(f"\nFigure 1 (basic) saved to: {OUTPUT_PATH}")
    
    # Enhanced version
    fig1_enhanced = create_figure_1_enhanced()
    save_figure_all_formats(fig1_enhanced, f"{OUTPUT_PATH}/figure1_discrimination_develop_enhanced")
    print(f"Figure 1 (enhanced) saved to: {OUTPUT_PATH}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

# %%

