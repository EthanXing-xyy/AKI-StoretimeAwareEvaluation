# %%
"""
Figure 2: Online Simulation Performance
==========================================
Generates a 1 row × 3 columns figure:
- Panel A: NHIS (Internal Validation)
- Panel B: CSHH (External Validation)
- Panel C: MIMIC-IV (External Validation)

Each panel contains 9 lines: 3 models × 3 prediction horizons
- X-axis: Hours before AKI onset (72 → 0)
- Y-axis: AUROC (0.5 ~ 1.0)

Line style:
- LSTM: solid / CNN: dashed / Transformer: dotted
- PH 0h: Blue / PH 48h: Orange / PH 72h: Gray

Key emphasis:
- PH 0h models: smooth upward gradient (→ clinical faithfulness)
- PH 48h/72h models: plateau or fluctuation (→ unstable)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory (resolve relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

# Color scheme for prediction horizons
# PH 0h is emphasized with vivid blue, others are muted
COLORS = {
    0: "#1565C0",   # PH 0h: Strong Blue (emphasized - best performance)
    48: "#E57373",  # PH 48h: Muted Red/Coral
    72: "#9E9E9E",  # PH 72h: Gray (least emphasized)
}

# Alpha values for prediction horizons (PH 0h most visible)
ALPHAS = {
    0: 1.0,    # PH 0h: fully opaque (emphasized)
    48: 0.6,   # PH 48h: semi-transparent
    72: 0.5,   # PH 72h: more transparent
}

# Line width multiplier by prediction horizon
PH_LINE_WIDTH_MULT = {
    0: 1.3,    # PH 0h: thicker lines
    48: 1.0,   # PH 48h: normal
    72: 0.9,   # PH 72h: thinner
}

# Line styles for models (Transformer = solid for emphasis)
LINE_STYLES = {
    "Transformer": "-",  # Transformer: solid (main model)
    "LSTM": "--",        # LSTM: dashed
    "CNN": ":",          # CNN: dotted
}

# Line width by model for visual distinction
LINE_WIDTHS = {
    "Transformer": 2.5,  # Thicker for emphasis
    "LSTM": 2.0,
    "CNN": 2.0,
}

# Marker styles for models
MARKERS = {
    "LSTM": "o",
    "CNN": "s",
    "Transformer": "^",
}

# Dataset configurations
DATASET_CONFIGS = {
    "ilsan_test": {
        "display_name": "NHIS",
        "csv_file": f"{BASE_DIR}/result/tables/online_ilsan_test_performance.csv",
        "panel_label": "A",
        "type": "Internal",
        "model_prefix": {
            "Transformer": "ITE Transformer - NHIS (Internal)",
            "LSTM": "LSTM-Attention - NHIS (Internal)",
            "CNN": "Masked CNN - NHIS (Internal)",
        }
    },
    "cchlmc_external": {
        "display_name": "CSHH",
        "csv_file": f"{BASE_DIR}/result/tables/online_cchlmc_external_performance.csv",
        "panel_label": "B",
        "type": "External",
        "model_prefix": {
            "Transformer": "ITE Transformer - CSHH (External)",
            "LSTM": "LSTM-Attention - CSHH (External)",
            "CNN": "Masked CNN - CSHH (External)",
        }
    },
    "mimic-iv_external": {
        "display_name": "MIMIC-IV",
        "csv_file": f"{BASE_DIR}/result/tables/online_mimic_iv_external_performance.csv",
        "panel_label": "C",
        "type": "External",
        "model_prefix": {
            "Transformer": "ITE Transformer - MIMIC-IV (External)",
            "LSTM": "LSTM-Attention - MIMIC-IV (External)",
            "CNN": "Masked CNN - MIMIC-IV (External)",
        }
    },
}

# Prediction horizons
PREDICTION_HORIZONS = [0, 48, 72]

# Time points (from CSV column names)
TIME_POINTS = [0, 12, 24, 36, 48, 60, 72]

# Output path
OUTPUT_PATH = f"{BASE_DIR}/result/draft/figure"


# %%
# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_auroc_from_csv(dataset_key: str) -> Dict[str, Dict[int, List[float]]]:
    """
    Load AUROC values from pre-computed CSV files.
    
    Args:
        dataset_key: Key for the dataset (ilsan_test, cchlmc_external, mimic-iv_external)
        
    Returns:
        Nested dict: {model_name: {prediction_horizon: [auroc_values]}}
    """
    config = DATASET_CONFIGS[dataset_key]
    csv_path = config["csv_file"]
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for AUROC metric
    df_auroc = df[df["Metric"] == "AUROC"]
    
    result = {}
    for model_name, prefix in config["model_prefix"].items():
        result[model_name] = {}
        
        for ph in PREDICTION_HORIZONS:
            # Construct model key (e.g., "ITE Transformer - Ilsan (Internal) (PW=0h)")
            model_key = f"{prefix} (PW={ph}h)"
            
            # Get AUROC values for this model and prediction horizon
            row = df_auroc[df_auroc["Model"] == model_key]
            
            if len(row) > 0:
                # Extract AUROC values for each time point
                auroc_values = []
                for tp in TIME_POINTS:
                    col_name = f"{tp}h"
                    value = row[col_name].values[0]
                    auroc_values.append(float(value))
                result[model_name][ph] = auroc_values
            else:
                result[model_name][ph] = [np.nan] * len(TIME_POINTS)
    
    return result


# %%
# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_figure_2():
    """
    Create Figure 2: Online Simulation Performance
    
    Structure: 1 row × 3 columns
    - Panel A: Ilsan (Internal)
    - Panel B: CCHLMC (External)
    - Panel C: MIMIC-IV (External)
    
    Each panel: 9 lines (3 models × 3 prediction horizons)
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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = list(DATASET_CONFIGS.keys())
    models = ["Transformer", "LSTM", "CNN"]
    
    # Load all data from CSV files
    print("Loading AUROC data from CSV files...")
    all_data = {}
    
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        print(f"\n  {config['display_name']}:")
        
        all_data[dataset] = load_auroc_from_csv(dataset)
        
        for model in models:
            for ph in PREDICTION_HORIZONS:
                auroc_values = all_data[dataset][model][ph]
                valid_aurocs = [v for v in auroc_values if not np.isnan(v)]
                if valid_aurocs:
                    print(f"    {model} PH={ph}h: mean AUROC={np.mean(valid_aurocs):.3f}")
    
    # Plot each panel
    print("\nGenerating figure...")
    for col_idx, dataset in enumerate(datasets):
        ax = axes[col_idx]
        dataset_config = DATASET_CONFIGS[dataset]
        
        # Plot lines for each model and prediction horizon
        # Plot PH 48h and 72h first (background), then PH 0h on top (foreground)
        for ph in [72, 48, 0]:  # Reverse order so PH 0h is on top
            for model in models:
                auroc_values = all_data[dataset][model][ph]
                
                # Calculate line width with PH emphasis
                lw = LINE_WIDTHS[model] * PH_LINE_WIDTH_MULT[ph]
                
                # Plot line
                ax.plot(
                    TIME_POINTS,
                    auroc_values,
                    color=COLORS[ph],
                    linestyle=LINE_STYLES[model],
                    linewidth=lw,
                    alpha=ALPHAS[ph],
                    zorder=10 if ph == 0 else 5,  # PH 0h on top
                )
        
        # Styling
        ax.set_xlim([72, 0])  # Reversed: 72 → 0
        ax.set_ylim([0.5, 1.0])
        ax.set_xlabel('Hours before AKI onset')
        ax.set_ylabel('AUROC')
        
        # Title with validation type
        title = f"{dataset_config['display_name']}"
        if dataset_config['type'] == "Internal":
            title += " (Internal)"
        else:
            title += " (External)"
        ax.set_title(title)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # X-axis ticks
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
        ax.set_xticklabels(['0', '12', '24', '36', '48', '60', '72'])
        
        # Y-axis ticks
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # Panel label (A, B, C)
        ax.text(
            -0.12, 1.05, dataset_config['panel_label'],
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top'
        )
    
    # Create custom legend
    # Model types (line styles) - Transformer first as main model
    model_legend_elements = [
        Line2D([0], [0], color='black', linestyle=LINE_STYLES[m], 
               linewidth=LINE_WIDTHS[m], label=m)
        for m in models
    ]
    
    # Prediction horizons (colors) - PH 0h emphasized as recommended
    ph_legend_elements = [
        Line2D([0], [0], color=COLORS[0], linestyle='-', 
               linewidth=3, alpha=ALPHAS[0], label='PH 0h (Recommended)'),
        Line2D([0], [0], color=COLORS[48], linestyle='-', 
               linewidth=2, alpha=ALPHAS[48], label='PH 48h'),
        Line2D([0], [0], color=COLORS[72], linestyle='-', 
               linewidth=2, alpha=ALPHAS[72], label='PH 72h'),
    ]
    
    # Add combined legend below the figure
    all_legend_elements = model_legend_elements + ph_legend_elements
    fig.legend(
        handles=all_legend_elements,
        loc='lower center',
        ncol=6,
        fontsize=9,
        framealpha=0.95,
        edgecolor='none',
        bbox_to_anchor=(0.5, -0.02)
    )
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.90, wspace=0.25)
    
    # Title removed for publication
    
    return fig


def create_figure_2_enhanced():
    """
    Create an enhanced version of Figure 2 with improved aesthetics and annotations.
    - Emphasizes the smooth upward gradient of PH 0h models
    - Highlights plateau/fluctuation of PH 48h/72h models
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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    
    datasets = list(DATASET_CONFIGS.keys())
    models = ["Transformer", "LSTM", "CNN"]
    
    # Load all data from CSV files
    print("Loading AUROC data from CSV files...")
    all_data = {}
    
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        print(f"\n  {config['display_name']}:")
        
        all_data[dataset] = load_auroc_from_csv(dataset)
        
        for model in models:
            for ph in PREDICTION_HORIZONS:
                auroc_values = all_data[dataset][model][ph]
                valid_aurocs = [v for v in auroc_values if not np.isnan(v)]
                if valid_aurocs:
                    print(f"    {model} PH={ph}h: mean AUROC={np.mean(valid_aurocs):.3f}")
    
    # Plot each panel
    print("\nGenerating enhanced figure...")
    for col_idx, dataset in enumerate(datasets):
        ax = axes[col_idx]
        dataset_config = DATASET_CONFIGS[dataset]
        
        # Plot lines for each model and prediction horizon
        # Plot PH 48h and 72h first (background), then PH 0h on top (foreground)
        for ph in [72, 48, 0]:  # Reverse order so PH 0h is on top
            for model in models:
                auroc_values = all_data[dataset][model][ph]
                
                # Calculate line width with PH emphasis
                lw = LINE_WIDTHS[model] * PH_LINE_WIDTH_MULT[ph]
                
                # Plot line
                ax.plot(
                    TIME_POINTS,
                    auroc_values,
                    color=COLORS[ph],
                    linestyle=LINE_STYLES[model],
                    linewidth=lw,
                    alpha=ALPHAS[ph],
                    zorder=10 if ph == 0 else 5,  # PH 0h on top
                )
                
                # Add markers at key time points for PH 0h only (most important)
                if ph == 0:
                    key_time_indices = [0, 2, 4, 6]  # 0h, 24h, 48h, 72h
                    for idx in key_time_indices:
                        if idx < len(auroc_values) and not np.isnan(auroc_values[idx]):
                            ax.scatter(
                                TIME_POINTS[idx], auroc_values[idx],
                                color=COLORS[ph], marker=MARKERS[model],
                                s=40, zorder=15, alpha=1.0, edgecolors='white', linewidth=0.8
                            )
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # Styling
        ax.set_xlim([74, -2])  # Slightly extended for visual breathing room
        ax.set_ylim([0.48, 1.02])
        ax.set_xlabel('Hours before AKI onset')
        ax.set_ylabel('AUROC')
        
        # Title with validation type
        title = f"{dataset_config['display_name']}"
        if dataset_config['type'] == "Internal":
            title += " (Internal)"
        else:
            title += " (External)"
        ax.set_title(title)
        
        # Grid
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        
        # X-axis ticks
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
        ax.set_xticklabels(['0', '12', '24', '36', '48', '60', '72'])
        
        # Y-axis ticks
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # Panel label (A, B, C)
        ax.text(
            -0.12, 1.05, dataset_config['panel_label'],
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top'
        )
    
    # Create custom legend with two rows
    # Row 1: Models (line styles) - Transformer first as main model
    model_legend_elements = [
        Line2D([0], [0], color='#444444', linestyle=LINE_STYLES[m], 
               linewidth=LINE_WIDTHS[m], label=m)
        for m in models
    ]
    
    # Row 2: Prediction horizons (colors) - PH 0h emphasized as recommended
    ph_legend_elements = [
        Line2D([0], [0], color=COLORS[0], linestyle='-', 
               linewidth=3.5, alpha=ALPHAS[0], label='PH 0h (Recommended)'),
        Line2D([0], [0], color=COLORS[48], linestyle='-', 
               linewidth=2, alpha=ALPHAS[48], label='PH 48h'),
        Line2D([0], [0], color=COLORS[72], linestyle='-', 
               linewidth=2, alpha=ALPHAS[72], label='PH 72h'),
    ]
    
    # Create two separate legends for clarity
    # Models legend
    model_legend = fig.legend(
        handles=model_legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=9,
        framealpha=0.95,
        edgecolor='none',
        bbox_to_anchor=(0.30, -0.02),
        title='Model Architecture',
        title_fontsize=9,
    )
    
    # PH legend
    ph_legend = fig.legend(
        handles=ph_legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=9,
        framealpha=0.95,
        edgecolor='none',
        bbox_to_anchor=(0.70, -0.02),
        title='Prediction Horizon',
        title_fontsize=9,
    )
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.25)
    
    # Title removed for publication
    
    return fig


def create_figure_2_alternative():
    """
    Alternative version: Model as color, PH as line style.
    This emphasizes model comparison within each prediction horizon.
    """
    # Alternative color scheme: colors for models
    ALT_COLORS = {
        "Transformer": "#1f77b4",  # Blue
        "LSTM": "#ff7f0e",         # Orange
        "CNN": "#2ca02c",          # Green
    }
    
    # Alternative line styles: by prediction horizon
    ALT_LINE_STYLES = {
        0: "-",      # PH 0h: solid
        48: "--",    # PH 48h: dashed
        72: ":",     # PH 72h: dotted
    }
    
    # Set up figure
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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    
    datasets = list(DATASET_CONFIGS.keys())
    models = ["Transformer", "LSTM", "CNN"]
    
    # Load all data
    print("Loading data for alternative figure...")
    all_data = {}
    
    for dataset in datasets:
        all_data[dataset] = load_auroc_from_csv(dataset)
    
    # Plot
    for col_idx, dataset in enumerate(datasets):
        ax = axes[col_idx]
        dataset_config = DATASET_CONFIGS[dataset]
        
        # Plot PH 48h and 72h first (background), then PH 0h on top
        for ph in [72, 48, 0]:
            for model in models:
                auroc_values = all_data[dataset][model][ph]
                
                # PH 0h is emphasized
                alpha = 1.0 if ph == 0 else 0.5
                lw = 2.8 if ph == 0 else 1.8
                
                ax.plot(
                    TIME_POINTS,
                    auroc_values,
                    color=ALT_COLORS[model],
                    linestyle=ALT_LINE_STYLES[ph],
                    linewidth=lw,
                    alpha=alpha,
                    zorder=10 if ph == 0 else 5,
                )
        
        # Styling
        ax.set_xlim([74, -2])
        ax.set_ylim([0.48, 1.02])
        ax.set_xlabel('Hours before AKI onset')
        ax.set_ylabel('AUROC')
        
        title = f"{dataset_config['display_name']}"
        if dataset_config['type'] == "Internal":
            title += " (Internal)"
        else:
            title += " (External)"
        ax.set_title(title)
        
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        ax.text(
            -0.12, 1.05, dataset_config['panel_label'],
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top'
        )
    
    # Legends
    model_legend_elements = [
        Line2D([0], [0], color=ALT_COLORS[m], linestyle='-', 
               linewidth=2.5, label=m)
        for m in models
    ]
    
    ph_legend_elements = [
        Line2D([0], [0], color='#444444', linestyle=ALT_LINE_STYLES[ph], 
               linewidth=2, label=f'PH {ph}h')
        for ph in PREDICTION_HORIZONS
    ]
    
    fig.legend(
        handles=model_legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=9,
        framealpha=0.95,
        edgecolor='none',
        bbox_to_anchor=(0.30, -0.02),
        title='Model',
        title_fontsize=9,
    )
    
    fig.legend(
        handles=ph_legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=9,
        framealpha=0.95,
        edgecolor='none',
        bbox_to_anchor=(0.70, -0.02),
        title='Prediction Horizon',
        title_fontsize=9,
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, top=0.88, wspace=0.25)
    
    # Title removed for publication
    
    return fig


# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 2: Online Simulation Performance")
    print("=" * 80)
    print("\nLine style: Transformer=solid (main), LSTM=dashed, CNN=dotted")
    print("Color: PH 0h=Blue (emphasized), PH 48h=Coral, PH 72h=Gray")
    print("Key message: PH 0h models show better performance (recommended)\n")
    
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
    fig2 = create_figure_2()
    save_figure_all_formats(fig2, f"{OUTPUT_PATH}/figure2_online_simulation")
    print(f"\nFigure 2 (basic) saved to: {OUTPUT_PATH}")
    
    # Enhanced version
    fig2_enhanced = create_figure_2_enhanced()
    save_figure_all_formats(fig2_enhanced, f"{OUTPUT_PATH}/figure2_online_simulation_enhanced")
    print(f"Figure 2 (enhanced) saved to: {OUTPUT_PATH}")
    
    # Alternative version (model as color, PH as line style)
    fig2_alt = create_figure_2_alternative()
    save_figure_all_formats(fig2_alt, f"{OUTPUT_PATH}/figure2_online_simulation_alt")
    print(f"Figure 2 (alternative) saved to: {OUTPUT_PATH}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

# %%
