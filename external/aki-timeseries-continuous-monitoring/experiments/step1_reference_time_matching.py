# %%
"""
Step 1: Non-AKI Reference Time Ratio-Based Matching

Replaces the random reference time (discharge time) for non-AKI patients with
a pseudo-onset time that follows the same temporal distribution as AKI onset ratios.

Method:
1. Compute onset_ratio = (admission_to_AKI) / (total_LOS) for each AKI patient
2. Sample from this distribution for each non-AKI patient
3. pseudo_onset = admission + (sampled_ratio × non-AKI_LOS)

Output:
- Matching CSV per hospital
- QC histograms comparing AKI onset ratio vs non-AKI pseudo-onset ratio
- KS test results
"""

import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils.config import (
    SEED,
    HOSPITALS,
    RESULTS_DIR,
    FIGURES_DIR,
    get_master_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Output directories
RESULTS_OUT = RESULTS_DIR / "reference_time"
FIGURES_OUT = FIGURES_DIR / "reference_time"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUT.mkdir(parents=True, exist_ok=True)


# %%
def compute_onset_ratios(master: pd.DataFrame) -> pd.Series:
    """
    Compute onset_ratio = loo / los for AKI patients.
    loo = hours from admission to AKI onset
    los = total length of stay in hours
    """
    aki = master.query("label == 1").copy()
    # onset_ratio: fraction of LOS at which AKI occurs
    ratios = aki["loo"] / aki["los"]
    # Clip to valid range (onset should be before discharge)
    ratios = ratios.clip(0.0, 1.0)
    return ratios.dropna()


def match_reference_times(
    master: pd.DataFrame,
    hospital: str,
    seed: int = SEED,
    min_obs_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Assign pseudo-onset times to non-AKI patients based on AKI onset ratio distribution.

    Args:
        master: Master table with columns [visit_id, label, los, loo]
        hospital: Hospital name for logging
        seed: Random seed for reproducibility
        min_obs_hours: Minimum observation time (pseudo-onset must be >= this)

    Returns:
        DataFrame with matched pseudo-onset information for non-AKI patients
    """
    rng = np.random.RandomState(seed)

    # Compute AKI onset ratio distribution
    onset_ratios = compute_onset_ratios(master)
    logger.info(f"[{hospital}] AKI onset ratios: n={len(onset_ratios)}, "
                f"mean={onset_ratios.mean():.3f}, median={onset_ratios.median():.3f}")

    # Get non-AKI patients
    non_aki = master.query("label == 0").copy()
    n_non_aki = len(non_aki)
    logger.info(f"[{hospital}] Non-AKI patients: n={n_non_aki}")

    # Sample from AKI onset ratio distribution (with replacement)
    sampled_ratios = rng.choice(onset_ratios.values, size=n_non_aki, replace=True)

    # Compute pseudo-onset hours
    pseudo_onset_hours = sampled_ratios * non_aki["los"].values

    # Clip: ensure pseudo_onset >= min_obs_hours and <= los
    pseudo_onset_hours = np.clip(pseudo_onset_hours, min_obs_hours, non_aki["los"].values)

    # Recompute effective ratios after clipping
    effective_ratios = pseudo_onset_hours / non_aki["los"].values

    result = pd.DataFrame({
        "visit_id": non_aki["visit_id"].values,
        "dataset": hospital,
        "label": 0,
        "los_hours": non_aki["los"].values,
        "sampled_ratio": sampled_ratios,
        "effective_ratio": effective_ratios,
        "pseudo_onset_hours": pseudo_onset_hours,
    })

    logger.info(f"[{hospital}] Pseudo-onset hours: mean={pseudo_onset_hours.mean():.1f}, "
                f"median={np.median(pseudo_onset_hours):.1f}")

    return result


# %%
def create_qc_histogram(
    aki_ratios: np.ndarray,
    non_aki_ratios: np.ndarray,
    hospital: str,
    ks_stat: float,
    ks_pval: float,
    save_path: Path,
):
    """Create QC histogram comparing AKI onset ratios vs non-AKI pseudo-onset ratios."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.linspace(0, 1, 31)

    # Overlaid histogram
    ax = axes[0]
    ax.hist(aki_ratios, bins=bins, alpha=0.6, label=f"AKI onset ratio (n={len(aki_ratios)})",
            color="#2196F3", density=True, edgecolor="white", linewidth=0.5)
    ax.hist(non_aki_ratios, bins=bins, alpha=0.6, label=f"Non-AKI pseudo-onset ratio (n={len(non_aki_ratios)})",
            color="#FF9800", density=True, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Onset Ratio (time to event / LOS)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{hospital.upper()} - Distribution Comparison", fontsize=12)
    ax.legend(fontsize=9)
    ax.text(0.05, 0.95, f"KS stat = {ks_stat:.4f}\np = {ks_pval:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # CDF comparison
    ax = axes[1]
    sorted_aki = np.sort(aki_ratios)
    sorted_non_aki = np.sort(non_aki_ratios)
    ax.plot(sorted_aki, np.arange(1, len(sorted_aki) + 1) / len(sorted_aki),
            label="AKI onset ratio", color="#2196F3", linewidth=2)
    ax.plot(sorted_non_aki, np.arange(1, len(sorted_non_aki) + 1) / len(sorted_non_aki),
            label="Non-AKI pseudo-onset ratio", color="#FF9800", linewidth=2, linestyle="--")
    ax.set_xlabel("Onset Ratio", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_title(f"{hospital.upper()} - CDF Comparison", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved QC histogram: {save_path}")


# %%
def run_matching_for_hospital(hospital: str, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Run reference time matching for a single hospital."""
    master = pd.read_parquet(get_master_path(hospital))

    # Filter to included patients
    if "inclusion_yn" in master.columns:
        master = master.query("inclusion_yn == 1")
    if "vital_yn" in master.columns:
        master = master.query("vital_yn == 1")

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing hospital: {hospital}")
    logger.info(f"Total patients: {len(master)}, AKI: {master['label'].sum()}, "
                f"Non-AKI: {(master['label'] == 0).sum()}")

    # Perform matching
    matching_result = match_reference_times(master, hospital, seed)

    # Compute onset ratios for QC
    aki_ratios = compute_onset_ratios(master).values
    non_aki_ratios = matching_result["effective_ratio"].values

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(aki_ratios, non_aki_ratios)
    logger.info(f"[{hospital}] KS test: stat={ks_stat:.4f}, p={ks_pval:.4f}")

    # Create QC histogram
    fig_path = FIGURES_OUT / f"qc_histogram_{hospital}.pdf"
    create_qc_histogram(aki_ratios, non_aki_ratios, hospital, ks_stat, ks_pval, fig_path)

    ks_result = {
        "hospital": hospital,
        "n_aki": len(aki_ratios),
        "n_non_aki": len(non_aki_ratios),
        "aki_ratio_mean": float(aki_ratios.mean()),
        "aki_ratio_median": float(np.median(aki_ratios)),
        "non_aki_ratio_mean": float(non_aki_ratios.mean()),
        "non_aki_ratio_median": float(np.median(non_aki_ratios)),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
    }

    return matching_result, ks_result


# %%
def create_combined_figure(all_results: dict[str, pd.DataFrame], all_ks: list[dict]):
    """Create a combined QC figure for all hospitals."""
    fig, axes = plt.subplots(1, len(HOSPITALS), figsize=(5 * len(HOSPITALS), 5))
    if len(HOSPITALS) == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 31)

    for idx, hospital in enumerate(HOSPITALS):
        ax = axes[idx]
        master = pd.read_parquet(get_master_path(hospital))
        aki_ratios = compute_onset_ratios(master).values

        matching = all_results[hospital]
        non_aki_ratios = matching["effective_ratio"].values

        ks_info = [k for k in all_ks if k["hospital"] == hospital][0]

        ax.hist(aki_ratios, bins=bins, alpha=0.6, label=f"AKI (n={len(aki_ratios)})",
                color="#2196F3", density=True, edgecolor="white", linewidth=0.5)
        ax.hist(non_aki_ratios, bins=bins, alpha=0.6, label=f"Non-AKI (n={len(non_aki_ratios)})",
                color="#FF9800", density=True, edgecolor="white", linewidth=0.5)

        display_names = {"ilsan": "NHIS", "cchlmc": "CSHH", "mimic-iv": "MIMIC-IV"}
        ax.set_title(display_names.get(hospital, hospital), fontsize=13, fontweight="bold")
        ax.set_xlabel("Onset Ratio", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9)
        ax.text(0.05, 0.95, f"KS p={ks_info['ks_pvalue']:.3f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    save_path = FIGURES_OUT / "qc_histogram_combined.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined QC figure: {save_path}")


# %%
def main():
    logger.info("=" * 70)
    logger.info("Step 1: Non-AKI Reference Time Ratio-Based Matching")
    logger.info("=" * 70)

    all_results = {}
    all_ks = []

    for hospital in HOSPITALS:
        matching_result, ks_result = run_matching_for_hospital(hospital)
        all_results[hospital] = matching_result
        all_ks.append(ks_result)

        # Save per-hospital CSV
        csv_path = RESULTS_OUT / f"reference_time_matching_{hospital}.csv"
        matching_result.to_csv(csv_path, index=False)
        logger.info(f"Saved matching result: {csv_path}")

    # Combine all hospitals
    combined = pd.concat(all_results.values(), ignore_index=True)
    combined_path = RESULTS_OUT / "reference_time_matching_all.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"Saved combined matching result: {combined_path}")

    # KS test summary
    ks_df = pd.DataFrame(all_ks)
    ks_path = RESULTS_OUT / "ks_test_summary.csv"
    ks_df.to_csv(ks_path, index=False)
    logger.info(f"Saved KS test summary: {ks_path}")

    # Combined QC figure
    create_combined_figure(all_results, all_ks)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    print(ks_df.to_string(index=False))

    return all_results, ks_df


# %%
if __name__ == "__main__":
    all_results, ks_df = main()
