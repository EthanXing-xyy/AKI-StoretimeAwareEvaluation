# %%
"""
Data loading wrappers for revision experiments.
Loads master tables, predictions, and online simulation data.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from utils.config import (
    PROJECT_ROOT,
    PROCESSED_DIR,
    DATASETS,
    HOSPITALS,
    ALL_MODEL_CONFIGS,
    DL_MODEL_CONFIGS,
    BASELINE_MODEL_CONFIGS,
    PREDICTION_HORIZONS,
    get_master_path,
    get_pred_path,
    get_online_path,
)
from utils.metrics import decode_pred_proba


def load_master(hospital: str) -> pd.DataFrame:
    """Load master_include.parquet for a hospital."""
    path = get_master_path(hospital)
    return pd.read_parquet(path)


def load_all_masters() -> dict[str, pd.DataFrame]:
    """Load master tables for all hospitals."""
    return {h: load_master(h) for h in HOSPITALS}


def load_predictions(
    model_name: str,
    pw: int,
    dataset: str,
) -> pd.DataFrame:
    """Load single-point prediction parquet for a model/pw/dataset combination."""
    config = ALL_MODEL_CONFIGS[model_name]
    path = get_pred_path(config, pw, dataset)
    df = pd.read_parquet(path, engine="fastparquet")

    # Normalize pred_proba column
    if "pred_proba" not in df.columns and "pred_proba_1" in df.columns:
        df = df.rename(columns={"pred_proba_1": "pred_proba"})

    # Decode bytes/logits to probabilities
    if df["pred_proba"].dtype == object or df["pred_proba"].dtype == bytes:
        df["pred_proba"] = decode_pred_proba(df["pred_proba"])
    # Apply sigmoid if values look like logits (outside [0,1])
    if df["pred_proba"].max() > 1.0 or df["pred_proba"].min() < 0.0:
        from utils.metrics import sigmoid
        df["pred_proba"] = df["pred_proba"].apply(sigmoid)

    return df.dropna(subset=["pred_proba"]).reset_index(drop=True)


def load_online_predictions(
    model_name: str,
    pw: int,
    dataset: str,
) -> pd.DataFrame:
    """Load online simulation prediction parquet."""
    config = ALL_MODEL_CONFIGS[model_name]
    path = get_online_path(config, pw, dataset)
    df = (
        pd.read_parquet(path, engine="fastparquet")
        .rename(columns={"pred_proba_1": "pred_proba"})
    )

    # Compute time_to_event (hours to event from current prediction time)
    df["time_to_event"] = np.where(
        df["label"] == 1,
        df["hours_to_aki"],
        df["hours_to_discharge"],
    )

    return df.reset_index(drop=True)


def load_online_with_matched_reference(
    model_name: str,
    pw: int,
    dataset: str,
    matching_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load online predictions and replace non-AKI reference time
    with matched pseudo-onset from matching_df.

    Args:
        matching_df: DataFrame with columns [visit_id, pseudo_onset_hours]
    """
    df = load_online_predictions(model_name, pw, dataset)

    # For non-AKI patients, replace time_to_event with pseudo-onset based reference
    if "pseudo_onset_hours" in matching_df.columns:
        pseudo_map = matching_df.set_index("visit_id")["pseudo_onset_hours"]
        non_aki_mask = df["label"] == 0
        matched_visits = df.loc[non_aki_mask, "visit_id"].map(pseudo_map)

        # time_to_event for non-AKI = pseudo_onset_hours - timestamp (hours elapsed)
        df.loc[non_aki_mask, "time_to_event"] = (
            matched_visits.values - df.loc[non_aki_mask, "timestamp"].values
        )

    return df
