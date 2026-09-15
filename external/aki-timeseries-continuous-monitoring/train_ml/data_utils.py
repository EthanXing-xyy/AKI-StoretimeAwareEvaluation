# %%
"""
Data utilities for ML baseline models (XGBoost, Logistic Regression).

Loads CNN-format .gz files and applies:
1. Last Observation Carry Forward (LOCF) imputation
2. Median imputation for remaining missing values (from scaling_info)
3. IQR normalization: (x - median) / iqr

Output: flat feature vector (48 time-series features + 3 demographics = 51 features)
"""

import joblib
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Optional


# Feature configuration
DEMO_COL = ["age", "sex", "bmi"]
VITAL_COL = ["sbp", "dbp", "pulse", "resp", "spo2", "temp"]
LAB_COL = [
    "albumin", "alp", "alt", "aptt", "ast", "bilirubin",
    "bnp", "bun", "calcium", "chloride", "cholesterol",
    "ck", "creatinine", "crp", "esr", "ggt", "glucose",
    "hba1c", "hco3", "hdl", "hematocrit", "hemoglobin",
    "lactate", "ldh", "magnesium", "pco2", "pdw", "ph",
    "phosphorus", "platelet", "po2", "potassium",
    "procalcitonin", "protein", "pt_inr", "rbc",
    "rdw", "sodium", "tco2", "triglyceride",
    "uric_acid", "wbc",
]
ALL_FEATURES = VITAL_COL + LAB_COL  # 48 time-series features


def load_scaling_info(data_dir: str) -> pd.DataFrame:
    """Load scaling info from ilsan development set."""
    path = f"{data_dir}/scaling_info/ilsan/scaling_info.parquet"
    scaling_info = (
        pd.read_parquet(path)
        .query("feature in @ALL_FEATURES or feature in @DEMO_COL")
    )
    # Reorder: time-series features first (matching ALL_FEATURES order), then demographics
    ts_scale = scaling_info.query("feature in @ALL_FEATURES").set_index("feature").loc[ALL_FEATURES].reset_index()
    demo_scale = scaling_info.query("feature in @DEMO_COL").set_index("feature").loc[DEMO_COL].reset_index()
    return pd.concat([ts_scale, demo_scale], ignore_index=True)


class ScalingParams:
    """Pre-computed scaling parameters to avoid repeated DataFrame queries."""

    def __init__(self, scaling_info: pd.DataFrame):
        ts_scaling = scaling_info.query("feature in @ALL_FEATURES").set_index("feature").loc[ALL_FEATURES]
        self.ts_medians = ts_scaling["median"].values.astype(np.float64)
        self.ts_iqrs = ts_scaling["iqr"].values.astype(np.float64)

        demo_scaling = scaling_info.query("feature in @DEMO_COL").set_index("feature").loc[DEMO_COL]
        self.demo_medians = demo_scaling["median"].values.astype(np.float64)
        self.demo_iqrs = demo_scaling["iqr"].values.astype(np.float64)


def locf_impute(data: np.ndarray) -> np.ndarray:
    """
    Apply Last Observation Carry Forward along time axis (axis=0).
    Uses pandas ffill for vectorized performance.

    Args:
        data: (seq_len, num_features) array with NaN for missing values

    Returns:
        Array with LOCF applied. First-row NaNs remain if no prior observation.
    """
    df = pd.DataFrame(data)
    df = df.ffill(axis=0)
    return df.values


def extract_flat_features(
    data: np.ndarray,
    meta: np.ndarray,
    params: ScalingParams,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert a patient's time-series data to a flat feature vector.

    Process:
    1. LOCF on time-series → take last row
    2. Median imputation for remaining NaN
    3. IQR normalization
    4. Concatenate with normalized demographics

    Args:
        data: Time-series array of shape (seq_len, 48)
        meta: Demographic features [age, sex, bmi]
        params: Pre-computed ScalingParams
        normalize: Whether to apply IQR normalization

    Returns:
        Flat feature vector of shape (51,)
    """
    # LOCF → last row
    filled = locf_impute(data)
    last_row = filled[-1, :].copy()

    # Median imputation for remaining NaNs
    still_nan = np.isnan(last_row)
    if still_nan.any():
        last_row[still_nan] = params.ts_medians[still_nan]

    # Normalize time-series
    if normalize:
        last_row = (last_row - params.ts_medians) / params.ts_iqrs

    # Normalize demographics
    demo = meta.copy()
    if normalize:
        demo = (demo - params.demo_medians) / params.demo_iqrs

    return np.concatenate([last_row, demo])


def extract_flat_features_at_time(
    data: np.ndarray,
    meta: np.ndarray,
    params: ScalingParams,
    time_idx: int,
    normalize: bool = True,
) -> Optional[np.ndarray]:
    """
    Extract flat features using data up to a specific time index.
    Used for online simulation.

    Args:
        data: Full time-series array of shape (seq_len, 48)
        meta: Demographic features [age, sex, bmi]
        params: Pre-computed ScalingParams
        time_idx: Number of time steps to include (1-indexed, so data[:time_idx])
        normalize: Whether to apply IQR normalization

    Returns:
        Flat feature vector of shape (51,) or None if time_idx < 1
    """
    if time_idx < 1:
        return None
    return extract_flat_features(data[:time_idx], meta, params, normalize)


def _process_single_patient(
    args: tuple,
    data_dir: str,
    ts_medians: np.ndarray,
    ts_iqrs: np.ndarray,
    demo_medians: np.ndarray,
    demo_iqrs: np.ndarray,
    normalize: bool,
) -> Optional[tuple]:
    """Process a single patient for multiprocessing. Returns (idx, features, label) or None."""
    idx, visit_id, label, age, sex, bmi = args

    data_path = f"{data_dir}/{visit_id}.gz"
    try:
        data_dict = joblib.load(data_path)
    except FileNotFoundError:
        return None

    data = data_dict["data"]
    meta = np.array([age, sex, bmi], dtype=np.float64)

    # LOCF → last row (inline for speed, avoid extra DataFrame overhead)
    df = pd.DataFrame(data)
    df = df.ffill(axis=0)
    last_row = df.iloc[-1].values.copy()

    # Median imputation
    still_nan = np.isnan(last_row)
    if still_nan.any():
        last_row[still_nan] = ts_medians[still_nan]

    # Normalize
    if normalize:
        last_row = (last_row - ts_medians) / ts_iqrs
        demo = (meta - demo_medians) / demo_iqrs
    else:
        demo = meta

    features = np.concatenate([last_row, demo]).astype(np.float32)
    return (idx, features, label)


def build_dataset(
    meta_table: pd.DataFrame,
    data_dir: str,
    scaling_info: pd.DataFrame,
    normalize: bool = True,
    num_workers: int = 8,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build flat feature matrix from all patients in meta_table.
    Uses multiprocessing for parallel I/O and feature extraction.

    Args:
        meta_table: DataFrame with visit_id, label, age, sex, bmi
        data_dir: Path to timeseries directory containing {visit_id}.gz files
        scaling_info: Scaling info DataFrame
        normalize: Whether to normalize
        num_workers: Number of parallel workers

    Returns:
        X: (n_samples, 51) feature matrix
        y: (n_samples,) label array
        meta_out: DataFrame with metadata for result saving
    """
    meta_table = meta_table.copy().reset_index(drop=True)
    if meta_table["sex"].dtype == object:
        meta_table["sex"] = meta_table["sex"].map({"M": 0, "F": 1})

    params = ScalingParams(scaling_info)

    # Prepare argument tuples
    task_args = [
        (idx, row["visit_id"], row["label"], float(row["age"]), float(row["sex"]), float(row["bmi"]))
        for idx, row in meta_table.iterrows()
    ]

    worker_fn = partial(
        _process_single_patient,
        data_dir=data_dir,
        ts_medians=params.ts_medians,
        ts_iqrs=params.ts_iqrs,
        demo_medians=params.demo_medians,
        demo_iqrs=params.demo_iqrs,
        normalize=normalize,
    )

    # Parallel processing with progress bar
    results_map = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, arg): arg[0] for arg in task_args}
        with tqdm(total=len(futures), desc="Building features") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    idx, features, label = result
                    results_map[idx] = (features, label)
                pbar.update(1)

    # Sort by original index to preserve order
    sorted_indices = sorted(results_map.keys())
    X_list = [results_map[i][0] for i in sorted_indices]
    y_list = [results_map[i][1] for i in sorted_indices]

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    meta_out = meta_table.iloc[sorted_indices].reset_index(drop=True)

    return X, y, meta_out
