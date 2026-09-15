# %%
"""
Online Inference Module for Transformer Model (Optimized with Batching & DataLoader)

This module simulates real-time online prediction where predictions are generated
at each unique timestamp as new data arrives. For each patient visit, it iterates
through timestamps and makes predictions using all available data up to that point.

Key Features:
- Generates prediction scores at each timestamp (online/streaming prediction)
- Supports cumulative data aggregation (all data up to current timestamp)
- Outputs time-series of prediction probabilities per patient
- OPTIMIZED: Batches all timestamps for a patient into a single forward pass
- OPTIMIZED: Supports multiprocessing for parallel patient processing
- OPTIMIZED: PyTorch DataLoader with num_workers for parallel data loading
"""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.transforms.collate_functions import collate_fn_transformer


# =============================================================================
# DataLoader-based Implementation
# =============================================================================

class OnlineInferenceDataset(Dataset):
    """
    Dataset for online inference that loads patient data and prepares
    all timestamp samples for a patient.
    
    Each item returns all cumulative data tensors for a single patient,
    along with metadata (visit_id, label, timestamps, observation counts).
    """
    
    def __init__(
        self,
        master_table: pd.DataFrame,
        data_dir: str,
        time_resolution: float = 1.0,
        min_observations: int = 1,
    ):
        """
        Args:
            master_table: DataFrame with 'visit_id' and 'label' columns
            data_dir: Directory containing patient data files (.gz)
            time_resolution: Time resolution for grouping timestamps (in hours)
            min_observations: Minimum observations required for prediction
        """
        self.master_table = master_table.reset_index(drop=True)
        self.data_dir = data_dir
        self.time_resolution = time_resolution
        self.min_observations = min_observations
    
    def __len__(self) -> int:
        return len(self.master_table)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load patient data and prepare all timestamp samples.
        
        Returns:
            Dictionary with:
            - 'data_tensors': List of tensors, each shape (seq_len, 3)
            - 'visit_id': Patient visit ID
            - 'label': Ground truth label
            - 'timestamps': List of timestamps
            - 'n_observations': List of observation counts
        """
        row = self.master_table.iloc[idx]
        visit_id = row["visit_id"]
        label = row["label"]
        
        data_path = f"{self.data_dir}/{visit_id}.gz"
        
        try:
            data_dict = joblib.load(data_path)
        except FileNotFoundError:
            # Return empty result for missing files
            return {
                'data_tensors': [],
                'visit_id': visit_id,
                'label': label,
                'timestamps': [],
                'n_observations': [],
            }
        
        data = data_dict["data"]  # Shape: (seq_len, 3)
        timestamps = data[:, 1]
        
        # Get unique timestamps
        if self.time_resolution > 0:
            rounded_timestamps = np.round(timestamps / self.time_resolution) * self.time_resolution
            unique_timestamps = np.unique(rounded_timestamps)
        else:
            unique_timestamps = np.unique(timestamps)
        
        # Prepare cumulative data for each timestamp
        data_tensors = []
        valid_timestamps = []
        observation_counts = []
        
        for current_time in unique_timestamps:
            if self.time_resolution > 0:
                time_mask = np.round(timestamps / self.time_resolution) * self.time_resolution <= current_time
            else:
                time_mask = timestamps <= current_time
            
            cumulative_data = data[time_mask]
            
            if len(cumulative_data) < self.min_observations:
                continue
            
            data_tensor = torch.from_numpy(cumulative_data).float()
            data_tensors.append(data_tensor)
            valid_timestamps.append(current_time)
            observation_counts.append(len(cumulative_data))
        
        return {
            'data_tensors': data_tensors,
            'visit_id': visit_id,
            'label': label,
            'timestamps': valid_timestamps,
            'n_observations': observation_counts,
        }


def online_inference_collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Custom collate function that flattens samples from multiple patients.
    Does NOT apply the transformer collate function here - that's done in chunks
    during inference to avoid OOM.
    
    Args:
        batch: List of dictionaries from OnlineInferenceDataset
        
    Returns:
        Dictionary with raw data tensors and metadata (NOT collated yet)
    """
    # Flatten all data tensors from all patients
    all_data_tensors = []
    all_metadata = []  # (visit_id, label, timestamp, n_observations)
    
    for item in batch:
        for i, data_tensor in enumerate(item['data_tensors']):
            all_data_tensors.append(data_tensor)
            all_metadata.append((
                item['visit_id'],
                item['label'],
                item['timestamps'][i],
                item['n_observations'][i],
            ))
    
    return {
        'data_tensors': all_data_tensors,  # List of tensors (not collated)
        'metadata': all_metadata,
    }


def run_online_inference_dataloader(
    lightning_module: torch.nn.Module,
    master_table: pd.DataFrame,
    data_dir: str,
    scaling_info: Optional[pd.DataFrame] = None,
    device: str = "cuda",
    time_resolution: float = 1.0,
    min_observations: int = 1,
    show_progress: bool = True,
    batch_size: int = 16,
    gpu_batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run online inference using PyTorch DataLoader for parallel data loading.
    
    This is the most optimized version:
    - Uses DataLoader with num_workers for parallel I/O
    - Prefetches data while GPU is processing
    - Chunks GPU inference to avoid OOM
    
    Args:
        lightning_module: Trained PyTorch Lightning module
        master_table: DataFrame with 'visit_id' and 'label' columns
        data_dir: Directory containing patient data files (.gz)
        scaling_info: DataFrame with normalization parameters
        device: Device to run inference on
        time_resolution: Time resolution for grouping timestamps (in hours)
        min_observations: Minimum observations required for prediction
        show_progress: Whether to show progress bar
        batch_size: Number of patients to load per DataLoader batch
        gpu_batch_size: Max samples per GPU forward pass (to avoid OOM)
        num_workers: Number of parallel workers for data loading
        prefetch_factor: Number of batches to prefetch per worker
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Dictionary mapping visit_id to DataFrame of online predictions
    """
    device_obj = torch.device(device if torch.cuda.is_available() or device != 'cuda' else 'cpu')
    lightning_module = lightning_module.to(device_obj)
    lightning_module.eval()
    
    # Create collate function for inference (applies scaling)
    if scaling_info is not None:
        inference_collate_fn = partial(collate_fn_transformer, scaling_info=scaling_info)
    else:
        inference_collate_fn = collate_fn_transformer
    
    # Create dataset
    dataset = OnlineInferenceDataset(
        master_table=master_table,
        data_dir=data_dir,
        time_resolution=time_resolution,
        min_observations=min_observations,
    )
    
    # Create DataLoader (collate_fn does NOT apply transformer collation - just flattens)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=online_inference_collate_fn,
        pin_memory=False,  # We'll handle GPU transfer in chunks
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    results = {}
    
    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Online Inference (DataLoader)")
    
    with torch.no_grad():
        for batch in iterator:
            data_tensors = batch['data_tensors']
            metadata = batch['metadata']
            
            if len(data_tensors) == 0:
                continue
            
            # Process in GPU-friendly chunks to avoid OOM
            all_probs = []
            for chunk_start in range(0, len(data_tensors), gpu_batch_size):
                chunk_end = min(chunk_start + gpu_batch_size, len(data_tensors))
                chunk_tensors = data_tensors[chunk_start:chunk_end]
                
                # Apply collate function to this chunk
                dummy_labels = [torch.LongTensor([0]) for _ in chunk_tensors]
                batch_items = list(zip(chunk_tensors, dummy_labels))
                collated = inference_collate_fn(batch_items)
                
                collated_data = collated['data'].to(device_obj)
                mask = collated['mask'].to(device_obj) if collated['mask'] is not None else None
                
                # Forward pass
                logits = lightning_module.backbone_model(collated_data, mask)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                
                # Free GPU memory
                del collated_data, mask, logits
            
            # Concatenate all chunk results
            all_probs = np.concatenate(all_probs, axis=0)
            
            # Organize results by visit_id
            for idx, (visit_id, label, timestamp, n_obs) in enumerate(metadata):
                result = {
                    "timestamp": timestamp,
                    "n_observations": n_obs,
                    "visit_id": visit_id,
                    "label": label,
                }
                for i, p in enumerate(all_probs[idx]):
                    result[f"pred_proba_{i}"] = p
                
                if visit_id not in results:
                    results[visit_id] = []
                results[visit_id].append(result)
    
    # Convert lists to DataFrames
    for visit_id in results:
        results[visit_id] = pd.DataFrame(results[visit_id])
    
    return results


# =============================================================================
# Original Implementation (kept for compatibility)
# =============================================================================


class OnlinePredictor:
    """
    Online predictor that generates predictions at each timestamp.
    
    This class simulates real-time prediction scenarios where the model
    makes predictions each time new data arrives (at each unique timestamp).
    
    OPTIMIZED: Uses batched inference for all timestamps at once.
    """
    
    def __init__(
        self,
        lightning_module: torch.nn.Module,
        scaling_info: Optional[pd.DataFrame] = None,
        device: str = "cuda",
        batch_size: int = 64,
    ):
        """
        Initialize the online predictor.
        
        Args:
            lightning_module: Trained PyTorch Lightning module
            scaling_info: DataFrame with normalization parameters (feature_id, median, iqr)
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
            batch_size: Batch size for inference (number of timestamps to process at once)
        """
        self.device = torch.device(device if torch.cuda.is_available() or device != 'cuda' else 'cpu')
        self.lightning_module = lightning_module.to(self.device)
        self.lightning_module.eval()
        self.scaling_info = scaling_info
        self.batch_size = batch_size
        
        # Create collate function (same as used in training)
        if scaling_info is not None:
            self.collate_fn = partial(collate_fn_transformer, scaling_info=scaling_info)
        else:
            self.collate_fn = collate_fn_transformer
        
    @torch.no_grad()
    def predict_single_timestamp(
        self,
        data: torch.Tensor,
    ) -> np.ndarray:
        """
        Make a single prediction with given data.
        
        Args:
            data: Input tensor of shape (seq_len, 3) with [feature_id, time, value]
            
        Returns:
            Prediction probabilities as numpy array of shape (num_classes,)
        """
        # Create a dummy label for collate function
        dummy_label = torch.LongTensor([0])
        
        # Apply collate function (handles normalization, padding, masking)
        batch = self.collate_fn([(data, dummy_label)])
        
        collated_data = batch["data"].to(self.device)
        mask = batch["mask"].to(self.device) if batch["mask"] is not None else None
        
        # Forward pass
        logits = self.lightning_module.backbone_model(collated_data, mask)
        probs = F.softmax(logits, dim=-1)
        
        return probs.cpu().numpy()[0]
    
    @torch.no_grad()
    def predict_batch(
        self,
        data_list: List[torch.Tensor],
    ) -> np.ndarray:
        """
        Make predictions for a batch of data tensors.
        
        Args:
            data_list: List of input tensors, each of shape (seq_len, 3)
            
        Returns:
            Prediction probabilities as numpy array of shape (batch_size, num_classes)
        """
        if len(data_list) == 0:
            return np.array([])
        
        # Create dummy labels for collate function
        dummy_labels = [torch.LongTensor([0]) for _ in data_list]
        
        # Apply collate function (handles normalization, padding, masking for batch)
        batch_items = list(zip(data_list, dummy_labels))
        batch = self.collate_fn(batch_items)
        
        collated_data = batch["data"].to(self.device)
        mask = batch["mask"].to(self.device) if batch["mask"] is not None else None
        
        # Forward pass
        logits = self.lightning_module.backbone_model(collated_data, mask)
        probs = F.softmax(logits, dim=-1)
        
        return probs.cpu().numpy()
    
    def predict_online(
        self,
        data_dict: Dict[str, np.ndarray],
        time_resolution: float = 1.0,
        min_observations: int = 1,
    ) -> pd.DataFrame:
        """
        Perform online prediction for a single patient visit.
        
        Iterates through unique timestamps and makes predictions using
        all data available up to that timestamp.
        
        OPTIMIZED: Collects all cumulative data and performs batched inference.
        
        Args:
            data_dict: Dictionary containing 'data' key with numpy array of shape (seq_len, 3)
                       where columns are [feature_id, time, value]
            time_resolution: Time resolution for grouping timestamps (in hours)
            min_observations: Minimum number of observations required to make prediction
            
        Returns:
            DataFrame with columns: [timestamp, pred_proba_0, pred_proba_1, ...]
        """
        data = data_dict["data"]  # Shape: (seq_len, 3)
        
        # Extract timestamps (column 1)
        timestamps = data[:, 1]
        
        # Get unique timestamps sorted
        unique_timestamps = np.unique(timestamps)
        
        # Group timestamps by resolution if specified
        if time_resolution > 0:
            # Round timestamps to resolution
            rounded_timestamps = np.round(timestamps / time_resolution) * time_resolution
            unique_timestamps = np.unique(rounded_timestamps)
        
        # Prepare all cumulative data tensors
        data_tensors = []
        valid_timestamps = []
        observation_counts = []
        
        for current_time in unique_timestamps:
            # Get mask for observations up to current timestamp
            if time_resolution > 0:
                time_mask = np.round(timestamps / time_resolution) * time_resolution <= current_time
            else:
                time_mask = timestamps <= current_time
            
            # Get cumulative data up to current timestamp
            cumulative_data = data[time_mask]
            
            # Skip if not enough observations
            if len(cumulative_data) < min_observations:
                continue
            
            # Convert to tensor
            data_tensor = torch.from_numpy(cumulative_data).float()
            data_tensors.append(data_tensor)
            valid_timestamps.append(current_time)
            observation_counts.append(len(cumulative_data))
        
        if len(data_tensors) == 0:
            return pd.DataFrame()
        
        # Batch inference - process in chunks of batch_size
        all_probs = []
        for i in range(0, len(data_tensors), self.batch_size):
            batch_tensors = data_tensors[i:i + self.batch_size]
            batch_probs = self.predict_batch(batch_tensors)
            all_probs.append(batch_probs)
        
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Build results
        results = []
        for idx, (timestamp, n_obs) in enumerate(zip(valid_timestamps, observation_counts)):
            result = {
                "timestamp": timestamp,
                "n_observations": n_obs,
            }
            for i, p in enumerate(all_probs[idx]):
                result[f"pred_proba_{i}"] = p
            results.append(result)
        
        return pd.DataFrame(results)


def _load_and_predict_single_patient(
    row_data: Tuple[str, int, str],
    predictor: OnlinePredictor,
    time_resolution: float,
    min_observations: int,
) -> Tuple[str, pd.DataFrame]:
    """
    Load data and run prediction for a single patient.
    Helper function for parallel processing.
    """
    visit_id, label, data_path = row_data
    
    try:
        data_dict = joblib.load(data_path)
    except FileNotFoundError:
        return visit_id, None
    
    pred_df = predictor.predict_online(
        data_dict=data_dict,
        time_resolution=time_resolution,
        min_observations=min_observations,
    )
    
    if len(pred_df) > 0:
        pred_df["visit_id"] = visit_id
        pred_df["label"] = label
    
    return visit_id, pred_df


def run_online_inference(
    lightning_module: torch.nn.Module,
    master_table: pd.DataFrame,
    data_dir: str,
    scaling_info: Optional[pd.DataFrame] = None,
    device: str = "cuda",
    time_resolution: float = 1.0,
    min_observations: int = 1,
    show_progress: bool = True,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Run online inference for all patients in the master table.
    
    OPTIMIZED: 
    - Uses batched inference for timestamps within each patient
    - Optional multiprocessing for loading data in parallel
    
    Args:
        lightning_module: Trained PyTorch Lightning module
        master_table: DataFrame with 'visit_id' and 'label' columns
        data_dir: Directory containing patient data files (.gz)
        scaling_info: DataFrame with normalization parameters
        device: Device to run inference on
        time_resolution: Time resolution for grouping timestamps (in hours)
        min_observations: Minimum observations required for prediction
        show_progress: Whether to show progress bar
        batch_size: Number of timestamps to process at once per patient
        num_workers: Number of parallel workers for data loading (0 = sequential)
        
    Returns:
        Dictionary mapping visit_id to DataFrame of online predictions
    """
    predictor = OnlinePredictor(
        lightning_module=lightning_module,
        scaling_info=scaling_info,
        device=device,
        batch_size=batch_size,
    )
    
    results = {}
    
    if num_workers > 0:
        # Parallel data loading with ThreadPoolExecutor
        # (ProcessPoolExecutor doesn't work well with CUDA tensors)
        row_data_list = [
            (row["visit_id"], row["label"], f"{data_dir}/{row['visit_id']}.gz")
            for _, row in master_table.iterrows()
        ]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_visit = {
                executor.submit(
                    _load_and_predict_single_patient,
                    row_data,
                    predictor,
                    time_resolution,
                    min_observations,
                ): row_data[0]
                for row_data in row_data_list
            }
            
            iterator = future_to_visit.items()
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(future_to_visit), desc="Online Inference")
            
            for future, visit_id in iterator:
                _, pred_df = future.result()
                if pred_df is not None and len(pred_df) > 0:
                    results[visit_id] = pred_df
    else:
        # Sequential processing
        iterator = master_table.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=len(master_table), desc="Online Inference")
        
        for _, row in iterator:
            visit_id = row["visit_id"]
            label = row["label"]
            
            # Load patient data
            data_path = f"{data_dir}/{visit_id}.gz"
            try:
                data_dict = joblib.load(data_path)
            except FileNotFoundError:
                print(f"Warning: Data file not found for visit_id {visit_id}")
                continue
            
            # Run online prediction
            pred_df = predictor.predict_online(
                data_dict=data_dict,
                time_resolution=time_resolution,
                min_observations=min_observations,
            )
            
            # Add metadata
            pred_df["visit_id"] = visit_id
            pred_df["label"] = label
            
            results[visit_id] = pred_df
    
    return results


def run_online_inference_batched_patients(
    lightning_module: torch.nn.Module,
    master_table: pd.DataFrame,
    data_dir: str,
    scaling_info: Optional[pd.DataFrame] = None,
    device: str = "cuda",
    time_resolution: float = 1.0,
    min_observations: int = 1,
    show_progress: bool = True,
    batch_size: int = 128,
    patient_batch_size: int = 16,
) -> Dict[str, pd.DataFrame]:
    """
    Run online inference with batching across BOTH timestamps AND patients.
    
    This is the most optimized version - it batches predictions from multiple
    patients together into larger GPU batches.
    
    Args:
        lightning_module: Trained PyTorch Lightning module
        master_table: DataFrame with 'visit_id' and 'label' columns
        data_dir: Directory containing patient data files (.gz)
        scaling_info: DataFrame with normalization parameters
        device: Device to run inference on
        time_resolution: Time resolution for grouping timestamps (in hours)
        min_observations: Minimum observations required for prediction
        show_progress: Whether to show progress bar
        batch_size: Max number of samples to process in one GPU forward pass
        patient_batch_size: Number of patients to load data for at once
        
    Returns:
        Dictionary mapping visit_id to DataFrame of online predictions
    """
    device_obj = torch.device(device if torch.cuda.is_available() or device != 'cuda' else 'cpu')
    lightning_module = lightning_module.to(device_obj)
    lightning_module.eval()
    
    if scaling_info is not None:
        collate_fn = partial(collate_fn_transformer, scaling_info=scaling_info)
    else:
        collate_fn = collate_fn_transformer
    
    results = {}
    
    # Process patients in batches
    total_patients = len(master_table)
    
    iterator = range(0, total_patients, patient_batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(total_patients + patient_batch_size - 1) // patient_batch_size, 
                       desc="Online Inference (Patient Batches)")
    
    for batch_start in iterator:
        batch_end = min(batch_start + patient_batch_size, total_patients)
        patient_batch = master_table.iloc[batch_start:batch_end]
        
        # Collect all data tensors and metadata for this patient batch
        all_data_tensors = []
        all_metadata = []  # (visit_id, label, timestamp, n_observations)
        
        for _, row in patient_batch.iterrows():
            visit_id = row["visit_id"]
            label = row["label"]
            
            data_path = f"{data_dir}/{visit_id}.gz"
            try:
                data_dict = joblib.load(data_path)
            except FileNotFoundError:
                continue
            
            data = data_dict["data"]
            timestamps = data[:, 1]
            
            if time_resolution > 0:
                rounded_timestamps = np.round(timestamps / time_resolution) * time_resolution
                unique_timestamps = np.unique(rounded_timestamps)
            else:
                unique_timestamps = np.unique(timestamps)
            
            for current_time in unique_timestamps:
                if time_resolution > 0:
                    time_mask = np.round(timestamps / time_resolution) * time_resolution <= current_time
                else:
                    time_mask = timestamps <= current_time
                
                cumulative_data = data[time_mask]
                
                if len(cumulative_data) < min_observations:
                    continue
                
                data_tensor = torch.from_numpy(cumulative_data).float()
                all_data_tensors.append(data_tensor)
                all_metadata.append((visit_id, label, current_time, len(cumulative_data)))
        
        if len(all_data_tensors) == 0:
            continue
        
        # Process all tensors in GPU batches
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(all_data_tensors), batch_size):
                batch_tensors = all_data_tensors[i:i + batch_size]
                dummy_labels = [torch.LongTensor([0]) for _ in batch_tensors]
                batch_items = list(zip(batch_tensors, dummy_labels))
                batch = collate_fn(batch_items)
                
                collated_data = batch["data"].to(device_obj)
                mask = batch["mask"].to(device_obj) if batch["mask"] is not None else None
                
                logits = lightning_module.backbone_model(collated_data, mask)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Organize results by visit_id
        for idx, (visit_id, label, timestamp, n_obs) in enumerate(all_metadata):
            result = {
                "timestamp": timestamp,
                "n_observations": n_obs,
                "visit_id": visit_id,
                "label": label,
            }
            for i, p in enumerate(all_probs[idx]):
                result[f"pred_proba_{i}"] = p
            
            if visit_id not in results:
                results[visit_id] = []
            results[visit_id].append(result)
    
    # Convert lists to DataFrames
    for visit_id in results:
        results[visit_id] = pd.DataFrame(results[visit_id])
    
    return results


def aggregate_online_results(
    results: Dict[str, pd.DataFrame],
    master_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aggregate online prediction results into a single DataFrame.
    
    Args:
        results: Dictionary mapping visit_id to DataFrame of predictions
        master_table: Optional DataFrame with patient metadata including
                     admission_dt, discharge_dt, aki_dt, los, loo columns
                     for calculating time differences
        
    Returns:
        Concatenated DataFrame with all predictions and time difference columns
    """
    if not results:
        return pd.DataFrame()
    
    all_results = pd.concat(results.values(), ignore_index=True)
    
    # If master_table is provided, calculate time differences
    if master_table is not None:
        # Merge with master table to get datetime columns
        datetime_cols = ["visit_id"]
        optional_cols = ["admission_dt", "discharge_dt", "aki_dt", "los", "loo"]
        for col in optional_cols:
            if col in master_table.columns:
                datetime_cols.append(col)
        
        master_subset = master_table[datetime_cols].drop_duplicates(subset=["visit_id"])
        all_results = all_results.merge(master_subset, on="visit_id", how="left")
        
        # Calculate prediction datetime (admission_dt + timestamp hours)
        if "admission_dt" in all_results.columns:
            all_results["admission_dt"] = pd.to_datetime(all_results["admission_dt"])
            all_results["prediction_dt"] = all_results["admission_dt"] + pd.to_timedelta(all_results["timestamp"], unit="h")
        
        # Calculate time difference: prediction_dt to aki_dt (in hours)
        # Positive value means AKI happens after prediction (prediction leads AKI)
        # Negative value means AKI happened before prediction
        if "aki_dt" in all_results.columns:
            all_results["aki_dt"] = pd.to_datetime(all_results["aki_dt"])
            all_results["hours_to_aki"] = (all_results["aki_dt"] - all_results["prediction_dt"]).dt.total_seconds() / 3600
            # Alternative: using loo (length of outcome) - timestamp
            if "loo" in all_results.columns:
                all_results["hours_to_aki_from_loo"] = all_results["loo"] - all_results["timestamp"]
        
        # Calculate time difference: prediction_dt to discharge_dt (in hours)
        # Positive value means discharge happens after prediction
        if "discharge_dt" in all_results.columns:
            all_results["discharge_dt"] = pd.to_datetime(all_results["discharge_dt"])
            all_results["hours_to_discharge"] = (all_results["discharge_dt"] - all_results["prediction_dt"]).dt.total_seconds() / 3600
            # Alternative: using los (length of stay) - timestamp
            if "los" in all_results.columns:
                all_results["hours_to_discharge_from_los"] = all_results["los"] - all_results["timestamp"]
    
    # Reorder columns
    base_cols = ["visit_id", "label", "timestamp", "n_observations"]
    datetime_info_cols = ["admission_dt", "prediction_dt", "aki_dt", "discharge_dt"]
    time_diff_cols = ["hours_to_aki", "hours_to_aki_from_loo", "hours_to_discharge", "hours_to_discharge_from_los", "los", "loo"]
    pred_cols = [c for c in all_results.columns if c.startswith("pred_proba_")]
    
    # Build final column order
    cols = []
    for col in base_cols:
        if col in all_results.columns:
            cols.append(col)
    for col in datetime_info_cols:
        if col in all_results.columns:
            cols.append(col)
    for col in time_diff_cols:
        if col in all_results.columns:
            cols.append(col)
    cols.extend(sorted(pred_cols))
    
    return all_results[cols]


def compute_online_metrics(
    results_df: pd.DataFrame,
    positive_class: int = 1,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compute metrics at different time points for online predictions.
    
    Args:
        results_df: DataFrame with online prediction results
        positive_class: Index of positive class for binary metrics
        thresholds: List of timestamp thresholds to evaluate at
        
    Returns:
        DataFrame with metrics at different time points
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    
    if thresholds is None:
        # Default: evaluate at quartiles of timestamps
        thresholds = results_df["timestamp"].quantile([0.25, 0.5, 0.75, 1.0]).values
    
    metrics_list = []
    
    for threshold in thresholds:
        # Get last prediction before threshold for each patient
        df_filtered = results_df[results_df["timestamp"] <= threshold]
        
        if len(df_filtered) == 0:
            continue
            
        # Get last prediction per patient
        last_preds = df_filtered.groupby("visit_id").last().reset_index()
        
        if len(last_preds) < 2:
            continue
        
        y_true = last_preds["label"].values
        y_prob = last_preds[f"pred_proba_{positive_class}"].values
        
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            continue
        
        metrics = {
            "timestamp_threshold": threshold,
            "n_patients": len(last_preds),
            "auroc": roc_auc_score(y_true, y_prob),
            "auprc": average_precision_score(y_true, y_prob),
        }
        
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


# %%
if __name__ == "__main__":
    # Example usage
    print("Online Inference Module for Transformer Model (Optimized)")
    print("=" * 60)
    print("\nOptimizations:")
    print("  1. Batched inference: All timestamps for a patient processed together")
    print("  2. Cross-patient batching: Multiple patients' timestamps in one GPU batch")
    print("  3. PyTorch DataLoader with num_workers for parallel I/O")
    print("\nUsage:")
    print("  from src.scripts.online_inference import run_online_inference_dataloader")
    print("  ")
    print("  # RECOMMENDED: DataLoader-based inference (parallel data loading)")
    print("  results = run_online_inference_dataloader(")
    print("      lightning_module=lightning_module,")
    print("      master_table=master_table,")
    print("      data_dir=data_dir,")
    print("      scaling_info=scaling_info,")
    print("      batch_size=64,      # patients per batch")
    print("      num_workers=4,      # parallel data loading workers")
    print("      prefetch_factor=2,  # batches to prefetch")
    print("  )")
    print("  ")
    print("  # Alternative: Manual batching (no DataLoader)")
    print("  from src.scripts.online_inference import run_online_inference_batched_patients")
    print("  results = run_online_inference_batched_patients(")
    print("      lightning_module=lightning_module,")
    print("      master_table=master_table,")
    print("      data_dir=data_dir,")
    print("      scaling_info=scaling_info,")
    print("      batch_size=256,        # samples per GPU batch")
    print("      patient_batch_size=64, # patients to load at once")
    print("  )")
# %%
