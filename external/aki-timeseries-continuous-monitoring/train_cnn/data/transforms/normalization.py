import pandas as pd
import torch


def normalize_feat_tensor(
    data: torch.Tensor,
    scaling_info: pd.DataFrame,
) -> torch.Tensor:
    """
    Optimized normalization function for PyTorch tensors.
    
    Args:
        data: Input tensor of shape (batch_size, time_steps, features)
        scaling_info: DataFrame with 'feature', 'median', and 'iqr' columns
        
    Returns:
        Normalized tensor of same shape as input
    """
    # Convert scaling info to tensors for vectorized operations
    # Ensure tensors are on the same device as data
    medians = torch.tensor(scaling_info['median'].values, dtype=data.dtype, device=data.device)
    iqrs = torch.tensor(scaling_info['iqr'].values, dtype=data.dtype, device=data.device)
    
    # Reshape for broadcasting: (features,) -> (1, 1, features)
    medians = medians.view(1, 1, -1)
    iqrs = iqrs.view(1, 1, -1)
    # Vectorized normalization
    normalized_data = (data - medians) / iqrs
    
    return normalized_data