import pandas as pd
import torch


def normalize_feat_tensor(
    data: torch.Tensor,
    scaling_info: pd.DataFrame,
) -> torch.Tensor:
    """
    Optimized normalization function for PyTorch tensors with feature_id lookup.
    
    Supports two data formats:
    1. 3D (batch, seq, 3): Transformer format with columns [feature_id, time, value]
       - feature_id is stored as a value in column 0
    2. 4D (batch, seq, feat, value): Standard format where feat dimension = feature_id
       - feature_id is the dimension index
    
    Each feature is normalized using: (value - median[feature_id]) / iqr[feature_id]
    
    Args:
        data: Input tensor of shape (batch, seq, 3) or (batch, seq, feat, value)
        scaling_info: DataFrame with 'feature_id', 'median', and 'iqr' columns
        
    Returns:
        Normalized tensor of same shape as input
        
    Example (3D case):
        >>> # Data: [[feature_id, time, value]]
        >>> data = torch.tensor([[[3.0, 0.0, 150.0]]])  # sbp=150
        >>> scaling_info = pd.DataFrame({
        ...     'feature_id': [3], 'median': [120.0], 'iqr': [30.0]
        ... })
        >>> result = normalize_feat_tensor(data, scaling_info)
        >>> # result[0, 0, 2] = (150 - 120) / 30 = 1.0
    """
    # Create a lookup dictionary from feature_id to (median, iqr)
    feature_id_to_median = dict(zip(scaling_info['feature_id'].values, scaling_info['median'].values))
    feature_id_to_iqr = dict(zip(scaling_info['feature_id'].values, scaling_info['iqr'].values))
    
    # Get max feature_id to create lookup tensors
    max_feature_id = int(scaling_info['feature_id'].max()) + 1
    
    # Create lookup tensors (indexed by feature_id)
    median_lookup = torch.zeros(max_feature_id, dtype=data.dtype, device=data.device)
    iqr_lookup = torch.ones(max_feature_id, dtype=data.dtype, device=data.device)  # default to 1 to avoid division by zero
    
    for feat_id, median in feature_id_to_median.items():
        median_lookup[int(feat_id)] = median
    
    for feat_id, iqr in feature_id_to_iqr.items():
        if iqr != 0:
            iqr_lookup[int(feat_id)] = iqr
    
    if data.dim() == 3 and data.shape[2] == 3:
        # Case 1: (batch, seq, 3) - Transformer format with [feature_id, time, value]
        # Clone data to avoid modifying the original
        normalized_data = data.clone()
        
        # Extract feature_ids (column 0)
        feature_ids = data[..., 0].long()  # Shape: (batch, seq)
        
        # Look up medians and iqrs for each feature_id
        medians = median_lookup[feature_ids]  # Shape: (batch, seq)
        iqrs = iqr_lookup[feature_ids]  # Shape: (batch, seq)
        
        # Normalize only the value column (column 2)
        normalized_data[..., 2] = (data[..., 2] - medians) / iqrs
        
        return normalized_data
        
    elif data.dim() == 4:
        # Case 2: (batch, seq, feat, value) - Standard format where feat index = feature_id
        # The feat dimension index corresponds to feature_id
        # Reshape for broadcasting: (features,) -> (1, 1, features, 1)
        medians = median_lookup[:data.shape[2]].view(1, 1, -1, 1)
        iqrs = iqr_lookup[:data.shape[2]].view(1, 1, -1, 1)
        
        # Vectorized normalization
        normalized_data = (data - medians) / iqrs
        
        return normalized_data
        
    else:
        raise ValueError(f"Expected 3D tensor with shape (batch, seq, 3) or 4D tensor with shape (batch, seq, feat, value), got {data.shape}")