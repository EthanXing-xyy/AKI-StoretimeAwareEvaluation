# %%
import numpy as np
import pandas as pd
import torch
import random

from typing import Tuple, List, Dict, Optional, Union
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

from src.data.transforms.normalization import normalize_feat_tensor

pd.set_option("display.max_columns", None)

# %%


def collate_fn_lstm_attention(
    batch: List[Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]],
    scaling_info: Optional[pd.DataFrame] = None
) -> Dict[str, Union[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Collate function for LSTM Attention model.
    
    Handles variable-length sequences by padding and creating masks.
    Uses the same data format as the Transformer model:
    - Input: [batch, seq_len, 3] where 3 = [feature_id, time, value]
    
    Args:
        batch: List of (data, label) tuples
        scaling_info: Optional DataFrame for normalization
        
    Returns:
        Dictionary with 'data', 'label', and 'mask' tensors
    """
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]

    if data[0].dim() == 2:
        # Data is (seq, feat)
        seq_lengths = torch.tensor([s.shape[0] for s in data])
        # Pad 2D data (seq, feat) -> (batch, max_seq, feat)
        data = pad_sequence(data, batch_first=True, padding_value=0.0)

    elif data[0].dim() == 3:
        # Data is (feat, seq, value)
        seq_lengths = torch.tensor([s.shape[1] for s in data])
        # (feat, seq, value) -> (seq, feat, value)
        data = [d.permute(1, 0, 2) for d in data]
        # Pad 3D data -> (batch, max_seq, feat, value)
        data = pad_sequence(data, batch_first=True, padding_value=0.0)

    else:
        # Fallback for other dimensions
        data = torch.stack(data)
        label = torch.stack(label).squeeze(1)
        return {"data": data, "label": label, "mask": None}

    max_len = data.shape[1]  # Sequence is now dimension 1
    mask = torch.arange(max_len).repeat(len(seq_lengths), 1) < seq_lengths.unsqueeze(1)

    # Apply normalization if scaling_info is provided (BEFORE masking)
    if scaling_info is not None:
        data = normalize_feat_tensor(data, scaling_info)

    if data.dim() == 3:
        # For (batch, max_seq, feat)
        expanded_mask = mask.unsqueeze(2).expand_as(data)
        data = data.masked_fill(~expanded_mask, float("nan"))
    elif data.dim() == 4:
        # For (batch, max_seq, feat, value)
        expanded_mask = mask.unsqueeze(2).unsqueeze(3).expand_as(data)
        data = data.masked_fill(~expanded_mask, float("nan"))

    label = torch.stack(label).squeeze(1)

    return {
        "data": torch.nan_to_num(data, nan=0.0),
        "label": label,
        "mask": mask,
    }

# %%

