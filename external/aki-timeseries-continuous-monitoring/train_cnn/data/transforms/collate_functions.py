# %%
import numpy as np
import pandas as pd
import torch
import random

from typing import Tuple, List, Dict, Optional, Union
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
# from torch.utils.data import default_collate

from data.transforms.resolution_control import ResolutionControlAugmentation
from data.transforms.normalization import normalize_feat_tensor

pd.set_option("display.max_columns", None)

# %%

class CollateFn1dCnnWithResolutionControl:
    def __init__(
        self,
        method: Optional[str] = None,
        resolution: Optional[Union[float, List[float], Dict[str, float]]] = None,
        feature_indices: Optional[List[int]] = None,
        scaling_info: Optional[pd.DataFrame] = None,
        apply_prob: float = 1.0,
        all_features: Optional[List[str]] = None,
        seq_len: int = 264,
    ):
        """
        A collate function that applies resolution control and normalization to specified features.
        This is designed for controlled experiments on feature precision.

        Args:
            method (str, optional): The rounding method: 'round', 'ceil', or 'floor'.
                                    If None, no resolution control is applied. Defaults to None.
            resolution (float, List[float], or Dict[str, float], optional): 
                The resolution for rounding.
                - float: applies to features in feature_indices.
                - List[float]: must correspond to feature_indices.
                - Dict[str, float]: keys are feature names, values are resolutions.
                If None, no resolution control is applied. Defaults to None.
            feature_indices (List[int], optional): A list of feature indices to apply resolution control to.
                                                   If None, it's applied to all features. Defaults to None.
            scaling_info (pd.DataFrame, optional): DataFrame with normalization parameters for time-series data.
                                                    Also includes 'age' and 'bmi' for demographic normalization.
                                                    If None, no normalization is applied. Defaults to None.
            apply_prob (float): The probability of applying resolution control. Defaults to 1.0.
            all_features (List[str], optional): List of all feature names. Required if resolution is a dict.
        
        Usage in DataLoader:
            collate_fn = CollateFn1dCnnWithResolutionControl(
                method='round', 
                resolution={'sbp': 10.0, 'dbp': 10.0, 'platelet': 0.1}, 
                all_features=feature_list,
                scaling_info=scaling_df,
                apply_prob=0.5,
            )
            data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        """
        self.augmentor = None
        self.augmentors = []
        self.feature_indices = feature_indices
        self.seq_len = seq_len
        
        if isinstance(resolution, dict):
            assert all_features is not None, "all_features must be provided if resolution is a dict"
            assert method is not None, "method must be provided if resolution is a dict"
            self.feature_indices = [all_features.index(feat) for feat in resolution.keys()]
            res_list = list(resolution.values())
            self.augmentors = [ResolutionControlAugmentation(method=method, resolution=r) for r in res_list]

        elif method is not None:
            assert method in ['ceil', 'floor', 'round'], "method must be 'ceil', 'floor', or 'round'"
            if isinstance(resolution, float):
                assert resolution > 0, "resolution must be a positive float"
                self.augmentor = ResolutionControlAugmentation(method=method, resolution=resolution)
            elif isinstance(resolution, list):
                if self.feature_indices is not None:
                    assert len(resolution) == len(self.feature_indices), \
                        "If resolution is a list, feature_indices must be a list of the same length."
                for r in resolution:
                    assert r > 0, "all resolutions must be positive floats"
                self.augmentors = [ResolutionControlAugmentation(method=method, resolution=r) for r in resolution]
            elif resolution is None:
                pass # self.augmentor will be None
            else:
                raise TypeError(f"resolution must be a float, a list of floats, a dict, or None, but got {type(resolution)}")
        
        self.apply_prob = apply_prob
        
        # Pre-compute meta normalization parameters from scaling_info
        # Meta features are [age, sex, bmi] at indices [0, 1, 2]
        # Only normalize age (idx 0) and bmi (idx 2), not sex (categorical)
        self.meta_median = None
        self.meta_iqr = None
        # Store time-series only scaling_info (exclude age, bmi for normalize_feat_tensor)
        self.ts_scaling_info = pd.DataFrame()
        
        if scaling_info is not None and not scaling_info.empty:
            # Create arrays for [age, sex, bmi] - sex will have median=0, iqr=1 (no change)
            meta_median = np.zeros(3)
            meta_iqr = np.ones(3)
            for _, row in scaling_info.iterrows():
                feat = row['feature']
                if feat == 'age':
                    meta_median[0] = row['median']
                    meta_iqr[0] = row['iqr']
                elif feat == 'bmi':
                    meta_median[2] = row['median']
                    meta_iqr[2] = row['iqr']
            self.meta_median = meta_median
            self.meta_iqr = meta_iqr
            
            # Filter out demographic features for time-series normalization
            self.ts_scaling_info = scaling_info[~scaling_info['feature'].isin(['age', 'bmi'])].reset_index(drop=True)

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, ...]]
    ) -> Dict[str, torch.Tensor]:
        
        # meta, data, label = [torch.stack(tensors) for tensors in zip(*batch)]
        meta, data, label = zip(*batch)
        meta = torch.stack(meta)
        data = pad_sequence(data, batch_first=True, padding_value=torch.nan, padding_side="left")
        
        label = torch.stack(label)
        
        # Apply demographic normalization (age, bmi) if meta_scaling_info is provided
        if self.meta_median is not None and self.meta_iqr is not None:
            meta_median_tensor = torch.tensor(self.meta_median, dtype=meta.dtype, device=meta.device)
            meta_iqr_tensor = torch.tensor(self.meta_iqr, dtype=meta.dtype, device=meta.device)
            meta = (meta - meta_median_tensor) / meta_iqr_tensor

        # Apply resolution control if augmentor is available
        if random.random() < self.apply_prob:
            if self.augmentor: # single resolution case
                if self.feature_indices:
                    # Apply to specific features
                    # data shape: (batch_size, time_step, feature)
                    # Select specific features: (batch_size, time_step, selected_features)
                    selected_data = data[:, :, self.feature_indices]
                    # Apply augmentation: (batch_size, time_step, selected_features)
                    augmented_data = self.augmentor(selected_data)
                    # Assign back to original tensor
                    data[:, :, self.feature_indices] = augmented_data
                else:
                    # Apply to all features
                    data = self.augmentor(data)
            elif self.augmentors: # multiple resolutions case
                assert self.feature_indices is not None, "feature_indices must be provided when resolution is a list or dict"
                for i, feature_idx in enumerate(self.feature_indices):
                    augmentor = self.augmentors[i]
                    # Select single feature: (batch_size, time_step, 1)
                    feature_data = data[:, :, feature_idx:feature_idx+1]
                    # Apply augmentation: (batch_size, time_step, 1)
                    augmented_feature = augmentor(feature_data.clone())
                    # Assign back to original tensor
                    data[:, :, feature_idx:feature_idx+1] = augmented_feature
                
        # Apply normalization if scaling_info is provided (time-series features only)
        if not self.ts_scaling_info.empty:
            data = normalize_feat_tensor(data, self.ts_scaling_info)
            
        data = torch.nan_to_num(data)
            
        # Pad sequence to length 264 along the time dimension (dim=1)
        # pad takes (pad_left, pad_right, pad_top, pad_bottom, ...) for each dimension, starting from the last
        # For 3D tensor (batch, time, feature), pad=(0,0,0,0,0,N) pads N zeros at the end of dim=1 (time)
        pad_len = self.seq_len - data.shape[1]
        if pad_len > 0:
            # pad format: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
            # For (batch, time, feature): pad=(0,0,0,pad_len)
            data = pad(data, (0, 0, pad_len, 0), "constant", 0)

        return {
            "meta": meta,
            "data": data,
            "label": label.squeeze(1),
        }


def collate_fn_1d_cnn(
    batch: List[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor]]
) -> Dict[str, torch.Tensor]:

    # Unpack and stack tensors in one go
    meta, data, label = [torch.stack(tensors) for tensors in zip(*batch)]

    return {
        "meta": meta,
        "data": data,
        "label": label.squeeze(1),
    }

def collate_fn_1d_cnn_with_augmentation(batch, apply_prob=0.5):
    
    # Unpack and stack tensors in one go
    meta, data, label = [torch.stack(tensors) for tensors in zip(*batch)]
    
    batch = {
        "meta": meta,
        "data": data,
        "label": label.squeeze(1),
    }

# apply 여부를 랜덤 결정
    if random.random() < apply_prob:
        # method도 랜덤으로 선택
        method = random.choice(['round', 'ceil', 'floor'])
        # resolution도 랜덤으로 선택
        resolution = random.choice([0.1, 0.01, 0.001, 0.0001])

        augmentor = ResolutionControlAugmentation(method=method, resolution=resolution)

        if isinstance(batch, (tuple, list)):
            inputs, labels = batch
            inputs = augmentor(inputs)
            return inputs, labels

        elif isinstance(batch, dict):
            batch['data'] = augmentor(batch['data'])
            return batch

        else:
            raise TypeError("Unsupported batch type in collate_fn.")

    else:
        # 적용 안하고 그대로 리턴
        return batch
    
