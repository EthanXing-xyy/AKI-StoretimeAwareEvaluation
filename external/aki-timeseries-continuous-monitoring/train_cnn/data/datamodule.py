import joblib
import pandas as pd
import pytorch_lightning as pl
import torch

from functools import partial
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Callable, List, Union, Dict

from data.transforms.collate_functions import CollateFn1dCnnWithResolutionControl
from shared.data import get_sampler

pd.set_option("display.max_columns", None)

class AKI_Dataset(Dataset):
    
    def __init__(self, meta_table, data_dir):
        
        super().__init__()
        
        self.meta_table = (
            meta_table
            .assign(
                sex = lambda df: df["sex"].map({"M": 0, "F": 1})
            )
        )
        self.data_dir = data_dir
        
    def __len__(
        self
    ) -> int:
        return len(self.meta_table)
    
    def __getitem__(
        self,
        idx:int,
    ) -> Tuple[torch.FloatTensor]:
        
        visit_id = self.meta_table.loc[idx, 'visit_id']
        data_path = f"{self.data_dir}/{visit_id}.gz"
        data_dict = joblib.load(data_path)
        
        meta = torch.from_numpy(self.meta_table.loc[idx, ["age", "sex", "bmi"]].to_numpy(dtype=float)).type(torch.FloatTensor)
        data = torch.from_numpy(data_dict["data"]).type(torch.FloatTensor)
        label = torch.LongTensor([self.meta_table.loc[idx, "label"]])
        
        return (meta, data, label)


class AKIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        meta_table: pd.DataFrame,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 1004,
        resolution_control_method: Optional[str] = None,
        resolution: Optional[Union[float, List[float], Dict[str, float]]] = None,
        resolution_control_features: Optional[List[int]] = None,
        scaling_info: Optional[pd.DataFrame] = None,
        apply_prob: float = 1.0,
        all_features: Optional[List[str]] = None,
        seq_len: int = 264,
    ):
        super().__init__()
        
        self.meta_table = meta_table
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        collate_fn_train = CollateFn1dCnnWithResolutionControl(
            method=resolution_control_method,
            resolution=resolution,
            feature_indices=resolution_control_features,
            scaling_info=scaling_info,
            apply_prob=apply_prob,
            all_features=all_features,
            seq_len=seq_len,
        )
        
        collate_fn_default = CollateFn1dCnnWithResolutionControl(
            method=resolution_control_method,
            resolution=resolution,
            feature_indices=resolution_control_features,
            scaling_info=scaling_info,
            apply_prob=0.0,
            all_features=all_features,
            seq_len=seq_len,
        )
        
        self.train_collate_fn = collate_fn_train
        self.default_collate_fn = collate_fn_default
        
    def setup(self, stage: Optional[str] = None):
        # Split data into train, val, test
        train_val_meta, test_meta = train_test_split(
            self.meta_table,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=self.meta_table['label']
        )
        
        train_meta, val_meta = train_test_split(
            train_val_meta,
            test_size=self.val_split/(1-self.test_split),
            random_state=self.random_state,
            stratify=train_val_meta['label']
        )
        
        if stage == "fit" or stage is None:
            self.train_dataset = AKI_Dataset(train_meta.reset_index(drop=True), self.data_dir)
            self.val_dataset = AKI_Dataset(val_meta.reset_index(drop=True), self.data_dir)
            
        if stage == "test" or stage is None:
            self.test_dataset = AKI_Dataset(test_meta.reset_index(drop=True), self.data_dir)
            
        if stage == "predict" or stage is None:
            self.predict_dataset = AKI_Dataset(test_meta.reset_index(drop=True), self.data_dir)
            
    def train_dataloader(self):
        labels = self.train_dataset.meta_table['label'].to_numpy()
        sampler = get_sampler(labels)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            collate_fn=self.train_collate_fn,
            prefetch_factor=8,
            persistent_workers=True,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            collate_fn=self.default_collate_fn,
            prefetch_factor=8,
            persistent_workers=True,
            shuffle=False,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            collate_fn=self.default_collate_fn,
            prefetch_factor=8,
            persistent_workers=True,
            shuffle=False,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # pin_memory=True,
            collate_fn=self.default_collate_fn,
            prefetch_factor=8,
            persistent_workers=True,
        )
        
    def teardown(self, stage: Optional[str] = None):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

class TestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        meta_table: pd.DataFrame,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        resolution_control_method: Optional[str] = None,
        resolution: Optional[Union[float, List[float], Dict[str, float]]] = None,
        resolution_control_features: Optional[List[int]] = None,
        scaling_info: Optional[pd.DataFrame] = None,
        apply_prob: float = 0.0,
        all_features: Optional[List[str]] = None,
        seq_len: int = 264,
    ):
        super().__init__()
        
        self.meta_table = meta_table
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.default_collate_fn = CollateFn1dCnnWithResolutionControl(
            method=resolution_control_method,
            resolution=resolution,
            feature_indices=resolution_control_features,
            scaling_info=scaling_info,
            apply_prob=apply_prob,
            all_features=all_features,
            seq_len=seq_len,
        )
        
    def setup(self, stage: Optional[str] = None):
        
        if stage == "predict" or stage is None:
            self.dataset = AKI_Dataset(self.meta_table, self.data_dir)
        
    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # pin_memory=True,
            collate_fn=self.default_collate_fn
        )
    
    def teardown(self, stage: Optional[str] = None):
        self.dataset = None
