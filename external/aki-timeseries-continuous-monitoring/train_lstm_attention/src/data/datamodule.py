import joblib
import pandas as pd
import pytorch_lightning as pl
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

from functools import partial
from src.data.transforms.collate_functions import collate_fn_lstm_attention
from shared.data import get_sampler

pd.set_option("display.max_columns", None)

class AKI_Dataset(Dataset):
    
    def __init__(self, master_table, data_dir):
        
        super().__init__()
        
        self.master_table = master_table
        self.data_dir = data_dir
        
    def __len__(
        self
    ) -> int:
        return len(self.master_table)
    
    def __getitem__(
        self,
        idx:int,
    ) -> Tuple[torch.FloatTensor]:
        
        visit_id = self.master_table.loc[idx, 'visit_id']
        data_path = f"{self.data_dir}/{visit_id}.gz"
        data_dict = joblib.load(data_path)
        
        data = torch.from_numpy(data_dict["data"]).type(torch.FloatTensor)
        label = torch.LongTensor([self.master_table.loc[idx, "label"]])
        
        return (data, label)


class AKIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        master_table: pd.DataFrame,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 1004,
        scaling_info: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        
        self.master_table = master_table
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        # Use partial if scaling_info is provided, otherwise use default
        if scaling_info is not None:
            collate_fn = partial(collate_fn_lstm_attention, scaling_info=scaling_info)
            self.train_collate_fn = collate_fn
            self.default_collate_fn = collate_fn
        else:
            self.train_collate_fn = collate_fn_lstm_attention    
            self.default_collate_fn = collate_fn_lstm_attention
        
    def setup(self, stage: Optional[str] = None):
        # Split data into train, val, test
        train_val_meta, test_meta = train_test_split(
            self.master_table,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=self.master_table['label']
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
        labels = self.train_dataset.master_table['label'].to_numpy()
        sampler = get_sampler(labels)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
        master_table: pd.DataFrame,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        scaling_info: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        
        self.master_table = master_table
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Use partial if scaling_info is provided, otherwise use default
        if scaling_info is not None:
            self.default_collate_fn = partial(collate_fn_lstm_attention, scaling_info=scaling_info)
        else:
            self.default_collate_fn = collate_fn_lstm_attention
        
    def setup(self, stage: Optional[str] = None):
        
        if stage == "predict" or stage is None:
            self.dataset = AKI_Dataset(self.master_table, self.data_dir)
        
    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.default_collate_fn
        )
    
    def teardown(self, stage: Optional[str] = None):
        self.dataset = None

