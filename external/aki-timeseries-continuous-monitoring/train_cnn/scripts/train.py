# %%
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shutil
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.callbacks import get_check_point_callback, get_progress_bar, get_tensor_board_logger
from shared.utils import set_seed
from data.datamodule import AKIDataModule
from lightning_modules.classifier_module import AKI_Simple_TrainModule
# %%
def train_model(
    train_module:pl.LightningModule,
    datamodule:pl.LightningDataModule,
    config:dict,
    tb_logger:pl.Callback,
    prog_bar:pl.Callback,
    check_point_callback:pl.Callback,
    early_stopping_callback:pl.Callback = None,
) -> tuple:
    
    # Prepare callbacks list
    callbacks = [prog_bar, check_point_callback]
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)
    
    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=config['max_epoch'],
        accelerator='mps' if config['device'] == 'mps' else 'gpu', 
        devices=1 if config['device'] == 'mps' else config['device'], 
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        log_every_n_steps=1, 
        accumulate_grad_batches=1,
        callbacks=callbacks, 
        deterministic=True,
        default_root_dir="../../../../result/logs",
    )
    
    trainer.fit(
        train_module, 
        datamodule=datamodule
    )
    
    shutil.copyfile(
        check_point_callback.best_model_path,
        f"{'/'.join(check_point_callback.best_model_path.split('/')[:-1])}/model_best.ckpt"
    )
    
    return train_module
