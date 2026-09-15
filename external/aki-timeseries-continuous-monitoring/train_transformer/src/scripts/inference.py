# %%
import pytorch_lightning as pl
import torch

from typing import List, Dict, Any
# %%

def inference_model(
    lightning_module:pl.LightningModule,
    datamodule:pl.LightningDataModule,
    prog_bar:pl.Callback,
    config:Dict[str, Any],
):
    
    trainer = pl.Trainer(
        precision="16-mixed",
        logger=False,
        accelerator="cpu" if config['device'] == 'cpu' else ('mps' if config['device'] == 'mps' else 'gpu'),
        devices=1 if config['device'] in ['cpu', 'mps'] else config['device'],
        callbacks=[prog_bar],
        deterministic=True,
        num_sanity_val_steps=0,
    )
    
    pred_results = trainer.predict(
        model=lightning_module,
        datamodule=datamodule,
    )
    
    return torch.cat(pred_results, dim=0)
# %%
