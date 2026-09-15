# %%
import pytorch_lightning as pl
import torch

from typing import List
# %%

def inference_model(
    lightning_module:pl.LightningModule,
    datamodule:pl.LightningDataModule,
    prog_bar:pl.Callback,
    devices:List[int] = [0],
):
    
    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=devices,
        callbacks=[prog_bar],
        deterministic=True,
    )
    
    pred_results = trainer.predict(
        model=lightning_module,
        datamodule=datamodule,
    )
    
    return torch.cat(pred_results, dim=0)
# %%
