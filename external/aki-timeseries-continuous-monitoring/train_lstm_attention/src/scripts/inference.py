# %%
import pytorch_lightning as pl
import torch

from typing import List
# %%

def inference_model(
    lightning_module:pl.LightningModule,
    datamodule:pl.LightningDataModule,
    prog_bar:pl.Callback,
    config:dict,
):
    """
    Run inference on the test set.
    
    Args:
        lightning_module: Trained PyTorch Lightning module
        datamodule: Data module with test data
        prog_bar: Progress bar callback
        config: Configuration dictionary with device settings
        
    Returns:
        Tensor of predictions
    """
    # Determine devices
    if config['device'] == 'cpu':
        accelerator = 'cpu'
        devices = 1
    elif config['device'] == 'mps':
        accelerator = 'mps'
        devices = 1
    else:
        accelerator = 'gpu'
        devices = config['device'] if isinstance(config['device'], list) else [config['device']]
    
    trainer = pl.Trainer(
        logger=False,
        accelerator=accelerator,
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

