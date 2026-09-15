# %%
import os
import pytorch_lightning as pl
import shutil
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
        precision="16-mixed",
        logger=tb_logger,
        max_epochs=config['max_epoch'],
        accelerator='cpu' if config['device'] == 'cpu' else ('mps' if config['device'] == 'mps' else 'gpu'),
        devices=1 if config['device'] in ['cpu', 'mps'] else config['device'],
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        log_every_n_steps=1, 
        accumulate_grad_batches=4,
        callbacks=callbacks, 
        deterministic=True,
        default_root_dir="../../../../result/logs",
        num_sanity_val_steps=0,
    )
    
    trainer.fit(
        train_module, 
        datamodule=datamodule
    )
    
    # Copy best model checkpoint
    best_model_path = check_point_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        # Get the directory where checkpoints are saved
        checkpoint_dir = os.path.dirname(best_model_path)
        target_path = os.path.join(checkpoint_dir, "model_best.ckpt")
        shutil.copyfile(best_model_path, target_path)
        print(f"Best model copied to: {target_path}")
    else:
        # Fallback: use last.ckpt if best_model_path is not available
        checkpoint_dir = check_point_callback.dirpath
        last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            target_path = os.path.join(checkpoint_dir, "model_best.ckpt")
            shutil.copyfile(last_ckpt, target_path)
            print(f"Using last.ckpt as model_best.ckpt: {target_path}")
        else:
            print(f"Warning: No checkpoint found to copy as model_best.ckpt")
    
    return train_module

