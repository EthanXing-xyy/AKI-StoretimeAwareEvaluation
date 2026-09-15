# %%
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
    
    shutil.copyfile(
        check_point_callback.best_model_path,
        f"{'/'.join(check_point_callback.best_model_path.split('/')[:-1])}/model_best.ckpt"
    )
    
    return train_module
