from pytorch_lightning.callbacks import EarlyStopping


def get_early_stopping_callback(
    monitor: str = "val/loss",
    patience: int = 10,
    mode: str = "min",
    verbose: bool = False,
) -> EarlyStopping:
    """
    Get early stopping callback for training.

    Args:
        monitor: Metric to monitor for early stopping
        patience: Number of epochs with no improvement after which training will be stopped
        mode: One of 'min' or 'max'. In 'min' mode, training will stop when the quantity
              monitored has stopped decreasing; in 'max' mode it will stop when the quantity
              monitored has stopped increasing
        verbose: If True, prints a message for each validation loss improvement

    Returns:
        EarlyStopping callback
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        verbose=verbose,
    )
