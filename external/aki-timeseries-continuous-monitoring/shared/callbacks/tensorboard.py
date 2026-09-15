import os
from pytorch_lightning.loggers import TensorBoardLogger


def get_tensor_board_logger(
    save_path: str,
    log_name: str,
) -> TensorBoardLogger:
    """Create TensorBoard logger."""
    os.makedirs(save_path, exist_ok=True)

    return TensorBoardLogger(
        save_dir=save_path,
        name=log_name,
    )
