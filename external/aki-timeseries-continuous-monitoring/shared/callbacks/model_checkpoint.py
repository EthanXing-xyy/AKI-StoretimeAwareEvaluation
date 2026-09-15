import os
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional


def get_check_point_callback(
    save_path: str,
    cv_num: Optional[int] = None,
    monitor: str = "val/loss",
    mode: str = "min",
    save_top_k: int = 3,
    save_last: bool = True,
) -> ModelCheckpoint:
    """Create model checkpoint callback."""
    os.makedirs(save_path, exist_ok=True)

    monitor_name = monitor.replace("/", "_")
    filename = f"model-{{epoch:03d}}-{{{monitor_name}:.4f}}"
    if cv_num is not None:
        filename = f"cv{cv_num}-" + filename

    return ModelCheckpoint(
        dirpath=f"{save_path}/cv_{cv_num}" if cv_num else save_path,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
        verbose=False,
        auto_insert_metric_name=False,
    )
