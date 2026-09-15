from pytorch_lightning.callbacks import Callback, RichProgressBar, TQDMProgressBar


def get_progress_bar(
    bar_type: str = "rich"
) -> Callback:

    if bar_type == "rich":
        bar = RichProgressBar()
    elif bar_type == "tqdm":
        bar = TQDMProgressBar()
    else:
        raise ValueError(f"Unknown progress bar type: {bar_type}. Use 'rich' or 'tqdm'.")

    return bar
