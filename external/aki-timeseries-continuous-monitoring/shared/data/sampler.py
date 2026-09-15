import numpy as np

from collections import Counter
from torch.utils.data import WeightedRandomSampler

def get_sampler(
    labels: np.ndarray,
) -> WeightedRandomSampler:

    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return (
        WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    )
