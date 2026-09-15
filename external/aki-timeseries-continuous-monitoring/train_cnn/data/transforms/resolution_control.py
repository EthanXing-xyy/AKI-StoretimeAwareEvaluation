import numpy as np
import torch


class ResolutionControlAugmentation:
    def __init__(self, method='round', resolution=0.1):
        """
        method: 'ceil', 'floor', or 'round'
        resolution: step size (e.g., 0.1, 0.001, 0.0001)
        """
        assert method in ['ceil', 'floor', 'round'], "method must be 'ceil', 'floor', or 'round'"
        assert resolution > 0, "resolution must be positive"
        self.method = method
        self.resolution = resolution

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return self._augment_tensor(x)
        elif isinstance(x, np.ndarray):
            return self._augment_numpy(x)
        else:
            raise TypeError("Input must be a torch.Tensor or np.ndarray")

    def _augment_tensor(self, x):
        scale = 1.0 / self.resolution
        if self.method == 'ceil':
            return torch.ceil(x * scale) / scale
        elif self.method == 'floor':
            return torch.floor(x * scale) / scale
        elif self.method == 'round':
            return torch.round(x * scale) / scale

    def _augment_numpy(self, x):
        scale = 1.0 / self.resolution
        if self.method == 'ceil':
            return np.ceil(x * scale) / scale
        elif self.method == 'floor':
            return np.floor(x * scale) / scale
        elif self.method == 'round':
            return np.round(x * scale) / scale
