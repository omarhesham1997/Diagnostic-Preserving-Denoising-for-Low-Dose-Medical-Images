"""The frozen disease classifier ("the judge"): TorchXRayVision DenseNet-121.

It is never trained or changed. We only compare its predictions on clean and denoised
images. Its outputs are calibrated so that 0.5 is the model's own operating point for
every disease, which is why the agreement threshold is 0.5.

Note: the `densenet121-res224-all` weights were trained on several datasets including
NIH ChestX-ray14, so absolute AUCs on our test images are optimistic. We only use
differences (clean vs denoised), which this does not affect in the same way.
"""

from __future__ import annotations

import numpy as np

from c21.constants import NIH_LABELS


def to_xrv_input(images: np.ndarray, size: int = 224):
    """[N, H, W] images in [0, 1] -> torch tensor [N, 1, size, size] scaled to [-1024, 1024].

    Values outside [0, 1] (possible in denoiser outputs) are clipped first, so every
    method is judged on a valid image.
    """
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(np.clip(np.asarray(images, dtype=np.float32), 0.0, 1.0))
    x = x.unsqueeze(1)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x * 2048.0 - 1024.0


class FrozenClassifier:
    """Wraps the TorchXRayVision model and returns probabilities for the 14 NIH diseases."""

    def __init__(self, weights: str = "densenet121-res224-all", device=None,
                 input_size: int = 224, batch_size: int = 32):
        import torch
        import torchxrayvision as xrv

        from c21.utils import get_device

        self.device = device or get_device()
        self.input_size = input_size
        self.batch_size = batch_size
        self.weights = weights
        self.model = xrv.models.DenseNet(weights=weights).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.labels = list(NIH_LABELS)
        self.output_index = [list(self.model.pathologies).index(name) for name in self.labels]
        self._torch = torch

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Probabilities, shape [N, 14] (float32), columns in NIH_LABELS order."""
        torch = self._torch
        out = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = to_xrv_input(images[start:start + self.batch_size], self.input_size)
                probs = self.model(batch.to(self.device))[:, self.output_index]
                out.append(probs.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(self.labels)), np.float32)

    @classmethod
    def from_config(cls, cfg: dict, device=None) -> "FrozenClassifier":
        c = cfg["classifier"]
        return cls(weights=c["weights"], device=device, input_size=c["input_size"],
                   batch_size=c.get("batch_size", 32))
