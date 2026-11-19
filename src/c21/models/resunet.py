"""Lightweight residual U-Net (ResU-Net) denoiser, as specified in the M1 report.

    x_hat = y - f(y)        the network f predicts the noise; we subtract it

Architecture (4 levels, channels 16 -> 32 -> 64 -> 128):
  - each level: two 3x3 convolutions, each followed by LeakyReLU(0.1); no batch norm
  - down: 2x2 max-pooling; up: bilinear x2 + 1x1 convolution, then concatenation
    with the encoder features of the same level (skip connection)
  - last layer: 1x1 convolution to one channel (the noise estimate), initialised to
    zero so the untrained network starts as the identity (x_hat = y)

Input height and width must be multiples of 8 (3 poolings); `denoise_numpy` pads
other sizes automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, slope: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Bilinear x2 upsampling + 1x1 conv, concatenate the skip features, then DoubleConv."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.reduce = nn.Conv2d(c_in, c_out, 1)
        self.conv = DoubleConv(2 * c_out, c_out)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, self.reduce(x)], dim=1))


class ResUNet(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        self.channels = tuple(channels)
        self.encoders = nn.ModuleList()
        c_prev = 1
        for c in self.channels:
            self.encoders.append(DoubleConv(c_prev, c))
            c_prev = c
        self.decoders = nn.ModuleList(
            Up(c_hi, c_lo) for c_hi, c_lo in zip(self.channels[:0:-1], self.channels[-2::-1])
        )
        self.head = nn.Conv2d(self.channels[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def predict_noise(self, y: torch.Tensor) -> torch.Tensor:
        skips = []
        x = y
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Noisy [B, 1, H, W] -> denoised [B, 1, H, W] (not clipped)."""
        return y - self.predict_noise(y)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Inference and checkpoints
# ---------------------------------------------------------------------------


@torch.no_grad()
def denoise_numpy(model: nn.Module, noisy: np.ndarray, device=None, batch_size: int = 16) -> np.ndarray:
    """Denoise a stack [N, H, W] (float, not clipped). Returns float32 in [0, 1]."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    n, h, w = noisy.shape
    pad_h, pad_w = (-h) % 8, (-w) % 8
    out = np.empty((n, h, w), dtype=np.float32)
    for start in range(0, n, batch_size):
        y = torch.as_tensor(np.asarray(noisy[start:start + batch_size], np.float32), device=device).unsqueeze(1)
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
        x_hat = model(y)[..., :h, :w]
        out[start:start + len(y)] = x_hat.clamp(0.0, 1.0).squeeze(1).cpu().numpy()
    model.train(was_training)
    return out


def save_checkpoint(path: Path, model: ResUNet, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "channels": list(model.channels),
                "parameters": count_parameters(model), **(extra or {})}, path)


def load_checkpoint(path: Path, device=None) -> tuple[ResUNet, dict]:
    ckpt = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    model = ResUNet(tuple(ckpt["channels"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device or "cpu").eval()
    return model, ckpt
