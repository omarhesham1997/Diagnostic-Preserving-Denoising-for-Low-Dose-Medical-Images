"""Classical baselines: no denoising, median filter, and Anscombe-BM3D.

Every method takes the noisy image y (NOT clipped) and the NoiseLevel, and returns
a denoised image clipped to [0, 1]. BM3D is given the true noise parameters
(an "oracle" setting); our network is also trained at the known noise level,
so the comparison is fair.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from c21.noise import NoiseLevel, gat_forward, gat_inverse_unbiased


def no_denoising(y: np.ndarray, level: NoiseLevel | None = None) -> np.ndarray:
    """The noisy input itself (clipped for display and scoring)."""
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def median_denoise(y: np.ndarray, level: NoiseLevel | None = None, size: int = 3) -> np.ndarray:
    """Median of each size x size neighbourhood."""
    return np.clip(median_filter(np.asarray(y, np.float32), size=size, mode="reflect"), 0.0, 1.0)


def bm3d_gat_denoise(y: np.ndarray, level: NoiseLevel) -> np.ndarray:
    """Anscombe-BM3D: stabilize the noise (std ~1), run BM3D, apply the unbiased inverse.

    BM3D expects values roughly in [0, 1], so the stabilized image is rescaled to
    [0, 1] and the noise std is rescaled with it.
    """
    import bm3d

    lam, sig = level.lambda_photons, level.sigma_electronic
    d = gat_forward(y, lam, sig)
    lo, hi = float(d.min()), float(d.max())
    scale = max(hi - lo, 1e-6)
    d_denoised = bm3d.bm3d((d - lo) / scale, sigma_psd=1.0 / scale) * scale + lo
    return np.clip(gat_inverse_unbiased(d_denoised, lam, sig), 0.0, 1.0).astype(np.float32)


# name -> (function, display name used in tables)
CLASSICAL_METHODS = {
    "noisy": (no_denoising, "Noisy (no denoising)"),
    "median": (median_denoise, "Median 3x3"),
    "bm3d": (bm3d_gat_denoise, "Anscombe-BM3D"),
}
