"""Simulated low-dose X-ray noise and the Anscombe variance-stabilizing transform.

Noise model, for a clean image x with values in [0, 1]:

    y = Poisson(lam * x) / lam + Normal(0, sigma^2)

    lam   = photon count at full brightness (smaller = lower dose = more noise)
    sigma = electronic (detector read) noise, in [0, 1] intensity units

The noise variance is x / lam + sigma^2, so brighter pixels are noisier.
y is NOT clipped to [0, 1] by default: clipping would distort the noise
statistics that the Anscombe transform and BM3D rely on. Clip only final outputs.

Generalized Anscombe transform (GAT), Makitalo & Foi, IEEE TIP 22(1):91-103, 2013.
It works on photon counts z = lam * y, whose noise is Poisson + Normal(0, (lam*sigma)^2).
After the GAT the noise has standard deviation close to 1 at every brightness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Noise levels and seeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseLevel:
    name: str
    lambda_photons: float
    sigma_electronic: float


def get_level(cfg: dict, name: str | None = None) -> NoiseLevel:
    """Noise level from the config. `name` defaults to cfg['noise']['primary']."""
    name = name or cfg["noise"]["primary"]
    params = cfg["noise"]["levels"][name]
    return NoiseLevel(name, float(params["lambda_photons"]), float(params["sigma_electronic"]))


def all_levels(cfg: dict) -> list[NoiseLevel]:
    return [get_level(cfg, name) for name in cfg["noise"]["levels"]]


def image_seed(base: int, index: int) -> int:
    """Seed for one evaluation image: each image gets its own, repeatable noise."""
    return int(base) + int(index)


# ---------------------------------------------------------------------------
# Adding noise (NumPy for fixed evaluation sets, PyTorch for training batches)
# ---------------------------------------------------------------------------


def add_poisson_gaussian_noise(
    image: np.ndarray,
    lambda_photons: float,
    sigma_electronic: float,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    clip: bool = False,
) -> np.ndarray:
    """Return a noisy float32 copy of `image` (values in [0, 1]).

    Pass either `seed` (one fixed seed per image) or `rng` (a shared generator).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    x = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    photons = rng.poisson(lambda_photons * x).astype(np.float64)
    y = photons / lambda_photons + rng.normal(0.0, sigma_electronic, size=x.shape)
    if clip:
        y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


def add_noise(image: np.ndarray, level: NoiseLevel, seed: int | None = None,
              rng: np.random.Generator | None = None, clip: bool = False) -> np.ndarray:
    """Same as add_poisson_gaussian_noise, taking a NoiseLevel."""
    return add_poisson_gaussian_noise(image, level.lambda_photons, level.sigma_electronic,
                                      seed=seed, rng=rng, clip=clip)


def make_noisy_set(clean: np.ndarray, level: NoiseLevel, seed_base: int) -> np.ndarray:
    """Fixed noisy copy of a stack of images [N, H, W]: image i uses seed base + i.

    Every method is evaluated on exactly these images. Output is float32, not clipped.
    """
    return np.stack([add_noise(img, level, seed=image_seed(seed_base, i)) for i, img in enumerate(clean)])


def add_noise_torch(x, lambda_photons, sigma_electronic, generator=None):
    """PyTorch version for training: new random noise on every call, on CPU or GPU.

    `x` is a float tensor with values in [0, 1]. `lambda_photons` and
    `sigma_electronic` may be floats or tensors that broadcast against `x`
    (for example one value per image, shape [B, 1, 1, 1]).
    """
    import torch

    x = x.clamp(0.0, 1.0)
    photons = torch.poisson(x * lambda_photons, generator=generator)
    read_noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return photons / lambda_photons + sigma_electronic * read_noise


# ---------------------------------------------------------------------------
# Generalized Anscombe transform (for BM3D)
# ---------------------------------------------------------------------------


def gat_forward(y: np.ndarray, lambda_photons: float, sigma_electronic: float) -> np.ndarray:
    """Noisy image (intensity units) -> stabilized image with noise std ~ 1."""
    counts = lambda_photons * np.asarray(y, dtype=np.float64)
    sigma_counts = lambda_photons * sigma_electronic
    return 2.0 * np.sqrt(np.maximum(counts + 3.0 / 8.0 + sigma_counts**2, 0.0))


def gat_inverse_algebraic(d: np.ndarray, lambda_photons: float, sigma_electronic: float) -> np.ndarray:
    """Plain algebraic inverse. Biased: makes denoised images slightly too dark.

    Kept only for comparison; use gat_inverse_unbiased in experiments.
    """
    sigma_counts = lambda_photons * sigma_electronic
    counts = (np.asarray(d, dtype=np.float64) / 2.0) ** 2 - 3.0 / 8.0 - sigma_counts**2
    return np.maximum(counts, 0.0) / lambda_photons


def gat_inverse_unbiased(d: np.ndarray, lambda_photons: float, sigma_electronic: float) -> np.ndarray:
    """Closed-form approximation of the exact unbiased inverse (Makitalo & Foi 2013).

    Use this after denoising in the GAT domain. Output is in intensity units (not clipped).
    """
    sigma_counts = lambda_photons * sigma_electronic
    d = np.maximum(np.asarray(d, dtype=np.float64), 0.8)  # formula is only valid for d > ~0.8
    s = np.sqrt(1.5)
    counts = (0.25 * d**2 + 0.25 * s / d - 11.0 / 8.0 / d**2
              + 0.625 * s / d**3 - 0.125 - sigma_counts**2)
    return np.maximum(counts, 0.0) / lambda_photons
