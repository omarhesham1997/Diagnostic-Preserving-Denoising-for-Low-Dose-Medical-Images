"""
Noise simulation module for low-dose chest radiographs (NIH ChestX-ray14).
Implements compound Poisson-Gaussian noise simulation and generalized Anscombe
variance stabilization for classical filtering (BM3D).
"""

from __future__ import annotations
import numpy as np


# Standardized experimental noise configurations
NOISE_REGIMES = {
    "low": {"lambda_photons": 100.0, "sigma_electronic": 0.01, "desc": "Low noise / moderate dose (~30 dB PSNR)"},
    "medium": {"lambda_photons": 50.0, "sigma_electronic": 0.02, "desc": "Medium noise / standard low dose (~24 dB PSNR)"},
    "high": {"lambda_photons": 25.0, "sigma_electronic": 0.04, "desc": "High noise / ultra-low dose (~19 dB PSNR)"},
}


def add_poisson_gaussian_noise(
    image: np.ndarray,
    lambda_photons: float = 50.0,
    sigma_electronic: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """
    Corrupts a clean normalized image in [0, 1] with compound Poisson-Gaussian noise:
        y = Poisson(lambda * x) / lambda + Normal(0, sigma^2)

    Args:
        image: Clean input image array with pixel values in [0, 1].
        lambda_photons: Photon scaling parameter governing Poisson noise intensity.
                        Lower values simulate lower radiation dose.
        sigma_electronic: Standard deviation of additive Gaussian detector read noise.
        seed: Optional random seed for reproducible benchmark evaluation.

    Returns:
        Corrupted image array clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)
    img_clamped = np.clip(image, 0.0, 1.0).astype(np.float64)

    # Signal-dependent Poisson photon noise
    scaled_photons = img_clamped * lambda_photons
    poisson_component = rng.poisson(scaled_photons).astype(np.float64) / lambda_photons

    # Additive Gaussian electronic read noise
    gaussian_component = rng.normal(loc=0.0, scale=sigma_electronic, size=image.shape)

    noisy_image = poisson_component + gaussian_component
    return np.clip(noisy_image, 0.0, 1.0).astype(image.dtype)


def generalized_anscombe_transform(
    noisy_img: np.ndarray,
    sigma_electronic: float = 0.02,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Applies the generalized Anscombe transform for Poisson-Gaussian noise
    to stabilize the variance to approximately 1.0.

    Formula: f(y) = (2 / alpha) * sqrt(max(0, alpha * y + (3/8) * alpha^2 + sigma^2))
    """
    term = alpha * np.maximum(0.0, noisy_img) + (3.0 / 8.0) * (alpha ** 2) + (sigma_electronic ** 2)
    return (2.0 / alpha) * np.sqrt(np.maximum(0.0, term))


def generalized_anscombe_inverse(
    stabilized_img: np.ndarray,
    sigma_electronic: float = 0.02,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Closed-form algebraic inverse of the generalized Anscombe transform.
    """
    val = (stabilized_img * (alpha / 2.0)) ** 2 - (3.0 / 8.0) * (alpha ** 2) - (sigma_electronic ** 2)
    recovered = np.maximum(0.0, val) / alpha
    return np.clip(recovered, 0.0, 1.0)
